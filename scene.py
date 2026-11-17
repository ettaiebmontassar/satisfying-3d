"""Rendu Blender (Cycles, CPU) d'une vidéo « How many cuts ».

blender -b -P scene.py -- objects/steak.json OUT_DIR [--frames A B] [--res 1080] [--samples 24] [--step 1]
Écrit OUT_DIR/f00000.png … (images 3D sans texte ; les compteurs sont ajoutés par overlay.py).
"""
import math, os, sys, time
import bpy, bmesh
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import core

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
CFG_PATH, OUT = argv[0], argv[1]
def opt(name, default, cast=float, n=1):
    if name in argv:
        i = argv.index(name)
        v = [cast(x) for x in argv[i + 1:i + 1 + n]]
        return v if n > 1 else v[0]
    return default
RES = opt("--res", 1080, int)
SAMPLES = opt("--samples", 24, int)
DENOISE = "--no-denoise" not in argv
MBLUR = "--no-mblur" not in argv
STEP = opt("--step", 1, int)

cfg = core.load(CFG_PATH)
PLAN = core.Plan(cfg)
SH = PLAN.shape
F0, F1 = opt("--frames", [0, PLAN.frames], int, 2)
os.makedirs(OUT, exist_ok=True)


# ================================================================ outils
def mat(name):
    m = bpy.data.materials.new(name); m.use_nodes = True
    nt = m.node_tree; nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial"); bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    nt.links.new(bsdf.outputs[0], out.inputs[0])
    return m, nt, bsdf


def ramp(nt, src, stops):
    r = nt.nodes.new("ShaderNodeValToRGB")
    els = r.color_ramp.elements
    while len(els) > len(stops):
        els.remove(els[-1])
    while len(els) < len(stops):
        els.new(0.5)
    for e, (pos, col) in zip(els, stops):
        e.position = pos; e.color = (*col, 1)
    nt.links.new(src, r.inputs[0])
    return r


def noise(nt, vec, scale, detail=6, rough=0.55, dist=0.0):
    n = nt.nodes.new("ShaderNodeTexNoise"); n.inputs["Scale"].default_value = scale
    n.inputs["Detail"].default_value = detail; n.inputs["Roughness"].default_value = rough
    n.inputs["Distortion"].default_value = dist
    if vec is not None:
        nt.links.new(vec, n.inputs["Vector"])
    return n


# ================================================================ matériaux
def coords(nt, coord_mode):
    """Coordonnées dans l'objet entier : directes (objet) ou reconstituées pour chaque cube (instances)."""
    if coord_mode == "object":
        tc = nt.nodes.new("ShaderNodeTexCoord"); return tc.outputs["Object"]
    at = nt.nodes.new("ShaderNodeAttribute"); at.attribute_type = "INSTANCER"; at.attribute_name = "orig"
    sz = nt.nodes.new("ShaderNodeAttribute"); sz.attribute_type = "INSTANCER"; sz.attribute_name = "scl"
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mul = nt.nodes.new("ShaderNodeVectorMath"); mul.operation = "MULTIPLY"
    nt.links.new(tc.outputs["Object"], mul.inputs[0]); nt.links.new(sz.outputs["Vector"], mul.inputs[1])
    add = nt.nodes.new("ShaderNodeVectorMath"); add.operation = "ADD"
    nt.links.new(at.outputs["Vector"], add.inputs[0]); nt.links.new(mul.outputs[0], add.inputs[1])
    return add.outputs[0]


def mixc(nt, fac, a, b):
    """Mélange de couleurs ; a, b = sorties de nœuds ou tuples RGB."""
    mx = nt.nodes.new("ShaderNodeMix"); mx.data_type = "RGBA"
    nt.links.new(fac, mx.inputs["Factor"])
    for sock, v in ((mx.inputs[6], a), (mx.inputs[7], b)):
        if isinstance(v, tuple):
            sock.default_value = (*v, 1)
        else:
            nt.links.new(v, sock)
    return mx.outputs[2]


def watermelon_material(name, coord_mode):
    """Pastèque : écorce rayée vert foncé / vert clair ; dans les cubes, écorce → blanc → chair rouge à pépins."""
    m, nt, bsdf = mat(name)
    vec = coords(nt, coord_mode)
    mp = nt.nodes.new("ShaderNodeMapping"); mp.inputs["Scale"].default_value = (1.0, 1.0, 1.0)
    nt.links.new(vec, mp.inputs[0])
    wave = nt.nodes.new("ShaderNodeTexWave"); wave.bands_direction = "Y"; wave.inputs["Scale"].default_value = 9.0
    wave.inputs["Distortion"].default_value = 9.0; wave.inputs["Detail"].default_value = 3.0
    nt.links.new(mp.outputs[0], wave.inputs[0])
    stripes = ramp(nt, wave.outputs["Fac"], [(0.35, (0.012, 0.07, 0.015)), (0.55, (0.09, 0.26, 0.05))])
    rind = stripes.outputs["Color"]
    if coord_mode == "object":
        col = cut_lines(nt, rind, (0.3, 0.02, 0.03))
        bsdf.inputs["Roughness"].default_value = 0.32; bsdf.inputs["Coat Weight"].default_value = 0.3
    else:
        dep = nt.nodes.new("ShaderNodeAttribute"); dep.attribute_type = "INSTANCER"; dep.attribute_name = "depth"
        k = nt.nodes.new("ShaderNodeMath"); k.operation = "MULTIPLY"; k.inputs[1].default_value = 33.3   # 3 cm → 1
        nt.links.new(dep.outputs["Fac"], k.inputs[0])
        flesh = ramp(nt, noise(nt, vec, 8.0, 3).outputs["Fac"], [(0.3, (0.72, 0.03, 0.05)), (0.7, (0.95, 0.12, 0.12))])
        layers = ramp(nt, k.outputs[0], [(0.0, (0, 0, 0)), (0.13, (0, 0, 0)), (0.17, (0.5, 0.5, 0.5)), (0.3, (0.5, 0.5, 0.5)), (0.38, (1, 1, 1))])
        # 0 = écorce, 0.5 = blanc, 1 = chair
        lo = nt.nodes.new("ShaderNodeMath"); lo.operation = "MULTIPLY"; lo.inputs[1].default_value = 2.0; lo.use_clamp = True
        nt.links.new(layers.outputs["Color"], lo.inputs[0])
        hi = nt.nodes.new("ShaderNodeMath"); hi.operation = "MULTIPLY_ADD"; hi.inputs[1].default_value = 2.0; hi.inputs[2].default_value = -1.0
        hi.use_clamp = True; nt.links.new(layers.outputs["Color"], hi.inputs[0])
        col = mixc(nt, lo.outputs[0], rind, (0.86, 0.9, 0.7))
        col = mixc(nt, hi.outputs[0], col, flesh.outputs["Color"])
        vor = nt.nodes.new("ShaderNodeTexVoronoi"); vor.inputs["Scale"].default_value = 38.0
        nt.links.new(vec, vor.inputs["Vector"])
        seed = nt.nodes.new("ShaderNodeMath"); seed.operation = "LESS_THAN"; seed.inputs[1].default_value = 0.1
        nt.links.new(vor.outputs["Distance"], seed.inputs[0])
        deep = nt.nodes.new("ShaderNodeMath"); deep.operation = "GREATER_THAN"; deep.inputs[1].default_value = 0.6
        nt.links.new(k.outputs[0], deep.inputs[0])
        both = nt.nodes.new("ShaderNodeMath"); both.operation = "MULTIPLY"
        nt.links.new(seed.outputs[0], both.inputs[0]); nt.links.new(deep.outputs[0], both.inputs[1])
        col = mixc(nt, both.outputs[0], col, (0.02, 0.015, 0.012))
        bsdf.inputs["Roughness"].default_value = 0.28; bsdf.inputs["Coat Weight"].default_value = 0.25
    nt.links.new(col, bsdf.inputs["Base Color"])
    return m


def cake_material(name, coord_mode):
    """Gâteau au chocolat : glaçage brillant dehors ; dedans, génoise et crème en couches (selon la hauteur)."""
    m, nt, bsdf = mat(name)
    vec = coords(nt, coord_mode)
    glaze = ramp(nt, noise(nt, vec, 30.0, 3).outputs["Fac"], [(0.4, (0.035, 0.014, 0.007)), (0.6, (0.06, 0.025, 0.012))]).outputs["Color"]
    if coord_mode == "object":
        col = cut_lines(nt, glaze, (0.55, 0.38, 0.24))
        bsdf.inputs["Roughness"].default_value = 0.3; bsdf.inputs["Coat Weight"].default_value = 0.2
        nt.links.new(col, bsdf.inputs["Base Color"])
        return m
    sep = nt.nodes.new("ShaderNodeSeparateXYZ"); nt.links.new(vec, sep.inputs[0])
    # couches selon la hauteur z (m) : génoise / crème / génoise / crème / génoise
    lay = nt.nodes.new("ShaderNodeValToRGB"); lay.color_ramp.interpolation = "CONSTANT"
    els = lay.color_ramp.elements
    els[0].position = 0.0; els[0].color = (0, 0, 0, 1)
    els[1].position = 0.27; els[1].color = (1, 1, 1, 1)
    for pos, v in ((0.36, 0), (0.62, 1), (0.71, 0)):
        e = els.new(pos); e.color = (v, v, v, 1)          # 0 = génoise, 1 = crème
    sc = nt.nodes.new("ShaderNodeMath"); sc.operation = "MULTIPLY"; sc.inputs[1].default_value = 1 / 0.09
    nt.links.new(sep.outputs[2], sc.inputs[0]); nt.links.new(sc.outputs[0], lay.inputs[0])
    sponge = ramp(nt, noise(nt, vec, 140.0, 2, 0.6).outputs["Fac"], [(0.35, (0.06, 0.022, 0.01)), (0.65, (0.16, 0.065, 0.028))]).outputs["Color"]
    cream = ramp(nt, noise(nt, vec, 40.0, 2).outputs["Fac"], [(0.4, (0.72, 0.56, 0.38)), (0.6, (0.82, 0.68, 0.5))]).outputs["Color"]
    inside = mixc(nt, lay.outputs["Color"], sponge, cream)
    dep = nt.nodes.new("ShaderNodeAttribute"); dep.attribute_type = "INSTANCER"; dep.attribute_name = "depth"
    outer = nt.nodes.new("ShaderNodeMath"); outer.operation = "LESS_THAN"; outer.inputs[1].default_value = 0.004
    nt.links.new(dep.outputs["Fac"], outer.inputs[0])
    col = mixc(nt, outer.outputs[0], inside, glaze)
    rough = nt.nodes.new("ShaderNodeMix"); rough.data_type = "FLOAT"
    nt.links.new(outer.outputs[0], rough.inputs["Factor"]); rough.inputs[2].default_value = 0.75; rough.inputs[3].default_value = 0.3
    nt.links.new(rough.outputs[0], bsdf.inputs["Roughness"])
    nt.links.new(col, bsdf.inputs["Base Color"])
    return m


def orange_material(name, coord_mode):
    """Orange : écorce piquetée ; dans les cubes, écorce → peau blanche → pulpe en quartiers (membranes claires)."""
    m, nt, bsdf = mat(name)
    vec = coords(nt, coord_mode)
    peel = ramp(nt, noise(nt, vec, 25.0, 4).outputs["Fac"], [(0.35, (0.75, 0.13, 0.0)), (0.65, (0.92, 0.22, 0.0))]).outputs["Color"]
    pores = nt.nodes.new("ShaderNodeTexVoronoi"); pores.inputs["Scale"].default_value = 420.0
    nt.links.new(vec, pores.inputs["Vector"])
    if coord_mode == "object":
        col = cut_lines(nt, peel, (1.0, 0.72, 0.38))
        bump = nt.nodes.new("ShaderNodeBump"); bump.inputs["Strength"].default_value = 0.12
        nt.links.new(pores.outputs["Distance"], bump.inputs["Height"]); nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
        bsdf.inputs["Roughness"].default_value = 0.22; bsdf.inputs["Coat Weight"].default_value = 0.45
        nt.links.new(col, bsdf.inputs["Base Color"])
        return m
    sep = nt.nodes.new("ShaderNodeSeparateXYZ"); nt.links.new(vec, sep.inputs[0])
    ang = nt.nodes.new("ShaderNodeMath"); ang.operation = "ARCTAN2"
    nt.links.new(sep.outputs[1], ang.inputs[0]); nt.links.new(sep.outputs[0], ang.inputs[1])
    seg = nt.nodes.new("ShaderNodeMath"); seg.operation = "MULTIPLY"; seg.inputs[1].default_value = 11 / (2 * math.pi)
    nt.links.new(ang.outputs[0], seg.inputs[0])
    pp = nt.nodes.new("ShaderNodeMath"); pp.operation = "PINGPONG"; pp.inputs[1].default_value = 0.5
    nt.links.new(seg.outputs[0], pp.inputs[0])                      # 0 = sur une membrane entre deux quartiers
    memb = nt.nodes.new("ShaderNodeMath"); memb.operation = "LESS_THAN"; memb.inputs[1].default_value = 0.05
    nt.links.new(pp.outputs[0], memb.inputs[0])
    juice = nt.nodes.new("ShaderNodeTexVoronoi"); juice.inputs["Scale"].default_value = 70.0
    nt.links.new(vec, juice.inputs["Vector"])
    flesh = ramp(nt, juice.outputs["Distance"], [(0.0, (1.0, 0.36, 0.0)), (0.45, (0.85, 0.17, 0.0))]).outputs["Color"]
    flesh = mixc(nt, memb.outputs[0], flesh, (1.0, 0.62, 0.3))
    dep = nt.nodes.new("ShaderNodeAttribute"); dep.attribute_type = "INSTANCER"; dep.attribute_name = "depth"
    k = nt.nodes.new("ShaderNodeMath"); k.operation = "MULTIPLY"; k.inputs[1].default_value = 1 / 0.012
    nt.links.new(dep.outputs["Fac"], k.inputs[0])
    lay = ramp(nt, k.outputs[0], [(0.0, (0, 0, 0)), (0.3, (0, 0, 0)), (0.38, (0.5, 0.5, 0.5)), (0.62, (0.5, 0.5, 0.5)), (0.7, (1, 1, 1))])
    lo = nt.nodes.new("ShaderNodeMath"); lo.operation = "MULTIPLY"; lo.inputs[1].default_value = 2.0; lo.use_clamp = True
    nt.links.new(lay.outputs["Color"], lo.inputs[0])
    hi = nt.nodes.new("ShaderNodeMath"); hi.operation = "MULTIPLY_ADD"; hi.inputs[1].default_value = 2.0; hi.inputs[2].default_value = -1.0
    hi.use_clamp = True; nt.links.new(lay.outputs["Color"], hi.inputs[0])
    col = mixc(nt, lo.outputs[0], peel, (0.95, 0.85, 0.62))
    col = mixc(nt, hi.outputs[0], col, flesh)
    bsdf.inputs["Roughness"].default_value = 0.25; bsdf.inputs["Coat Weight"].default_value = 0.35
    nt.links.new(col, bsdf.inputs["Base Color"])
    return m


def pizza_material(name, coord_mode, radius=0.13):
    """Pizza : croûte dorée sur le bord, fromage fondu doré + pepperoni dessus ; pâte claire à l'intérieur des cubes."""
    m, nt, bsdf = mat(name)
    vec = coords(nt, coord_mode)
    flat = nt.nodes.new("ShaderNodeVectorMath"); flat.operation = "MULTIPLY"; flat.inputs[1].default_value = (1, 1, 0)
    nt.links.new(vec, flat.inputs[0])
    ln = nt.nodes.new("ShaderNodeVectorMath"); ln.operation = "LENGTH"; nt.links.new(flat.outputs[0], ln.inputs[0])
    rr = nt.nodes.new("ShaderNodeMath"); rr.operation = "DIVIDE"; rr.inputs[1].default_value = radius
    nt.links.new(ln.outputs["Value"], rr.inputs[0])
    crust_m = nt.nodes.new("ShaderNodeMath"); crust_m.operation = "GREATER_THAN"; crust_m.inputs[1].default_value = 0.84
    nt.links.new(rr.outputs[0], crust_m.inputs[0])
    crust = ramp(nt, noise(nt, vec, 60.0, 4).outputs["Fac"], [(0.3, (0.25, 0.08, 0.01)), (0.55, (0.5, 0.22, 0.04)), (0.75, (0.62, 0.34, 0.1))]).outputs["Color"]
    cheese = ramp(nt, noise(nt, vec, 45.0, 5, 0.6, 0.8).outputs["Fac"], [(0.3, (0.55, 0.22, 0.01)), (0.6, (0.7, 0.36, 0.03)), (0.78, (0.38, 0.12, 0.01))]).outputs["Color"]
    pep_v = nt.nodes.new("ShaderNodeTexVoronoi"); pep_v.inputs["Scale"].default_value = 17.0; pep_v.inputs["Randomness"].default_value = 0.55
    nt.links.new(flat.outputs[0], pep_v.inputs["Vector"])
    pep = nt.nodes.new("ShaderNodeMath"); pep.operation = "LESS_THAN"; pep.inputs[1].default_value = 0.3
    nt.links.new(pep_v.outputs["Distance"], pep.inputs[0])
    pep_col = ramp(nt, noise(nt, vec, 80.0, 3).outputs["Fac"], [(0.35, (0.32, 0.015, 0.01)), (0.65, (0.55, 0.035, 0.02))]).outputs["Color"]
    top = mixc(nt, pep.outputs[0], cheese, pep_col)
    top = mixc(nt, crust_m.outputs[0], top, crust)
    if coord_mode == "object":
        col = cut_lines(nt, top, (0.98, 0.9, 0.7))
        bsdf.inputs["Roughness"].default_value = 0.55
    else:
        # dans chaque cube : garniture sur la partie haute (z > 1 cm), pâte en dessous ; croûte dorée sur le bord extérieur
        dough = ramp(nt, noise(nt, vec, 120.0, 3).outputs["Fac"], [(0.35, (0.58, 0.38, 0.16)), (0.65, (0.7, 0.5, 0.25))]).outputs["Color"]
        sep = nt.nodes.new("ShaderNodeSeparateXYZ"); nt.links.new(vec, sep.inputs[0])
        high = nt.nodes.new("ShaderNodeMath"); high.operation = "GREATER_THAN"; high.inputs[1].default_value = 0.0105
        nt.links.new(sep.outputs[2], high.inputs[0])
        inner = nt.nodes.new("ShaderNodeMath"); inner.operation = "SUBTRACT"; inner.inputs[0].default_value = 1.0
        nt.links.new(crust_m.outputs[0], inner.inputs[1])
        topping = nt.nodes.new("ShaderNodeMath"); topping.operation = "MULTIPLY"
        nt.links.new(high.outputs[0], topping.inputs[0]); nt.links.new(inner.outputs[0], topping.inputs[1])
        dep = nt.nodes.new("ShaderNodeAttribute"); dep.attribute_type = "INSTANCER"; dep.attribute_name = "depth"
        shell = nt.nodes.new("ShaderNodeMath"); shell.operation = "LESS_THAN"; shell.inputs[1].default_value = 0.004
        nt.links.new(dep.outputs["Fac"], shell.inputs[0])
        rimc = nt.nodes.new("ShaderNodeMath"); rimc.operation = "MULTIPLY"
        nt.links.new(shell.outputs[0], rimc.inputs[0]); nt.links.new(crust_m.outputs[0], rimc.inputs[1])
        col = mixc(nt, topping.outputs[0], dough, top)
        col = mixc(nt, rimc.outputs[0], col, crust)
        bsdf.inputs["Roughness"].default_value = 0.55
    nt.links.new(col, bsdf.inputs["Base Color"])
    return m


def avocado_material(name, coord_mode, pit=(0.025, 0.0, 0.012), pit_r=0.038):
    """Avocat : peau sombre granuleuse ; chair vert vif sous la peau → jaune pâle au cœur ; gros noyau brun."""
    m, nt, bsdf = mat(name)
    vec = coords(nt, coord_mode)
    skin = ramp(nt, noise(nt, vec, 40.0, 4).outputs["Fac"], [(0.35, (0.012, 0.03, 0.008)), (0.65, (0.05, 0.09, 0.02))]).outputs["Color"]
    bumps = nt.nodes.new("ShaderNodeTexVoronoi"); bumps.inputs["Scale"].default_value = 180.0
    nt.links.new(vec, bumps.inputs["Vector"])
    if coord_mode == "object":
        col = cut_lines(nt, skin, (0.7, 0.8, 0.25))
        bump = nt.nodes.new("ShaderNodeBump"); bump.inputs["Strength"].default_value = 0.5
        nt.links.new(bumps.outputs["Distance"], bump.inputs["Height"]); nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
        bsdf.inputs["Roughness"].default_value = 0.45; bsdf.inputs["Coat Weight"].default_value = 0.2
        nt.links.new(col, bsdf.inputs["Base Color"])
        return m
    dep = nt.nodes.new("ShaderNodeAttribute"); dep.attribute_type = "INSTANCER"; dep.attribute_name = "depth"
    k = nt.nodes.new("ShaderNodeMath"); k.operation = "MULTIPLY"; k.inputs[1].default_value = 1 / 0.02
    nt.links.new(dep.outputs["Fac"], k.inputs[0])
    flesh = ramp(nt, k.outputs[0], [(0.0, (0.012, 0.03, 0.008)), (0.1, (0.012, 0.03, 0.008)), (0.16, (0.08, 0.34, 0.015)),
                                    (0.6, (0.3, 0.55, 0.04)), (1.0, (0.6, 0.66, 0.14))]).outputs["Color"]
    d = nt.nodes.new("ShaderNodeVectorMath"); d.operation = "DISTANCE"; d.inputs[1].default_value = pit
    nt.links.new(vec, d.inputs[0])
    inpit = nt.nodes.new("ShaderNodeMath"); inpit.operation = "LESS_THAN"; inpit.inputs[1].default_value = pit_r
    nt.links.new(d.outputs["Value"], inpit.inputs[0])
    pitc = ramp(nt, noise(nt, vec, 60.0, 3).outputs["Fac"], [(0.35, (0.2, 0.08, 0.025)), (0.65, (0.36, 0.16, 0.05))]).outputs["Color"]
    col = mixc(nt, inpit.outputs[0], flesh, pitc)
    bsdf.inputs["Roughness"].default_value = 0.4; bsdf.inputs["Coat Weight"].default_value = 0.25
    nt.links.new(col, bsdf.inputs["Base Color"])
    return m


def donut_material(name, coord_mode, R=0.12, c=0.62, w=0.38, H=0.07):
    """Donut : glaçage rose + vermicelles sur le dessus, pâte dorée dessous, mie moelleuse à l'intérieur."""
    m, nt, bsdf = mat(name)
    vec = coords(nt, coord_mode)
    sep = nt.nodes.new("ShaderNodeSeparateXYZ"); nt.links.new(vec, sep.inputs[0])
    flat = nt.nodes.new("ShaderNodeVectorMath"); flat.operation = "MULTIPLY"; flat.inputs[1].default_value = (1, 1, 0)
    nt.links.new(vec, flat.inputs[0])
    ln = nt.nodes.new("ShaderNodeVectorMath"); ln.operation = "LENGTH"; nt.links.new(flat.outputs[0], ln.inputs[0])
    # hauteur locale du donut : H * sqrt(1 - ((r/R - c)/w)^2) ; glaçage là où z > 55 % de cette hauteur
    q = nt.nodes.new("ShaderNodeMath"); q.operation = "MULTIPLY_ADD"; q.inputs[1].default_value = 1 / (R * w); q.inputs[2].default_value = -c / w
    nt.links.new(ln.outputs["Value"], q.inputs[0])
    q2 = nt.nodes.new("ShaderNodeMath"); q2.operation = "POWER"; q2.inputs[1].default_value = 2.0; nt.links.new(q.outputs[0], q2.inputs[0])
    one = nt.nodes.new("ShaderNodeMath"); one.operation = "SUBTRACT"; one.inputs[0].default_value = 1.0; one.use_clamp = True
    nt.links.new(q2.outputs[0], one.inputs[1])
    sq = nt.nodes.new("ShaderNodeMath"); sq.operation = "SQRT"; nt.links.new(one.outputs[0], sq.inputs[0])
    lim = nt.nodes.new("ShaderNodeMath"); lim.operation = "MULTIPLY"; lim.inputs[1].default_value = H * 0.55
    nt.links.new(sq.outputs[0], lim.inputs[0])
    upper = nt.nodes.new("ShaderNodeMath"); upper.operation = "GREATER_THAN"
    nt.links.new(sep.outputs[2], upper.inputs[0]); nt.links.new(lim.outputs[0], upper.inputs[1])
    glaze = ramp(nt, noise(nt, vec, 20.0, 3).outputs["Fac"], [(0.4, (0.85, 0.16, 0.36)), (0.6, (0.95, 0.3, 0.5))]).outputs["Color"]
    spr = nt.nodes.new("ShaderNodeTexVoronoi"); spr.inputs["Scale"].default_value = 75.0
    nt.links.new(vec, spr.inputs["Vector"])
    dot = nt.nodes.new("ShaderNodeMath"); dot.operation = "LESS_THAN"; dot.inputs[1].default_value = 0.2
    nt.links.new(spr.outputs["Distance"], dot.inputs[0])
    sprc = nt.nodes.new("ShaderNodeHueSaturation"); sprc.inputs["Saturation"].default_value = 1.6; sprc.inputs["Value"].default_value = 1.5
    nt.links.new(spr.outputs["Color"], sprc.inputs["Color"])
    topc = mixc(nt, dot.outputs[0], glaze, sprc.outputs["Color"])
    crust = ramp(nt, noise(nt, vec, 30.0, 4).outputs["Fac"], [(0.35, (0.5, 0.24, 0.07)), (0.65, (0.72, 0.42, 0.14))]).outputs["Color"]
    outside = mixc(nt, upper.outputs[0], crust, topc)
    if coord_mode == "object":
        col = cut_lines(nt, outside, (1.0, 0.9, 0.75))
        bsdf.inputs["Roughness"].default_value = 0.35; bsdf.inputs["Coat Weight"].default_value = 0.35
    else:
        crumb = ramp(nt, noise(nt, vec, 150.0, 3).outputs["Fac"], [(0.35, (0.82, 0.62, 0.36)), (0.65, (0.92, 0.76, 0.5))]).outputs["Color"]
        dep = nt.nodes.new("ShaderNodeAttribute"); dep.attribute_type = "INSTANCER"; dep.attribute_name = "depth"
        shell = nt.nodes.new("ShaderNodeMath"); shell.operation = "LESS_THAN"; shell.inputs[1].default_value = 0.004
        nt.links.new(dep.outputs["Fac"], shell.inputs[0])
        col = mixc(nt, shell.outputs[0], crumb, outside)
        # glaçage point par point : là où ce point du cube était à moins de 3,5 mm sous le dessus du donut
        topz = nt.nodes.new("ShaderNodeMath"); topz.operation = "MULTIPLY_ADD"; topz.inputs[1].default_value = H; topz.inputs[2].default_value = -0.0035
        nt.links.new(sq.outputs[0], topz.inputs[0])
        near = nt.nodes.new("ShaderNodeMath"); near.operation = "GREATER_THAN"
        nt.links.new(sep.outputs[2], near.inputs[0]); nt.links.new(topz.outputs[0], near.inputs[1])
        glz = nt.nodes.new("ShaderNodeMath"); glz.operation = "MULTIPLY"
        nt.links.new(near.outputs[0], glz.inputs[0]); nt.links.new(upper.outputs[0], glz.inputs[1])
        col = mixc(nt, glz.outputs[0], col, topc)
        bsdf.inputs["Roughness"].default_value = 0.45
    nt.links.new(col, bsdf.inputs["Base Color"])
    return m


def kiwi_material(name, coord_mode, zc=0.042, ry=0.088, rz=0.05):
    """Kiwi couché : peau brune duveteuse ; dedans chair vert vif rayonnante, anneau de graines noires, cœur crème
    (axe du fruit le long de x, à la hauteur zc)."""
    m, nt, bsdf = mat(name)
    vec = coords(nt, coord_mode)
    skin = ramp(nt, noise(nt, vec, 90.0, 6, 0.7).outputs["Fac"], [(0.3, (0.07, 0.035, 0.012)), (0.7, (0.16, 0.09, 0.035))]).outputs["Color"]
    if coord_mode == "object":
        col = cut_lines(nt, skin, (0.55, 0.85, 0.12))
        fuzz = noise(nt, vec, 400.0, 2)
        bump = nt.nodes.new("ShaderNodeBump"); bump.inputs["Strength"].default_value = 0.35
        nt.links.new(fuzz.outputs["Fac"], bump.inputs["Height"]); nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
        bsdf.inputs["Roughness"].default_value = 0.85; bsdf.inputs["Sheen Weight"].default_value = 0.25
        nt.links.new(col, bsdf.inputs["Base Color"])
        return m
    sep = nt.nodes.new("ShaderNodeSeparateXYZ"); nt.links.new(vec, sep.inputs[0])
    # coordonnées autour de l'axe du fruit : u = y/ry, v = (z-zc)/rz
    u = nt.nodes.new("ShaderNodeMath"); u.operation = "MULTIPLY"; u.inputs[1].default_value = 1 / ry
    nt.links.new(sep.outputs[1], u.inputs[0])
    v = nt.nodes.new("ShaderNodeMath"); v.operation = "MULTIPLY_ADD"; v.inputs[1].default_value = 1 / rz; v.inputs[2].default_value = -zc / rz
    nt.links.new(sep.outputs[2], v.inputs[0])
    cmb = nt.nodes.new("ShaderNodeCombineXYZ"); nt.links.new(u.outputs[0], cmb.inputs[0]); nt.links.new(v.outputs[0], cmb.inputs[1])
    ln = nt.nodes.new("ShaderNodeVectorMath"); ln.operation = "LENGTH"; nt.links.new(cmb.outputs[0], ln.inputs[0])
    rr = ln.outputs["Value"]                                       # 0 sur l'axe, ~1 près de la peau
    ang = nt.nodes.new("ShaderNodeMath"); ang.operation = "ARCTAN2"
    nt.links.new(v.outputs[0], ang.inputs[0]); nt.links.new(u.outputs[0], ang.inputs[1])
    rays = nt.nodes.new("ShaderNodeMath"); rays.operation = "MULTIPLY"; rays.inputs[1].default_value = 34 / (2 * math.pi)
    nt.links.new(ang.outputs[0], rays.inputs[0])
    pp = nt.nodes.new("ShaderNodeMath"); pp.operation = "PINGPONG"; pp.inputs[1].default_value = 0.5
    nt.links.new(rays.outputs[0], pp.inputs[0])
    # chair : vert foncé près de la peau → vert vif → vert clair vers le cœur, avec rayons plus clairs
    flesh = ramp(nt, rr, [(0.0, (0.8, 0.76, 0.42)), (0.22, (0.75, 0.72, 0.36)), (0.3, (0.2, 0.5, 0.0)),
                          (0.6, (0.08, 0.34, 0.0)), (1.0, (0.03, 0.16, 0.0))]).outputs["Color"]
    streak = ramp(nt, pp.outputs[0], [(0.0, (1, 1, 1)), (0.12, (0, 0, 0))]).outputs["Color"]
    sk = nt.nodes.new("ShaderNodeMath"); sk.operation = "MULTIPLY"; sk.inputs[1].default_value = 0.35
    nt.links.new(streak, sk.inputs[0])
    flesh = mixc(nt, sk.outputs[0], flesh, (0.4, 0.62, 0.08))
    # graines : petites taches noires dans l'anneau 0.3 < rr < 0.46
    vor = nt.nodes.new("ShaderNodeTexVoronoi"); vor.inputs["Scale"].default_value = 110.0
    nt.links.new(vec, vor.inputs["Vector"])
    dot = nt.nodes.new("ShaderNodeMath"); dot.operation = "LESS_THAN"; dot.inputs[1].default_value = 0.34
    nt.links.new(vor.outputs["Distance"], dot.inputs[0])
    ring = ramp(nt, rr, [(0.27, (0, 0, 0)), (0.3, (1, 1, 1)), (0.5, (1, 1, 1)), (0.53, (0, 0, 0))]).outputs["Color"]
    seed = nt.nodes.new("ShaderNodeMath"); seed.operation = "MULTIPLY"
    nt.links.new(dot.outputs[0], seed.inputs[0]); nt.links.new(ring, seed.inputs[1])
    flesh = mixc(nt, seed.outputs[0], flesh, (0.015, 0.01, 0.008))
    # peau fine (2 mm) sur la couche extérieure
    dep = nt.nodes.new("ShaderNodeAttribute"); dep.attribute_type = "INSTANCER"; dep.attribute_name = "depth"
    sk2 = nt.nodes.new("ShaderNodeMath"); sk2.operation = "LESS_THAN"; sk2.inputs[1].default_value = 0.002
    nt.links.new(dep.outputs["Fac"], sk2.inputs[0])
    col = mixc(nt, sk2.outputs[0], flesh, skin)
    bsdf.inputs["Roughness"].default_value = 0.3; bsdf.inputs["Coat Weight"].default_value = 0.4
    nt.links.new(col, bsdf.inputs["Base Color"])
    return m


def dragon_material(name, coord_mode):
    """Fruit du dragon : peau rose vif à écailles (pointes vertes) ; dedans écorce rose fine puis chair magenta
    pleine de petites graines noires."""
    m, nt, bsdf = mat(name)
    vec = coords(nt, coord_mode)
    skin = ramp(nt, noise(nt, vec, 30.0, 4).outputs["Fac"], [(0.35, (0.75, 0.02, 0.18)), (0.65, (0.95, 0.08, 0.3))]).outputs["Color"]
    sc = nt.nodes.new("ShaderNodeTexVoronoi"); sc.inputs["Scale"].default_value = 26.0; sc.feature = "DISTANCE_TO_EDGE"
    nt.links.new(vec, sc.inputs["Vector"])
    tip = ramp(nt, sc.outputs["Distance"], [(0.0, (1, 1, 1)), (0.05, (1, 1, 1)), (0.1, (0, 0, 0))]).outputs["Color"]
    skin = mixc(nt, tip, skin, (0.25, 0.45, 0.04))                  # bords des écailles verts
    if coord_mode == "object":
        col = cut_lines(nt, skin, (1.0, 0.55, 0.75))
        bump = nt.nodes.new("ShaderNodeBump"); bump.inputs["Strength"].default_value = 0.6
        nt.links.new(sc.outputs["Distance"], bump.inputs["Height"]); nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
        bsdf.inputs["Roughness"].default_value = 0.3; bsdf.inputs["Coat Weight"].default_value = 0.5
        nt.links.new(col, bsdf.inputs["Base Color"])
        return m
    juice = noise(nt, vec, 18.0, 3)
    flesh = ramp(nt, juice.outputs["Fac"], [(0.35, (0.5, 0.0, 0.2)), (0.65, (0.72, 0.02, 0.32))]).outputs["Color"]
    vor = nt.nodes.new("ShaderNodeTexVoronoi"); vor.inputs["Scale"].default_value = 170.0
    nt.links.new(vec, vor.inputs["Vector"])
    dot = nt.nodes.new("ShaderNodeMath"); dot.operation = "LESS_THAN"; dot.inputs[1].default_value = 0.27
    nt.links.new(vor.outputs["Distance"], dot.inputs[0])
    flesh = mixc(nt, dot.outputs[0], flesh, (0.012, 0.006, 0.006))
    dep = nt.nodes.new("ShaderNodeAttribute"); dep.attribute_type = "INSTANCER"; dep.attribute_name = "depth"
    k = nt.nodes.new("ShaderNodeMath"); k.operation = "MULTIPLY"; k.inputs[1].default_value = 1 / 0.004
    nt.links.new(dep.outputs["Fac"], k.inputs[0])
    rind = ramp(nt, k.outputs[0], [(0.0, (1, 1, 1)), (0.7, (1, 1, 1)), (0.9, (0, 0, 0))]).outputs["Color"]
    col = mixc(nt, rind, flesh, (0.9, 0.12, 0.38))
    bsdf.inputs["Roughness"].default_value = 0.28; bsdf.inputs["Coat Weight"].default_value = 0.4
    nt.links.new(col, bsdf.inputs["Base Color"])
    return m


def pomegranate_material(name, coord_mode):
    """Grenade : peau rouge brique mate ; dedans écorce crème, cloisons blanches et grains rouge rubis brillants."""
    m, nt, bsdf = mat(name)
    vec = coords(nt, coord_mode)
    skin = ramp(nt, noise(nt, vec, 14.0, 5).outputs["Fac"], [(0.3, (0.22, 0.01, 0.02)), (0.7, (0.42, 0.03, 0.04))]).outputs["Color"]
    if coord_mode == "object":
        col = cut_lines(nt, skin, (1.0, 0.85, 0.65))
        bump = nt.nodes.new("ShaderNodeBump"); bump.inputs["Strength"].default_value = 0.3
        nt.links.new(noise(nt, vec, 60.0, 4).outputs["Fac"], bump.inputs["Height"]); nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
        bsdf.inputs["Roughness"].default_value = 0.6; bsdf.inputs["Coat Weight"].default_value = 0.1
        nt.links.new(col, bsdf.inputs["Base Color"])
        return m
    ar = nt.nodes.new("ShaderNodeTexVoronoi"); ar.inputs["Scale"].default_value = 55.0
    nt.links.new(vec, ar.inputs["Vector"])
    ed = nt.nodes.new("ShaderNodeTexVoronoi"); ed.inputs["Scale"].default_value = 55.0; ed.feature = "DISTANCE_TO_EDGE"
    nt.links.new(vec, ed.inputs["Vector"])
    aril = ramp(nt, ar.outputs["Distance"], [(0.0, (0.85, 0.08, 0.12)), (0.25, (0.55, 0.0, 0.05)), (0.6, (0.3, 0.0, 0.03))]).outputs["Color"]
    gap = ramp(nt, ed.outputs["Distance"], [(0.0, (1, 1, 1)), (0.012, (1, 1, 1)), (0.03, (0, 0, 0))]).outputs["Color"]
    col = mixc(nt, gap, aril, (0.8, 0.6, 0.45))
    # cloisons blanches entre les loges (6 secteurs)
    sep = nt.nodes.new("ShaderNodeSeparateXYZ"); nt.links.new(vec, sep.inputs[0])
    ang = nt.nodes.new("ShaderNodeMath"); ang.operation = "ARCTAN2"
    nt.links.new(sep.outputs[1], ang.inputs[0]); nt.links.new(sep.outputs[0], ang.inputs[1])
    sg = nt.nodes.new("ShaderNodeMath"); sg.operation = "MULTIPLY"; sg.inputs[1].default_value = 6 / (2 * math.pi)
    nt.links.new(ang.outputs[0], sg.inputs[0])
    pp = nt.nodes.new("ShaderNodeMath"); pp.operation = "PINGPONG"; pp.inputs[1].default_value = 0.5
    nt.links.new(sg.outputs[0], pp.inputs[0])
    memb = nt.nodes.new("ShaderNodeMath"); memb.operation = "LESS_THAN"; memb.inputs[1].default_value = 0.018
    nt.links.new(pp.outputs[0], memb.inputs[0])
    col = mixc(nt, memb.outputs[0], col, (0.95, 0.84, 0.66))
    dep = nt.nodes.new("ShaderNodeAttribute"); dep.attribute_type = "INSTANCER"; dep.attribute_name = "depth"
    k = nt.nodes.new("ShaderNodeMath"); k.operation = "MULTIPLY"; k.inputs[1].default_value = 1 / 0.01
    nt.links.new(dep.outputs["Fac"], k.inputs[0])
    lay = ramp(nt, k.outputs[0], [(0.0, (0, 0, 0)), (0.1, (0, 0, 0)), (0.14, (0.5, 0.5, 0.5)), (0.55, (0.5, 0.5, 0.5)), (0.6, (1, 1, 1))])
    lo = nt.nodes.new("ShaderNodeMath"); lo.operation = "MULTIPLY"; lo.inputs[1].default_value = 2.0; lo.use_clamp = True
    nt.links.new(lay.outputs["Color"], lo.inputs[0])
    hi = nt.nodes.new("ShaderNodeMath"); hi.operation = "MULTIPLY_ADD"; hi.inputs[1].default_value = 2.0; hi.inputs[2].default_value = -1.0
    hi.use_clamp = True; nt.links.new(lay.outputs["Color"], hi.inputs[0])
    out = mixc(nt, lo.outputs[0], skin, (0.93, 0.78, 0.55))         # peau → écorce crème
    out = mixc(nt, hi.outputs[0], out, col)                          # → grains
    bsdf.inputs["Roughness"].default_value = 0.15; bsdf.inputs["Coat Weight"].default_value = 0.6
    nt.links.new(out, bsdf.inputs["Base Color"])
    return m


def strawberry_material(name, coord_mode, zc=0.04, ry=0.1, rz=0.06, cap_x=0.115):
    """Fraise couchée (pointe vers -x) : rouge vif brillant, akènes jaunes, collerette verte ; dedans chair rouge
    → rose → cœur blanc le long de l'axe, avec stries claires rayonnantes."""
    m, nt, bsdf = mat(name)
    vec = coords(nt, coord_mode)
    sep = nt.nodes.new("ShaderNodeSeparateXYZ"); nt.links.new(vec, sep.inputs[0])
    red = ramp(nt, noise(nt, vec, 12.0, 4).outputs["Fac"], [(0.3, (0.55, 0.0, 0.02)), (0.7, (0.85, 0.03, 0.04))]).outputs["Color"]
    seeds = nt.nodes.new("ShaderNodeTexVoronoi"); seeds.inputs["Scale"].default_value = 70.0
    nt.links.new(vec, seeds.inputs["Vector"])
    sd = nt.nodes.new("ShaderNodeMath"); sd.operation = "LESS_THAN"; sd.inputs[1].default_value = 0.09
    nt.links.new(seeds.outputs["Distance"], sd.inputs[0])
    skin = mixc(nt, sd.outputs[0], red, (0.6, 0.38, 0.02))
    cap = nt.nodes.new("ShaderNodeMath"); cap.operation = "GREATER_THAN"; cap.inputs[1].default_value = cap_x
    nt.links.new(sep.outputs[0], cap.inputs[0])
    skin = mixc(nt, cap.outputs[0], skin, (0.06, 0.25, 0.02))
    if coord_mode == "object":
        col = cut_lines(nt, skin, (1.0, 0.75, 0.75))
        bump = nt.nodes.new("ShaderNodeBump"); bump.inputs["Strength"].default_value = 0.4; bump.invert = True
        nt.links.new(seeds.outputs["Distance"], bump.inputs["Height"]); nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
        bsdf.inputs["Roughness"].default_value = 0.2; bsdf.inputs["Coat Weight"].default_value = 0.7
        nt.links.new(col, bsdf.inputs["Base Color"])
        return m
    u = nt.nodes.new("ShaderNodeMath"); u.operation = "MULTIPLY"; u.inputs[1].default_value = 1 / ry
    nt.links.new(sep.outputs[1], u.inputs[0])
    v = nt.nodes.new("ShaderNodeMath"); v.operation = "MULTIPLY_ADD"; v.inputs[1].default_value = 1 / rz; v.inputs[2].default_value = -zc / rz
    nt.links.new(sep.outputs[2], v.inputs[0])
    cmb = nt.nodes.new("ShaderNodeCombineXYZ"); nt.links.new(u.outputs[0], cmb.inputs[0]); nt.links.new(v.outputs[0], cmb.inputs[1])
    ln = nt.nodes.new("ShaderNodeVectorMath"); ln.operation = "LENGTH"; nt.links.new(cmb.outputs[0], ln.inputs[0])
    ang = nt.nodes.new("ShaderNodeMath"); ang.operation = "ARCTAN2"
    nt.links.new(v.outputs[0], ang.inputs[0]); nt.links.new(u.outputs[0], ang.inputs[1])
    rays = nt.nodes.new("ShaderNodeMath"); rays.operation = "MULTIPLY"; rays.inputs[1].default_value = 22 / (2 * math.pi)
    nt.links.new(ang.outputs[0], rays.inputs[0])
    pp = nt.nodes.new("ShaderNodeMath"); pp.operation = "PINGPONG"; pp.inputs[1].default_value = 0.5
    nt.links.new(rays.outputs[0], pp.inputs[0])
    flesh = ramp(nt, ln.outputs["Value"], [(0.0, (0.95, 0.75, 0.68)), (0.22, (0.92, 0.6, 0.55)), (0.4, (0.9, 0.18, 0.2)),
                                            (0.75, (0.75, 0.02, 0.05)), (1.0, (0.6, 0.0, 0.03))]).outputs["Color"]
    st = ramp(nt, pp.outputs[0], [(0.0, (1, 1, 1)), (0.1, (0, 0, 0))]).outputs["Color"]
    stk = nt.nodes.new("ShaderNodeMath"); stk.operation = "MULTIPLY"; stk.inputs[1].default_value = 0.4
    nt.links.new(st, stk.inputs[0])
    flesh = mixc(nt, stk.outputs[0], flesh, (0.95, 0.65, 0.6))
    dep = nt.nodes.new("ShaderNodeAttribute"); dep.attribute_type = "INSTANCER"; dep.attribute_name = "depth"
    sk2 = nt.nodes.new("ShaderNodeMath"); sk2.operation = "LESS_THAN"; sk2.inputs[1].default_value = 0.0015
    nt.links.new(dep.outputs["Fac"], sk2.inputs[0])
    col = mixc(nt, sk2.outputs[0], flesh, skin)
    bsdf.inputs["Roughness"].default_value = 0.22; bsdf.inputs["Coat Weight"].default_value = 0.5
    nt.links.new(col, bsdf.inputs["Base Color"])
    return m


def meat_material(name, coord_mode):
    """Viande crue persillée. coord_mode = 'object' (steak entier) ou 'instancer' (cubes : position d'origine)."""
    m, nt, bsdf = mat(name)
    vec = coords(nt, coord_mode)
    marb = noise(nt, vec, 16.0, 6, 0.55, 1.2)
    veins = ramp(nt, marb.outputs["Fac"], [(0.0, (0.0, 0, 0)), (0.488, (0.0, 0, 0)), (0.5, (1, 1, 1)), (0.512, (0.0, 0, 0))])
    tone = noise(nt, vec, 6.0, 3, 0.5)
    base = ramp(nt, tone.outputs["Fac"], [(0.3, (0.42, 0.035, 0.045)), (0.7, (0.62, 0.08, 0.075))])
    mix = nt.nodes.new("ShaderNodeMix"); mix.data_type = "RGBA"
    nt.links.new(veins.outputs["Color"], mix.inputs["Factor"])
    nt.links.new(base.outputs["Color"], mix.inputs[6]); mix.inputs[7].default_value = (0.85, 0.72, 0.64, 1)
    col = mix.outputs[2]
    if coord_mode == "object":
        col = cut_lines(nt, col)
    nt.links.new(col, bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.38
    bsdf.inputs["Coat Weight"].default_value = 0.15
    bump = nt.nodes.new("ShaderNodeBump"); bump.inputs["Strength"].default_value = 0.15
    nt.links.new(noise(nt, vec, 90.0, 4).outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return m


CUT = {}   # entrées du shader de coupes, mises à jour à chaque image


def cut_lines(nt, col, line_col=(0.16, 0.02, 0.02)):
    """Assombrit de fines lignes aux plans de coupe déjà faits (sur chaque axe : origine, pas, limite)."""
    tc = nt.nodes.new("ShaderNodeTexCoord"); sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Object"], sep.inputs[0])
    total = None
    for ax in range(3):
        v = sep.outputs[ax]
        o = nt.nodes.new("ShaderNodeValue"); s = nt.nodes.new("ShaderNodeValue"); lim = nt.nodes.new("ShaderNodeValue")
        CUT[ax] = (o, s, lim)
        sub = nt.nodes.new("ShaderNodeMath"); sub.operation = "SUBTRACT"; nt.links.new(v, sub.inputs[0]); nt.links.new(o.outputs[0], sub.inputs[1])
        div = nt.nodes.new("ShaderNodeMath"); div.operation = "DIVIDE"; nt.links.new(sub.outputs[0], div.inputs[0]); nt.links.new(s.outputs[0], div.inputs[1])
        fr = nt.nodes.new("ShaderNodeMath"); fr.operation = "PINGPONG"; fr.inputs[1].default_value = 0.5
        nt.links.new(div.outputs[0], fr.inputs[0])                          # distance au plan le plus proche (en pas)
        lw = nt.nodes.new("ShaderNodeValue"); CUT[(ax, "w")] = lw
        line = nt.nodes.new("ShaderNodeMath"); line.operation = "LESS_THAN"; nt.links.new(fr.outputs[0], line.inputs[0]); nt.links.new(lw.outputs[0], line.inputs[1])
        done = nt.nodes.new("ShaderNodeMath"); done.operation = "LESS_THAN"; nt.links.new(v, done.inputs[0]); nt.links.new(lim.outputs[0], done.inputs[1])
        both = nt.nodes.new("ShaderNodeMath"); both.operation = "MULTIPLY"; nt.links.new(line.outputs[0], both.inputs[0]); nt.links.new(done.outputs[0], both.inputs[1])
        if total is None:
            total = both
        else:
            mx = nt.nodes.new("ShaderNodeMath"); mx.operation = "MAXIMUM"; nt.links.new(total.outputs[0], mx.inputs[0]); nt.links.new(both.outputs[0], mx.inputs[1]); total = mx
    mix = nt.nodes.new("ShaderNodeMix"); mix.data_type = "RGBA"
    nt.links.new(total.outputs[0], mix.inputs["Factor"]); nt.links.new(col, mix.inputs[6]); mix.inputs[7].default_value = (*line_col, 1)
    return mix.outputs[2]


def wood_material():
    m, nt, bsdf = mat("wood")
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping"); mp.inputs["Scale"].default_value = (1.0, 14.0, 1.0)
    nt.links.new(tc.outputs["Object"], mp.inputs[0])
    wave = nt.nodes.new("ShaderNodeTexWave"); wave.inputs["Scale"].default_value = 6.0; wave.inputs["Distortion"].default_value = 6.0
    wave.inputs["Detail"].default_value = 4.0; wave.bands_direction = "Y"
    nt.links.new(mp.outputs[0], wave.inputs[0])
    r = ramp(nt, wave.outputs["Fac"], [(0.2, (0.10, 0.035, 0.014)), (0.8, (0.2, 0.08, 0.035))])
    nt.links.new(r.outputs["Color"], bsdf.inputs["Base Color"]); bsdf.inputs["Roughness"].default_value = 0.55
    return m


def plain_material(name, c0, c1, scale=3.0, rough=0.8):
    m, nt, bsdf = mat(name)
    r = ramp(nt, noise(nt, None, scale, 5, 0.6).outputs["Fac"], [(0.3, c0), (0.7, c1)])
    nt.links.new(r.outputs["Color"], bsdf.inputs["Base Color"]); bsdf.inputs["Roughness"].default_value = rough
    return m


def steel_material():
    m, nt, bsdf = mat("steel")
    bsdf.inputs["Base Color"].default_value = (0.82, 0.83, 0.85, 1); bsdf.inputs["Metallic"].default_value = 1.0
    bsdf.inputs["Roughness"].default_value = 0.22
    return m


# ================================================================ scène
def build():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"; sc.cycles.device = "CPU"
    sc.cycles.samples = SAMPLES; sc.cycles.use_denoising = DENOISE
    sc.cycles.max_bounces = 3; sc.cycles.diffuse_bounces = 1; sc.cycles.glossy_bounces = 1; sc.cycles.transmission_bounces = 0
    sc.cycles.use_adaptive_sampling = True; sc.cycles.adaptive_threshold = 0.03
    sc.cycles.denoiser = "OPENIMAGEDENOISE"; sc.render.threads_mode = "AUTO"
    sc.render.resolution_x, sc.render.resolution_y = RES, RES * 16 // 9
    sc.render.film_transparent = False
    sc.render.use_persistent_data = True
    sc.render.use_motion_blur = MBLUR; sc.render.motion_blur_shutter = 0.5
    sc.render.fps = core.FPS
    sc.view_settings.view_transform = "AgX"; sc.view_settings.look = "AgX - Medium High Contrast"
    sc.render.image_settings.file_format = "PNG"; sc.render.image_settings.color_mode = "RGB"
    sc.render.image_settings.compression = 15

    w = bpy.data.worlds.new("w"); sc.world = w; w.use_nodes = True
    w.node_tree.nodes["Background"].inputs[0].default_value = (0.09, 0.06, 0.05, 1)
    w.node_tree.nodes["Background"].inputs[1].default_value = 0.25

    # plan de travail + mur
    bpy.ops.mesh.primitive_plane_add(size=6, location=(0, 0, -0.024))
    th = cfg.get("theme", {})
    cc = th.get("counter", [[0.05, 0.048, 0.046], [0.09, 0.085, 0.08]])
    wc = th.get("wall", [[0.09, 0.03, 0.018], [0.17, 0.06, 0.03]])
    bpy.context.object.data.materials.append(plain_material("counter", tuple(cc[0]), tuple(cc[1]), 4.0, 0.6))
    bpy.ops.mesh.primitive_plane_add(size=6, location=(0, 0.75, 1.0), rotation=(math.pi / 2, 0, 0))
    bpy.context.object.data.materials.append(plain_material("wall", tuple(wc[0]), tuple(wc[1]), 2.5, 0.9))
    # planche
    bpy.ops.mesh.primitive_cube_add(location=(0, 0, -0.012))
    b = bpy.context.object; b.scale = (0.24, 0.17, 0.012)
    bev = b.modifiers.new("bev", "BEVEL"); bev.width = 0.004; bev.segments = 3
    b.data.materials.append(wood_material())

    # objet à couper
    verts, faces = SH.mesh()
    me = bpy.data.meshes.new("steak"); me.from_pydata(verts, [], faces); me.update()
    bm = bmesh.new(); bm.from_mesh(me); bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-6)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces); bm.to_mesh(me); bm.free()
    for p in me.polygons:
        p.use_smooth = True
    steak = bpy.data.objects.new("steak", me); sc.collection.objects.link(steak)
    MATS = {"meat": meat_material, "watermelon": watermelon_material, "cake": cake_material, "orange": orange_material, "pizza": pizza_material, "avocado": avocado_material, "donut": donut_material, "kiwi": kiwi_material, "dragon": dragon_material, "pomegranate": pomegranate_material, "strawberry": strawberry_material}
    material = MATS[cfg.get("material", "meat")]
    steak.data.materials.append(material("whole", "object"))

    # cubes : nuage de points + Geometry Nodes (instances)
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, -5))
    proto = bpy.context.object; proto.name = "proto"
    pb = proto.modifiers.new("bev", "BEVEL"); pb.width = 0.06; pb.segments = 2
    bpy.ops.object.modifier_apply(modifier="bev")
    proto.data.materials.append(material("cube", "instancer"))
    proto.hide_render = True
    cme = bpy.data.meshes.new("cubes"); cubes = bpy.data.objects.new("cubes", cme); sc.collection.objects.link(cubes)
    cubes.cycles.use_motion_blur = False
    gn = cubes.modifiers.new("inst", "NODES"); gn.node_group = instancer_group(proto)

    # couteau
    knife = make_knife()

    # caméra + lumières
    cam_data = bpy.data.cameras.new("cam"); cam_data.lens = 50; cam_data.sensor_width = 36
    cam = bpy.data.objects.new("cam", cam_data); sc.collection.objects.link(cam); sc.camera = cam
    cam.location = (0.34, -0.66, 0.50)
    look_at(cam, (0.0, 0.035, 0.0))
    for name, loc, power, size, col in (("key", (-0.45, -0.5, 0.7), 14, 0.7, (1.0, 0.9, 0.8)),
                                        ("fill", (0.6, -0.4, 0.35), 3, 0.8, (0.85, 0.9, 1.0)),
                                        ("rim", (0.1, 0.55, 0.45), 9, 0.5, (1.0, 0.85, 0.7))):
        ld = bpy.data.lights.new(name, "AREA"); ld.energy = power; ld.size = size; ld.color = col
        lo = bpy.data.objects.new(name, ld); sc.collection.objects.link(lo); lo.location = loc; look_at(lo, (0, 0, 0))
    return sc, steak, cubes, knife


def look_at(obj, target):
    from mathutils import Vector
    d = Vector(target) - obj.location
    obj.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()


def instancer_group(proto):
    ng = bpy.data.node_groups.new("cubes", "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    N = ng.nodes; L = ng.links
    gi = N.new("NodeGroupInput"); go = N.new("NodeGroupOutput")
    oi = N.new("GeometryNodeObjectInfo"); oi.inputs["Object"].default_value = proto
    m2p = N.new("GeometryNodeMeshToPoints")
    inst = N.new("GeometryNodeInstanceOnPoints")
    rot = N.new("GeometryNodeInputNamedAttribute"); rot.data_type = "FLOAT_VECTOR"; rot.inputs["Name"].default_value = "rot"
    scl = N.new("GeometryNodeInputNamedAttribute"); scl.data_type = "FLOAT_VECTOR"; scl.inputs["Name"].default_value = "scl"
    e2r = N.new("FunctionNodeEulerToRotation")
    L.new(gi.outputs[0], m2p.inputs["Mesh"]); L.new(m2p.outputs[0], inst.inputs["Points"])
    L.new(oi.outputs["Geometry"], inst.inputs["Instance"])
    L.new(rot.outputs["Attribute"], e2r.inputs[0]); L.new(e2r.outputs[0], inst.inputs["Rotation"])
    L.new(scl.outputs["Attribute"], inst.inputs["Scale"])
    L.new(inst.outputs[0], go.inputs[0])
    return ng


def make_knife():
    """Lame (acier) + manche (bois sombre). Repère : lame le long de +Y, tranchant en bas (z=0), dos à z=0.05."""
    verts = [(-0.0012, 0.0, 0.0), (0.0012, 0.0, 0.0), (-0.0012, 0.34, 0.012), (0.0012, 0.34, 0.012),
             (-0.0012, 0.0, 0.055), (0.0012, 0.0, 0.055), (-0.0012, 0.34, 0.05), (0.0012, 0.34, 0.05)]
    faces = [(0, 2, 3, 1), (4, 5, 7, 6), (0, 4, 6, 2), (1, 3, 7, 5), (0, 1, 5, 4), (2, 6, 7, 3)]
    me = bpy.data.meshes.new("blade"); me.from_pydata(verts, [], faces); me.update()
    knife = bpy.data.objects.new("knife", me); bpy.context.scene.collection.objects.link(knife)
    knife.data.materials.append(steel_material())
    bpy.ops.mesh.primitive_cube_add(location=(0, -0.06, 0.03))
    h = bpy.context.object; h.scale = (0.008, 0.06, 0.014); h.parent = knife
    h.data.materials.append(plain_material("handle", (0.05, 0.03, 0.02), (0.1, 0.06, 0.04), 8.0, 0.5))
    return knife


# ================================================================ animation du couteau
UP = 0.13           # hauteur de repos du tranchant au-dessus de la planche


def knife_keys(knife):
    """Images clés du couteau pour tous les rounds (le flou de mouvement vient de là)."""
    knife.animation_data_create()
    def key(t, loc, rot):
        f = t * core.FPS
        knife.location = loc; knife.rotation_euler = rot
        knife.keyframe_insert("location", frame=f); knife.keyframe_insert("rotation_euler", frame=f)
    x0, x1, y0, y1, _, _ = SH.bbox()
    ROT = {0: (0, 0, 0), 1: (0, 0, math.pi / 2), 2: (0, math.pi / 2, 0)}
    for r in PLAN.rounds:
        times, cuts = r["times"], r["cuts"]
        key(r["t0"], (cuts[0][1], -0.17, UP + 0.05), ROT[0])      # déjà au-dessus de la 1re coupe (pas de traversée)
        prev = r["t0"]
        for k, ((ax, c), t) in enumerate(zip(cuts, times)):
            nxt = times[k + 1] if k + 1 < len(times) else r["t_cut_end"]
            half = max(0.012, min(0.16, (t - prev) * 0.45, (nxt - t) * 0.45))
            if ax == 0:
                key(t - half, (c, -0.17, UP), ROT[0]); key(t, (c, -0.17, -0.002), ROT[0]); key(t + half, (c, -0.17, UP), ROT[0])
            elif ax == 1:
                key(t - half, (0.17, c, UP), ROT[1]); key(t, (0.17, c, -0.002), ROT[1]); key(t + half, (0.17, c, UP), ROT[1])
            else:
                side = 1 if k % 2 == 0 else -1
                key(t - half, (side * 0.36, -0.17, c), ROT[2]); key(t + half, (-side * 0.36, -0.17, c), ROT[2])
            prev = t
        key(r["t_col"], (x1 + 0.1, -0.2, UP + 0.1), ROT[0]); key(r["t_end"] - 0.01, (x1 + 0.4, -0.4, UP + 0.4), ROT[0])
    if PLAN.loop:                                           # le couteau revient à sa position de départ
        first = PLAN.rounds[0]["cuts"][0][1]
        key(PLAN.t_loop + 0.05, (x1 + 0.4, -0.4, UP + 0.4), ROT[0])
        key(PLAN.total, (first, -0.17, UP + 0.05), ROT[0])
    for fc in knife.animation_data.action.fcurves:
        for kp in fc.keyframe_points:
            kp.interpolation = "LINEAR"


# ================================================================ rendu image par image
def main():
    sc, steak, cubes, knife = build()
    knife_keys(knife)
    cur = {"i": -1, "C": None}
    t_start = time.time()
    for f in range(F0, F1, STEP):
        t = f / core.FPS
        i, r = PLAN.round_at(t)
        sc.frame_set(f)
        x0, x1, y0, y1, z0, z1 = SH.bbox()
        span = (x1 - x0, y1 - y0, z1 - z0); org = (x0, y0, z0)
        done = int(np.searchsorted(r["times"], t, side="right"))
        nx, ny, nz = r["nxyz"]
        for ax, n_ax, lo in ((0, nx, 0), (1, ny, nx), (2, nz, nx + ny)):
            o, s, lim = CUT[ax]
            step = span[ax] / (n_ax + 1)
            o.outputs[0].default_value = org[ax]; s.outputs[0].default_value = step
            k_done = max(0, min(n_ax, done - lo))
            lim.outputs[0].default_value = (org[ax] + step * (k_done + 0.5)) if k_done > 0 else -10.0
            CUT[(ax, "w")].outputs[0].default_value = min(0.06, 0.0006 / step)     # ~0,6 mm de large
        back = PLAN.rewind(t)
        if back >= 0.999:                                   # fin de boucle : objet entier, sans coupes = 1re image
            for ax in range(3):
                CUT[ax][2].outputs[0].default_value = -10.0
        collapsed = t >= r["t_col"] and back < 0.999
        steak.hide_render = collapsed
        if collapsed:
            if cur["i"] != i:
                cur["i"], cur["C"] = i, core.Collapse(SH, r["nxyz"], seed=i, stack=cfg.get("stack", "z"))
                set_points(cubes, cur["C"])
            C = cur["C"]; p, rot = C.at(t - r["t_col"])
            if back > 0:                                    # rembobinage : chaque cube regagne sa place
                p = C.p1 + (C.p0 - C.p1) * back; rot = C.rot1 * (1 - back)
            update_points(cubes, p, rot)
            cubes.hide_render = False
        else:
            cubes.hide_render = True
        sc.render.filepath = os.path.join(OUT, f"f{f:05d}.png")
        bpy.ops.render.render(write_still=True)
        el = time.time() - t_start
        print(f"image {f} ({t:.2f}s) faite — {el / max(1, (f - F0) // STEP + 1):.1f} s/image", flush=True)


def set_points(obj, C):
    me = bpy.data.meshes.new("pts"); me.vertices.add(C.n)
    me.vertices.foreach_set("co", C.p0.ravel())
    a = me.attributes.new("orig", "FLOAT_VECTOR", "POINT"); a.data.foreach_set("vector", C.p0.ravel())
    s = me.attributes.new("scl", "FLOAT_VECTOR", "POINT"); s.data.foreach_set("vector", np.tile(C.size * 0.93, (C.n, 1)).ravel())
    me.attributes.new("rot", "FLOAT_VECTOR", "POINT")
    dp = me.attributes.new("depth", "FLOAT", "POINT"); dp.data.foreach_set("value", C.depth)
    old = obj.data; obj.data = me
    if old.users == 0:
        bpy.data.meshes.remove(old)


def update_points(obj, p, rot):
    me = obj.data
    me.vertices.foreach_set("co", p.ravel())
    me.attributes["rot"].data.foreach_set("vector", rot.ravel())
    me.update()


if __name__ == "__main__":
    main()

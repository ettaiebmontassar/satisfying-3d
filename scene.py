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
def meat_material(name, coord_mode):
    """Viande crue persillée. coord_mode = 'object' (steak entier) ou 'instancer' (cubes : position d'origine)."""
    m, nt, bsdf = mat(name)
    if coord_mode == "object":
        tc = nt.nodes.new("ShaderNodeTexCoord"); vec = tc.outputs["Object"]
    else:
        at = nt.nodes.new("ShaderNodeAttribute"); at.attribute_type = "INSTANCER"; at.attribute_name = "orig"
        sz = nt.nodes.new("ShaderNodeAttribute"); sz.attribute_type = "INSTANCER"; sz.attribute_name = "scl"
        tc = nt.nodes.new("ShaderNodeTexCoord")
        mul = nt.nodes.new("ShaderNodeVectorMath"); mul.operation = "MULTIPLY"
        nt.links.new(tc.outputs["Object"], mul.inputs[0]); nt.links.new(sz.outputs["Vector"], mul.inputs[1])
        add = nt.nodes.new("ShaderNodeVectorMath"); add.operation = "ADD"
        nt.links.new(at.outputs["Vector"], add.inputs[0]); nt.links.new(mul.outputs[0], add.inputs[1])
        vec = add.outputs[0]          # position d'origine de ce point du cube dans le steak entier
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


def cut_lines(nt, col):
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
    nt.links.new(total.outputs[0], mix.inputs["Factor"]); nt.links.new(col, mix.inputs[6]); mix.inputs[7].default_value = (0.16, 0.02, 0.02, 1)
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
    bpy.context.object.data.materials.append(plain_material("counter", (0.05, 0.048, 0.046), (0.09, 0.085, 0.08), 4.0, 0.6))
    bpy.ops.mesh.primitive_plane_add(size=6, location=(0, 0.75, 1.0), rotation=(math.pi / 2, 0, 0))
    bpy.context.object.data.materials.append(plain_material("wall", (0.09, 0.03, 0.018), (0.17, 0.06, 0.03), 2.5, 0.9))
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
    steak.data.materials.append(meat_material("meat_whole", "object"))

    # cubes : nuage de points + Geometry Nodes (instances)
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, -5))
    proto = bpy.context.object; proto.name = "proto"
    pb = proto.modifiers.new("bev", "BEVEL"); pb.width = 0.06; pb.segments = 2
    bpy.ops.object.modifier_apply(modifier="bev")
    proto.data.materials.append(meat_material("meat_cube", "instancer"))
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
        key(r["t0"], (x1 + 0.08, -0.15, UP + 0.05), ROT[0])
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
        collapsed = t >= r["t_col"]
        steak.hide_render = collapsed
        if collapsed:
            if cur["i"] != i:
                cur["i"], cur["C"] = i, core.Collapse(SH, r["nxyz"], seed=i)
                set_points(cubes, cur["C"])
            C = cur["C"]; p, rot = C.at(t - r["t_col"])
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

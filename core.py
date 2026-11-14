"""Calculs partagés (sans Blender) : forme de l'objet, minutage des coupes, effondrement en cubes.

Utilisé par scene.py (rendu Blender), overlay.py (compteurs) et audio.py (bruitages).
Unités : mètres et secondes. Le plan de travail est z = 0.
"""
import json, math, os
import numpy as np

FPS = 30
HERE = os.path.dirname(os.path.abspath(__file__))


# ================================================================ forme (steak)
class Shape:
    """Pavé arrondi au contour irrégulier : contour polaire r(θ), épaisseur H·s(ρ)."""

    def __init__(self, cfg):
        s = cfg["shape"]
        self.ax, self.ay, self.H = s["ax"], s["ay"], s["height"]
        self.wob = s.get("wobble", [[2, 0.06, 0.0], [3, -0.05, 0.5], [1, 0.04, 1.6]])
        self.edge = s.get("edge", 8.0)
        self.power = s.get("power", 0.35)
        th = np.linspace(0, 2 * np.pi, 720)
        self.rmax = float(self.r(th).max())

    def r(self, th):
        out = np.ones_like(th, dtype=float)
        for k, a, ph in self.wob:
            out = out + a * np.cos(k * th + ph)
        return out

    def rho(self, x, y):
        u, v = x / self.ax, y / self.ay
        return np.hypot(u, v) / self.r(np.arctan2(v, u))

    def top(self, rho):
        """Hauteur du dessus pour un rayon normalisé ρ (0 au centre, 1 au bord)."""
        rho = np.clip(rho, 0, 1)
        return self.H * (1 - rho ** self.edge) ** self.power

    def bbox(self):
        return (-self.ax * self.rmax, self.ax * self.rmax, -self.ay * self.rmax, self.ay * self.rmax, 0.0, self.H)

    def mesh(self, rings=48, segs=192):
        """Sommets/faces d'un solide fermé : dessus bombé + dessous plat (bords soudés à ρ = 1)."""
        verts, faces = [], []
        th = np.linspace(0, 2 * np.pi, segs, endpoint=False)
        rr = self.r(th)
        rhos = np.sin(np.linspace(0, np.pi / 2, rings + 1))
        def ring(rho, top):
            z = self.top(np.full(segs, rho)) if top else np.zeros(segs)
            z = np.maximum(z, 0) if top else z
            return np.stack([self.ax * rho * rr * np.cos(th), self.ay * rho * rr * np.sin(th), z], 1)
        for top in (True, False):
            base = len(verts)
            for i, rho in enumerate(rhos):
                verts.extend(ring(rho, top).tolist())
            for i in range(rings):
                for j in range(segs):
                    a = base + i * segs + j; b = base + i * segs + (j + 1) % segs
                    c = base + (i + 1) * segs + (j + 1) % segs; d = base + (i + 1) * segs + j
                    faces.append([a, b, c, d] if top else [a, d, c, b])
        return verts, faces


# ================================================================ minutage
def split_cuts(n, shape):
    """Répartit n coupes sur x, y, z pour obtenir des cubes à peu près réguliers."""
    x0, x1, y0, y1, z0, z1 = shape.bbox()
    L = np.array([x1 - x0, y1 - y0, z1 - z0])
    raw = n * L / L.sum()
    k = np.floor(raw).astype(int)
    for i in np.argsort(-(raw - k))[: n - k.sum()]:
        k[i] += 1
    return [int(v) for v in k]


class Plan:
    """Rounds successifs : coupes (x puis y puis z), pause, effondrement, compte des cubes."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.shape = Shape(cfg)
        self.rounds = []
        t = cfg.get("intro", 0.4)
        for rd in cfg["rounds"]:
            n, dur = rd["cuts"], rd["dur"]
            nx, ny, nz = split_cuts(n, self.shape)
            x0, x1, y0, y1, z0, z1 = self.shape.bbox()
            cuts = []                                 # (axe, coordonnée)
            cuts += [(0, x0 + (x1 - x0) * (k + 1) / (nx + 1)) for k in range(nx)]
            cuts += [(1, y0 + (y1 - y0) * (k + 1) / (ny + 1)) for k in range(ny)]
            cuts += [(2, z0 + (z1 - z0) * (k + 1) / (nz + 1)) for k in range(nz)]
            # rythme : on accélère au fil du round (les premières coupes se voient bien)
            u = np.linspace(0, 1, n + 1)[1:]
            lead = rd.get("lead", cfg.get("lead", 0.25))          # délai avant la 1re coupe
            times = t + lead + (dur - lead - 0.2) * (u ** 1.6)
            t_cut_end = t + dur
            t_col = t_cut_end + rd.get("pause", 0.35)
            t_end = t_col + rd.get("collapse", 2.4)
            self.rounds.append(dict(n=n, nxyz=(nx, ny, nz), cuts=cuts, times=times, t0=t, t_cut_end=t_cut_end,
                                    t_col=t_col, t_end=t_end, cubes=count_cubes(self.shape, nx, ny, nz)))
            t = t_end
        self.total = t + cfg.get("outro", 0.0)
        self.frames = int(math.ceil(self.total * FPS))

    def round_at(self, t):
        for i, r in enumerate(self.rounds):
            if t < r["t_end"]:
                return i, r
        return len(self.rounds) - 1, self.rounds[-1]

    def counters(self, t):
        """(coupes faites, cubes affichés) à l'instant t."""
        i, r = self.round_at(t)
        cuts = int(np.searchsorted(r["times"], t, side="right"))
        if t < r["t_col"]:
            return cuts, 0
        k = min(1.0, (t - r["t_col"]) / 0.9)
        k = 1 - (1 - k) ** 3
        return cuts, int(round(r["cubes"] * k))


# ================================================================ cubes
def _columns(shape, gx, gy, gz):
    """Grille gx×gy colonnes ; pour chacune, nombre de cellules (sur gz) dont le centre est dans l'objet."""
    x0, x1, y0, y1, z0, z1 = shape.bbox()
    xs = x0 + (np.arange(gx) + 0.5) * (x1 - x0) / gx
    ys = y0 + (np.arange(gy) + 0.5) * (y1 - y0) / gy
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    rho = shape.rho(X, Y)
    h = np.where(rho < 1, shape.top(rho), 0.0)
    dz = (z1 - z0) / gz
    cnt = np.clip(np.floor(h / dz + 0.5), 0, gz).astype(np.int64)
    return X, Y, cnt, dz, rho


def count_cubes(shape, nx, ny, nz):
    gx, gy, gz = nx + 1, ny + 1, nz + 1
    tot = 0
    x0, x1, y0, y1, _, _ = shape.bbox()
    step = max(1, 4_000_000 // max(gy, 1))           # par tranches pour limiter la mémoire
    for a in range(0, gx, step):
        xs = x0 + (np.arange(a, min(gx, a + step)) + 0.5) * (x1 - x0) / gx
        ys = y0 + (np.arange(gy) + 0.5) * (y1 - y0) / gy
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        rho = shape.rho(X, Y)
        h = np.where(rho < 1, shape.top(rho), 0.0)
        tot += int(np.clip(np.floor(h / ((shape.H) / gz) + 0.5), 0, gz).sum())
    return tot


class Collapse:
    """Cubes d'un round et leur chute (fausse physique au ralenti, déterministe)."""

    def __init__(self, shape, nxyz, cap=400_000, seed=0, g=1.4, spread=0.18, stack="z"):
        nx, ny, nz = nxyz
        gx, gy, gz = nx + 1, ny + 1, nz + 1
        n_true = count_cubes(shape, nx, ny, nz)
        if n_true > cap:                                   # trop de cubes : grille plus grossière (même allure)
            k = (n_true / cap) ** (1 / 3)
            gx, gy, gz = max(2, round(gx / k)), max(2, round(gy / k)), max(1, round(gz / k))
        X, Y, cnt, dz, rho = _columns(shape, gx, gy, gz)
        x0, x1, y0, y1, _, _ = shape.bbox()
        dx, dy = (x1 - x0) / gx, (y1 - y0) / gy
        reps = cnt.ravel()
        px = np.repeat(X.ravel(), reps); py = np.repeat(Y.ravel(), reps); pr = np.repeat(rho.ravel(), reps)
        start = np.concatenate([[0], np.cumsum(reps)[:-1]])
        level = np.arange(reps.sum()) - np.repeat(start, reps)
        pz = (level + 0.5) * dz
        self.p0 = np.stack([px, py, pz], 1).astype(np.float32)
        # profondeur sous la surface (écorce, croûte…) : verticale, ou depuis le bord pour les flancs raides
        top = shape.top(pr)
        side = (1 - np.clip(pr, 0, 1)) * min(shape.ax, shape.ay)
        self.depth = np.maximum(0, np.minimum(top - pz, side * 1.4)).astype(np.float32)
        self.size = np.array([dx, dy, dz], np.float32)
        n = len(px)
        rng = np.random.default_rng(seed)
        # point d'arrivée : un tas en dôme (même volume), un peu plus large que l'objet.
        # Chaque cube va vers une place du dôme du même secteur angulaire et à peu près au même rayon ;
        # à rayon égal, les plus hauts (ou, si stack='depth', les plus profonds = le cœur) finissent en haut.
        H = shape.H
        f = 1.0 + spread
        x0b, x1b, y0b, y1b, _, _ = shape.bbox()
        gxs = np.arange(x0b * f, x1b * f, dx) + dx / 2; gys = np.arange(y0b * f, y1b * f, dy) + dy / 2
        BX, BY = np.meshgrid(gxs, gys, indexing="ij"); BX, BY = BX.ravel(), BY.ravel()
        q = shape.rho(BX / f, BY / f); inside = q < 1
        BX, BY, q = BX[inside], BY[inside], q[inside]
        prof = 1 - q ** 2
        hm = n * dz * 0.92 / max(prof.sum(), 1e-9)
        for _ in range(30):                                  # hauteur du dôme pour loger tous les cubes
            cap = np.floor(hm * prof / (dz * 0.92)).astype(np.int64)
            if cap.sum() >= n:
                break
            hm *= 1.04
        lvl = np.arange(cap.sum()) - np.repeat(np.concatenate([[0], np.cumsum(cap)[:-1]]), cap)
        sx, sy, sq = np.repeat(BX, cap), np.repeat(BY, cap), np.repeat(q, cap)
        sz = (lvl + 0.5) * dz * 0.92
        S = 48
        def sector(x, y):
            return np.floor((np.arctan2(y / shape.ay, x / shape.ax) + np.pi) / (2 * np.pi) * S).astype(np.int64) % S
        rb = 1 if stack == "depth" else 20          # 'depth' : tri par hauteur sur tout le secteur (pas d'anneaux d'écorce)
        slot_order = np.lexsort((sz, np.floor(sq * rb), sector(sx, sy)))
        slot_order = slot_order[np.linspace(0, len(slot_order) - 1, n).astype(np.int64)]
        top_ = shape.top(pr)
        side_ = (1 - np.clip(pr, 0, 1)) * min(shape.ax, shape.ay)
        dep_ = np.maximum(0, np.minimum(top_ - pz, side_ * 1.4))
        cube_order = np.lexsort((dep_ if stack == "depth" else pz, np.floor(np.clip(pr, 0, 1) * rb), sector(px, py)))
        fx = np.empty(n); fy = np.empty(n); fz = np.empty(n)
        fx[cube_order] = sx[slot_order] + 0.3 * dx * rng.standard_normal(n)
        fy[cube_order] = sy[slot_order] + 0.3 * dy * rng.standard_normal(n)
        fz[cube_order] = sz[slot_order]
        self.p1 = np.stack([fx, fy, fz], 1).astype(np.float32)
        # départs : l'extérieur s'effrite d'abord, puis le cœur
        self.delay = (0.45 * (1 - np.clip(pr, 0, 1)) + 0.12 * rng.random(n) + 0.25 * (1 - pz / H)).astype(np.float32)
        drop = np.maximum(pz - fz, 0.002)
        self.T = (np.sqrt(2 * drop / g) + 0.18).astype(np.float32)
        self.g = g
        self.rot1 = np.stack([0.35 * rng.standard_normal(n), 0.35 * rng.standard_normal(n), 0.5 * rng.standard_normal(n)], 1).astype(np.float32)
        self.n_true, self.n = n_true, n

    def at(self, tr):
        """Positions et rotations tr secondes après le début de l'effondrement."""
        u = np.clip((tr - self.delay) / self.T, 0, 1)[:, None]
        e = 1 - (1 - u) ** 2
        p = self.p0 + (self.p1 - self.p0) * e
        # chute en arc : légère avance verticale (gravité) puis repos
        fall = np.clip(u[:, 0], 0, 1) ** 2
        p[:, 2] = self.p0[:, 2] + (self.p1[:, 2] - self.p0[:, 2]) * fall
        rot = self.rot1 * e
        return p, rot


def load(path):
    with open(path, encoding="utf8") as f:
        return json.load(f)

"""Ajoute les compteurs CUTS / CUBES sur les images rendues puis encode un segment vidéo.

python3 overlay.py objects/steak.json FRAMES_DIR out.mp4 [--frames A B]
"""
import os, subprocess, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import core

HERE = os.path.dirname(os.path.abspath(__file__))
FONT = os.path.join(HERE, "fonts", "ArchivoBlack-Regular.ttf")
WHITE, AMBER = (246, 240, 230), (236, 168, 58)


def counter(img, value, label, cy, big, small, pop=0.0):
    W, H = img.size
    s = W / 1080
    fb = ImageFont.truetype(FONT, int(big * s * (1 + 0.12 * pop))); fs = ImageFont.truetype(FONT, int(small * s))
    txt = f"{value:,}"
    shadow = Image.new("RGBA", img.size, (0, 0, 0, 0)); d = ImageDraw.Draw(shadow)
    d.text((W / 2, cy + 6 * s), txt, font=fb, fill=(0, 0, 0, 150), anchor="mm")
    d.text((W / 2, cy + (big * 0.78 + 4) * s), label, font=fs, fill=(0, 0, 0, 150), anchor="mm")
    img.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(6 * s)))
    d = ImageDraw.Draw(img)
    d.text((W / 2, cy), txt, font=fb, fill=WHITE, anchor="mm")
    d.text((W / 2, cy + big * 0.78 * s), label, font=fs, fill=AMBER, anchor="mm")


def pill(img, text, cy, size, a=1.0):
    """Bandeau d'accroche (texte noir sur jaune) en haut de l'image."""
    W = img.size[0]; s = W / 1080
    f = ImageFont.truetype(FONT, int(size * s))
    l, t_, r, b = f.getbbox(text)
    w, h = r - l, b - t_
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0)); d = ImageDraw.Draw(layer)
    pad_x, pad_y = 34 * s, 20 * s
    box = (W / 2 - w / 2 - pad_x, cy - h / 2 - pad_y, W / 2 + w / 2 + pad_x, cy + h / 2 + pad_y)
    d.rounded_rectangle(box, radius=int(30 * s), fill=(255, 210, 63, int(240 * a)))
    d.text((W / 2, cy), text, font=f, fill=(25, 18, 12, int(255 * a)), anchor="mm")
    img.alpha_composite(layer)


def subscribe_times(plan):
    """(apparition, clic) du bouton d'abonnement : fin de vidéo, quand le dernier compteur est fini."""
    t0 = plan.total - 2.9
    return t0, t0 + 1.1


def subscribe(img, t, t0, t_click):
    """Bouton « SUBSCRIBE » qui apparaît, curseur qui clique, puis « SUBSCRIBED ✓ »."""
    if t < t0:
        return
    W, H = img.size; s = W / 1080
    k = min(1.0, (t - t0) / 0.35)
    sc = 1 - (1 - k) ** 3 * (1 + 2.5 * k)                  # petit rebond
    press = 1 - 0.08 * max(0.0, 1 - abs(t - t_click) / 0.12)
    done = t >= t_click
    cx, cy = W / 2, 0.235 * H
    w, h = 520 * s * sc * press, 120 * s * sc * press
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0)); d = ImageDraw.Draw(layer)
    d.rounded_rectangle((cx - w / 2 + 6 * s, cy - h / 2 + 10 * s, cx + w / 2 + 6 * s, cy + h / 2 + 10 * s), radius=int(h / 2), fill=(0, 0, 0, 110))
    d.rounded_rectangle((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), radius=int(h / 2),
                        fill=(70, 70, 74, 245) if done else (222, 28, 36, 245))
    f = ImageFont.truetype(FONT, max(8, int(56 * s * sc * press)))
    label = "SUBSCRIBED" if done else "SUBSCRIBE"
    tx = cx + (22 * s * sc if done else 0)
    d.text((tx, cy), label, font=f, fill=(255, 255, 255, 255), anchor="mm")
    if done:                                             # coche
        x0, y0 = cx - 205 * s * sc, cy
        d.line([(x0 - 22 * s, y0), (x0 - 6 * s, y0 + 16 * s), (x0 + 22 * s, y0 - 18 * s)], fill=(255, 255, 255, 255), width=int(9 * s))
    # curseur (main stylisée = flèche blanche) qui arrive sur le bouton puis clique
    u = min(1.0, max(0.0, (t - (t0 + 0.35)) / max(0.05, t_click - t0 - 0.35)))
    u = 1 - (1 - u) ** 2
    px, py = cx + 150 * s + (1 - u) * 260 * s, cy + 20 * s + (1 - u) * 300 * s
    cs = 1.6 * s * (0.85 if abs(t - t_click) < 0.12 else 1.0)
    arrow = [(0, 0), (0, 44), (11, 34), (20, 54), (28, 50), (19, 31), (34, 31)]
    pts = [(px + x * cs, py + y * cs) for x, y in arrow]
    d.polygon([(x + 3 * s, y + 4 * s) for x, y in pts], fill=(0, 0, 0, 120))
    d.polygon(pts, fill=(255, 255, 255, 255), outline=(20, 20, 20, 255))
    if abs(t - t_click) < 0.35:                          # onde du clic
        r = (t - t_click + 0.35) / 0.7 * 70 * s
        d.ellipse((px - r, py - r, px + r, py + r), outline=(255, 255, 255, int(200 * (1 - (t - t_click + 0.35) / 0.7))), width=int(5 * s))
    img.alpha_composite(layer)


def compose(plan, src, f):
    img = Image.open(src).convert("RGBA")
    t = f / core.FPS
    cuts, cubes = plan.counters(t)
    _, r = plan.round_at(t)
    pop_cut = 0.0
    if cuts == r["n"] and t < r["t_cut_end"] + 0.4:
        pop_cut = max(0.0, 1 - (t - r["t_cut_end"]) / 0.4)
    pop_cube = max(0.0, 1 - abs(t - (r["t_col"] + 0.9)) / 0.3) if t >= r["t_col"] else 0.0
    H = img.size[1]
    hook = plan.cfg.get("hook")
    if hook and t < plan.cfg.get("hook_dur", 3.0):
        k = min(1.0, (plan.cfg.get("hook_dur", 3.0) - t) / 0.3)        # disparaît en fondu
        pill(img, hook, 0.2 * H, 46, k)
    counter(img, cuts, "CUTS", 0.085 * H, 150, 44, pop_cut)
    counter(img, cubes, "CUBES", 0.765 * H, 96, 40, pop_cube)
    if plan.cfg.get("subscribe", True):
        subscribe(img, t, *subscribe_times(plan))
    return img.convert("RGB")


def main():
    cfg, frames, out = sys.argv[1], sys.argv[2], sys.argv[3]
    plan = core.Plan(core.load(cfg))
    a, b = 0, plan.frames
    if "--frames" in sys.argv:
        i = sys.argv.index("--frames"); a, b = int(sys.argv[i + 1]), int(sys.argv[i + 2])
    first = Image.open(os.path.join(frames, f"f{a:05d}.png"))
    W, H = first.size
    ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                           "-r", str(core.FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "16",
                           "-pix_fmt", "yuv420p", out], stdin=subprocess.PIPE)
    for f in range(a, b):
        p = os.path.join(frames, f"f{f:05d}.png")
        ff.stdin.write(np.asarray(compose(plan, p, f)).tobytes())
    ff.stdin.close(); ff.wait()
    print("écrit", out)


if __name__ == "__main__":
    main()

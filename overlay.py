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
    counter(img, cuts, "CUTS", 0.085 * H, 150, 44, pop_cut)
    counter(img, cubes, "CUBES", 0.765 * H, 96, 40, pop_cube)
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

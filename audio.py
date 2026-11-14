"""Bande-son ASMR synthétisée : coup de couteau à chaque coupe, pluie de petits cubes à l'effondrement.

python3 audio.py objects/steak.json out.wav
"""
import sys, wave
import numpy as np
from scipy.signal import butter, sosfilt, fftconvolve
import core

SR = 48000
rng = np.random.default_rng(3)


def bp(x, a, b):
    return sosfilt(butter(2, [a / (SR / 2), b / (SR / 2)], "band", output="sos"), x)


def env(n, a, d):
    t = np.arange(n) / SR
    return np.minimum(1, t / a) * np.exp(-t / d)


def chop(bright=1.0):
    """Lame qui traverse la viande puis touche la planche."""
    n = int(0.12 * SR)
    slice_ = bp(rng.standard_normal(n), 1800, 7000) * env(n, 0.004, 0.025) * 0.5 * bright
    t = np.arange(n) / SR
    thock = (np.sin(2 * np.pi * 140 * t) * 0.8 + bp(rng.standard_normal(n), 300, 1500) * 0.6) * env(n, 0.001, 0.03)
    out = np.zeros(n + int(0.03 * SR)); out[:n] += slice_; out[int(0.03 * SR):] += thock
    return out


def swish(d):
    """Lame à plat qui glisse (coupes horizontales)."""
    n = int(d * SR)
    x = bp(rng.standard_normal(n), 2500, 9000)
    return x * np.sin(np.linspace(0, np.pi, n)) ** 2 * 0.35


def tick(f):
    n = int(0.05 * SR)
    t = np.arange(n) / SR
    return (np.sin(2 * np.pi * f * t) * 0.5 + bp(rng.standard_normal(n), f * 0.8, min(f * 3, 20000)) * 0.5) * env(n, 0.0005, 0.008)


def main():
    plan = core.Plan(core.load(sys.argv[1]))
    out = np.zeros(int((plan.total + 1) * SR))
    def add(t, x, g):
        i = int(t * SR)
        if i < 0 or i >= len(out):
            return
        j = min(len(out), i + len(x)); out[i:j] += g * x[: j - i]
    for i, r in enumerate(plan.rounds):
        times = r["times"]
        gaps = np.diff(np.concatenate([[r["t0"]], times]))
        for (ax, _), t, gap in zip(r["cuts"], times, gaps):
            g = float(np.clip(np.sqrt(gap / 0.25), 0.18, 1.0))       # coupes rapprochées = plus doux (sinon saturation)
            if ax == 2:
                add(t - min(0.12, gap * 0.45), swish(max(0.03, min(0.24, gap * 0.9))), g)
            else:
                add(t - 0.03, chop(rng.uniform(0.8, 1.2)), g * rng.uniform(0.8, 1.0))
        # effondrement : pluie de petits cubes (échantillon des arrivées)
        C = core.Collapse(plan.shape, r["nxyz"], cap=60_000, seed=i, stack=plan.cfg.get("stack", "z"))
        land = C.delay + C.T
        k = min(len(land), 900)
        pick = rng.choice(len(land), k, replace=False)
        for idx in pick:
            add(r["t_col"] + float(land[idx]), tick(rng.uniform(1800, 5200)), 0.9 / np.sqrt(k / 60))
        n = int(1.4 * SR)
        rumble = bp(rng.standard_normal(n), 60, 400) * env(n, 0.05, 0.4) * 0.6
        add(r["t_col"] + 0.1, rumble, 1.0)
    if plan.cfg.get("subscribe", True):                       # clic sur « SUBSCRIBE » + petite cloche
        import overlay
        t0, tc = overlay.subscribe_times(plan)
        add(tc, tick(3000) + tick(1500) * 0.6, 1.2)
        n = int(0.9 * SR); tt_ = np.arange(n) / SR
        bell = (np.sin(2 * np.pi * 1320 * tt_) + 0.5 * np.sin(2 * np.pi * 2640 * tt_)) * np.exp(-tt_ / 0.25) * 0.35
        add(tc + 0.12, bell, 1.0)
    # petite pièce : réverbération courte
    ir_n = int(0.35 * SR); ir = rng.standard_normal(ir_n) * np.exp(-np.arange(ir_n) / SR / 0.08) * 0.04; ir[0] = 1
    y = fftconvolve(out, ir)[: len(out)]
    y = y / (np.abs(y).max() + 1e-9) * 0.8
    with wave.open(sys.argv[2], "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((y * 32767).astype(np.int16).tobytes())
    print("écrit", sys.argv[2], f"{len(y) / SR:.1f} s")


if __name__ == "__main__":
    main()

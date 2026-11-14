# satisfying-3d

3D "How many cuts can it take?" Shorts, rendered with Blender (Cycles) in GitHub Actions.

- `objects/<name>.json` — the object to cut (shape, rounds of cuts).
- `core.py` — shape, cut timing, cube collapse (shared, no Blender needed).
- `scene.py` — Blender scene + frame rendering (`blender -b -P scene.py -- objects/steak.json out/ --frames A B`).
- `overlay.py` — CUTS / CUBES counters + video encoding.
- `audio.py` — synthesized ASMR sound (chops, slicing, falling cubes).
- Actions → "Render a satisfying Short" → object = `steak`.

Font: Archivo Black (SIL Open Font License, see `fonts/OFL.txt`).

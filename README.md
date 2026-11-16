# satisfying-3d

3D "How many cuts can it take?" Shorts, rendered with Blender (Cycles) in GitHub Actions.

- `objects/<name>.json` — the object to cut (shape, rounds of cuts).
- `core.py` — shape, cut timing, cube collapse (shared, no Blender needed).
- `scene.py` — Blender scene + frame rendering (`blender -b -P scene.py -- objects/steak.json out/ --frames A B`).
- `overlay.py` — CUTS / CUBES counters + video encoding.
- `audio.py` — synthesized ASMR sound (chops, slicing, falling cubes).
- Actions → "Render a satisfying Short" → object = `steak`.

Font: Archivo Black (SIL Open Font License, see `fonts/OFL.txt`).

## The game (`web/`)

"How Many Cuts?": the same concept as a browser game. Swipe across the giant fruit; four rounds (10, 100, 500, 1,000 cuts), then it falls into cubes. The score is the cutting time.

- `web/index.html`, `web/game.js` — page, game logic, 3D scene (Three.js, in `web/vendor/`).
- `web/fruits.js` — the fruits: shapes and colours ported from `objects/*.json` and `scene.py`.
- `web/config.js` — YouTube link and ad keys (Adsterra). Ads stay off until `ads.on` is true and the keys are filled in.
- Run it locally: `cd web && python3 -m http.server 8000`, then open http://localhost:8000.
- Online: the workflow "Deploy the game (GitHub Pages)" publishes `web/` on each push (Settings → Pages → Source = GitHub Actions).


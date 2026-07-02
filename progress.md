# progress.md — shacam-win MVP build log

Build date: 2026-07-02. All 5 phases of `plan.md` implemented, reviewed, tested, and committed.
Repo: https://github.com/Vin124/handy (private).

## How to run

```bash
# deps live in a Python 3.11 venv (mediapipe has no 3.13 wheels)
.venv\Scripts\python.exe -m src.main --mode skeleton
.venv\Scripts\python.exe -m src.main --mode image --image assets/test.png
.venv\Scripts\python.exe -m pytest tests -q
```

Keys: `q` quit · `s` toggle skeleton overlay · `r` reset image transform (image mode) · `+`/`-` keyboard scale (image mode).

## Phase-by-phase

| Phase | Commit | Shipped | Tests |
|-------|--------|---------|-------|
| 0 — Scaffold | `e53a045` | `camera.py`, `fps.py`, `main.py` (argparse, mirror, FPS, `q`) | 9 |
| 1 — Hand skeleton | `55c0061` | `hand_tracker.py` (MediaPipe Tasks API `HandLandmarker`, VIDEO mode, model auto-download to `models/`), `skeleton.py` (21-point drawing) | 23 |
| 2 — Gestures | `24d94e0` | `gestures.py` (pinch w/ hysteresis 0.05 enter / 0.07 exit, grab fist heuristic, frozen `GestureState`), debug label + pinch dot | 34 |
| 3 — Image mode | `8d9a4f6` | `overlay.py` (alpha compositing w/ edge clipping, frozen `Transform`, grab/move/two-hand-resize state machines), `assets/test.png`, `--debug-skeleton` | 69 |
| 4 — Polish | `6f8bb5d` | `hud.py` (mode/FPS/gesture/status strip + key legend), `s`/`r` runtime keys, threshold tuning docs | 90 |
| cleanup | `52f0c56` | deduplicated gesture-label logic (skeleton → hud) | 90 |

## Verification performed

- **Unit tests:** 90 passing (pure logic only: arg parsing, FPS counter, coordinate conversion, landmark conversion, model-download error paths, pinch/hysteresis/grab math, transform clamp/move/scale, compositing incl. all four clip edges + alpha blend on known pixels, grab/resize state machines, HUD text composition, key handling).
- **Live smoke runs (headless-captured):** both modes launch, track, and sustain FPS — skeleton ~25–30 FPS, image mode ~21–25 FPS (two-hand tracking costs more). Model auto-download verified (7.8 MB → `models/`).
- **Not yet human-verified:** pinch threshold *feel*, grab/move/resize ergonomics, HUD legibility — needs your eyeball + webcam. Tune `PINCH_ENTER_THRESHOLD` / `PINCH_EXIT_THRESHOLD` in `src/gestures.py` (tuning guide in comments there).

## Key decisions

- Python 3.11 venv (`py -3.11`), not system 3.13 — mediapipe 0.10.35 has no 3.13 wheels.
- MediaPipe Tasks API in VIDEO mode with strictly-increasing monotonic timestamps; result objects never leak past `hand_tracker.py`.
- All coordinates normalized 0–1 end-to-end; pixel conversion only at draw/composite time (per CLAUDE.md).
- Immutable state throughout: frozen dataclasses with functional updates (`Transform`, `GestureState`, grab/resize/interaction states).
- Pinch hysteresis (enter < exit threshold) kills label flicker; no-hand frames reset the latch so tracking dropouts can't leave phantom pinches.
- Grab requires a pinch *rising edge* over the image (drag-over can't steal a grab); grab offset prevents jump-centering; two-hand resize outranks grab and leaves grab edge-state coherent.
- Center clamped to frame so the image can never be lost off-screen; scale clamped 0.1–6.0.

## Known limitations / follow-ups

- Hand slots can swap between frames in two-hand tracking (MediaPipe ordering) — accepted MVP jitter, commented in `main.py`.
- `composite()` re-runs `cv2.resize` every frame even at unchanged scale — trivial at 300×300, cache if larger assets arrive.
- `is_grab` (fist) is detected and displayed but drives no interaction yet.
- Out of scope per spec: virtual camera, screen capture, iPhone/iPad, multi-image, packaging.

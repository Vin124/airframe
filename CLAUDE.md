# CLAUDE.md

Windows webcam hand-tracking MVP (Shacam-style). Debug harness, not the production app. Python + OpenCV + MediaPipe.

Full context in `@spec.md`. Phased build with per-phase prompts in `@plan.md`.

## Commands
```bash
pip install -r requirements.txt
python -m src.main --mode skeleton            # live hand skeleton over webcam
python -m src.main --mode image --image assets/test.png   # + pinch-to-grab image
```
Quit with `q`.

## Structure
```
src/main.py         # entry: argparse, mode dispatch, main loop
src/camera.py       # webcam open/read/release
src/hand_tracker.py # MediaPipe wrapper -> 21 normalized landmarks per hand
src/skeleton.py     # draw landmarks + connections (skeleton mode)
src/gestures.py     # pinch/grab detection from landmarks (image mode)
src/overlay.py      # image composite + grab/move/resize (image mode)
```

## Conventions
- Every phase in `plan.md` must leave the app runnable — no half-wired states.
- Pass landmarks as normalized coords (0-1); convert to pixels only at draw time.
- Landmark indices: thumb tip = 4, index tip = 8. Pinch = normalized distance(4,8) under a threshold defined as a constant in `gestures.py`.
- Mirror the webcam frame horizontally (selfie view).
- Print FPS to console so we can see if tracking keeps up.

## Gotchas
- Prefer MediaPipe Tasks API (`HandLandmarker`, `hand_landmarker.task`) in VIDEO/LIVE_STREAM mode; fall back to legacy `mp.solutions.hands` only if the installed version makes that simpler.
- No virtual camera, screen capture, or iPhone/iPad in this MVP - display in an OpenCV window only.
- Single hand, single image is fine for the MVP.

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
Pipeline: webcam **source** -> **scene** of layers -> **compositor** -> preview **sink**.
```
src/main.py           # entry: argparse, mode dispatch, main loop (wires the pipeline)
src/camera.py         # low-level webcam open/read/release
src/hand_tracker.py   # MediaPipe wrapper -> 21 normalized landmarks per hand
src/skeleton.py       # draw landmarks + connections (debug overlay)
src/gestures.py       # pinch/grab detection from landmarks
src/hud.py            # heads-up display text + strip
src/fps.py            # frame-rate counter
src/sources/          # frame producers (Source contract) -> latest frame as numpy array
    base.py           #   Source ABC + SourceError
    webcam.py         #   WebcamSource: live feed, applies the selfie mirror
    image.py          #   ImageSource: loaded BGRA overlay image
src/scene.py          # Transform geometry + Layer (source+transform+z_index+selected) + Scene
src/compositor.py     # alpha-composite layers onto the webcam frame in place, by z-order
src/interaction.py    # grab/move/resize state machines over the selected layer
src/sinks/            # frame consumers (Sink contract)
    base.py           #   Sink ABC + SinkError
    preview.py        #   PreviewSink: cv2.imshow window
    virtualcam.py     #   VirtualCamSink: pyvirtualcam broadcast (OBS backend)
```

## Conventions
- Every phase in `plan.md` must leave the app runnable — no half-wired states.
- Pass landmarks as normalized coords (0-1); convert to pixels only at draw time.
- Landmark indices: thumb tip = 4, index tip = 8. Pinch = normalized distance(4,8) under a threshold defined as a constant in `gestures.py`.
- Mirror the webcam frame horizontally (selfie view).
- Print FPS to console so we can see if tracking keeps up.

## Gotchas
- Prefer MediaPipe Tasks API (`HandLandmarker`, `hand_landmarker.task`) in VIDEO/LIVE_STREAM mode; fall back to legacy `mp.solutions.hands` only if the installed version makes that simpler.
- Virtual camera output is opt-in (`--virtualcam` flag / 'v' key) via pyvirtualcam and needs the OBS Virtual Camera backend (OBS Studio 26+). No screen capture (Phase 7) or iPhone/iPad (not planned) yet.
- Single hand, single image is fine for the MVP.

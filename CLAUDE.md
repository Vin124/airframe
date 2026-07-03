# CLAUDE.md

Windows webcam hand-tracking MVP (Shacam-style). Debug harness, not the production app. Python + OpenCV + MediaPipe.

Full context in `@spec.md`. Phased build with per-phase prompts in `@plan.md`.

## Commands
```bash
pip install -r requirements.txt
python -m src.main --mode skeleton            # live hand skeleton over webcam
python -m src.main --mode image --image assets/test.png   # + pinch-to-grab image
python -m src.main --mode image --source window --window-title "Notepad"  # grab a live window
python -m src.main --mode image --source screen --monitor-index 1         # grab a monitor (1-based)
```
Runtime keys: `q` quit, `s` toggle skeleton, `r` reset transform, `+/-` scale,
`v` toggle virtual cam. Image-mode multi-layer: `a`/`w`/`m` add an image / window
/ screen layer (reusing the launch args), `x` remove the selected layer, `f`
bring it to the front. A pinch grabs the top-most layer under the pinch point;
the selected layer is outlined.

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
    capture.py        #   WindowSource / MonitorSource: live Windows Graphics Capture (threaded)
src/scene.py          # Transform geometry + Layer (source+transform+z_index+selected+layer_id) + Scene (multi-layer add/remove/raise, hit-test selection)
src/compositor.py     # alpha-composite layers onto the webcam frame in place by z-order + draw the selected-layer highlight
src/interaction.py    # grab/move/resize state machines; step() resolves top-most selection on pinch start, drives the selected layer
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
- Virtual camera output is opt-in (`--virtualcam` flag / 'v' key) via pyvirtualcam and needs the OBS Virtual Camera backend (OBS Studio 26+).
- Live window/screen overlays (`--source window|screen`, image mode) use `windows-capture` (Windows Graphics Capture) on their own thread; frames are copied off the reused native buffer (latest-frame-wins, no lock). Monitor index is 1-based (1 = primary). A window that isn't found fails fast; a window closing mid-run leaves the app running (overlay disappears). iPhone/iPad mirroring is not planned.
- Multiple overlay layers are supported at runtime (Phase 8): add/remove layers,
  pinch to grab whichever is under the pinch point, bring-to-front, all live.
  Selection is single (exactly one active layer); layers carry a stable
  `layer_id` so remove/raise/fit target a layer across move/resize copies.

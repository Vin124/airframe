# CLAUDE.md

## Project
**shacam-win** — early MVP of a webcam hand-tracking tool (Shacam-style), for Windows, built to run on a laptop webcam.

This repo is **not** the production app. It's a debug harness to prove out real-time hand tracking and gesture-driven image manipulation before any of the heavy production pieces (virtual camera driver, DirectX compositing) get built.

## What we're building (MVP scope only)
Two runnable modes, selected at launch:

1. **`skeleton` mode** — open the webcam, run hand tracking, draw the 21-point hand skeleton live over the video feed. Pure debug view. This is the first thing to get working and must be testable on the laptop webcam.
2. **`image` mode** — same webcam + hand tracking, but with a loaded image on screen that the hand can **pinch to grab, move, and resize**. This is `skeleton` mode plus a gesture layer plus a draggable overlay.

Mode 2 is a superset of mode 1. Build mode 1 first, fully working, then layer mode 2 on top.

## Out of scope for MVP Chunk 1 (since lifted by `plan-2.md`)
Chunk 2 lifts most of the original exclusions: virtual camera output shipped in
Phase 6 (`--virtualcam` / 'v' key via pyvirtualcam + OBS Virtual Camera); live
window/screen capture shipped in Phase 7 (`--source window|screen` via
windows-capture); multi-layer manipulation follows in Phase 8.
Still out of scope:
- iPhone/iPad mirroring
- Any packaging/installer (Chunk 3)

## Tech stack
- **Python 3.11**
- **OpenCV** (`opencv-python`) — webcam capture + window display + drawing
- **MediaPipe** — hand landmark detection (21 landmarks per hand). Prefer the Tasks API (`HandLandmarker`, `hand_landmarker.task` model) in LIVE_STREAM/VIDEO mode; fall back to the legacy `mp.solutions.hands` if the installed version makes that simpler. Whatever runs real-time.
- **NumPy** — array/coordinate math

## Project structure
As of Chunk 2 / Phase 5 the loose MVP modules are refactored into a pipeline:
webcam **source** -> **scene** of layers -> **compositor** -> preview **sink**,
with gesture-driven **interaction** on the selected layer. The MVP scope above
(two modes, single hand, single image, OpenCV window only) is unchanged — this
is purely an internal restructure so later features plug in cleanly.
```
shacam-win/
├── CLAUDE.md
├── plan.md
├── plan-2.md               # Chunk 2 phases (5-10)
├── requirements.txt
├── assets/
│   └── test.png            # any test image for image mode
└── src/
    ├── main.py             # entry point, arg parsing, mode dispatch, main loop (wires the pipeline)
    ├── camera.py           # low-level webcam open/read/release wrapper
    ├── hand_tracker.py     # MediaPipe wrapper -> normalized 21-landmark list
    ├── skeleton.py         # draw landmarks + connections (debug overlay)
    ├── gestures.py         # pinch / grab detection from landmarks
    ├── hud.py              # heads-up display text + strip
    ├── fps.py              # frame-rate counter
    ├── sources/            # frame producers (Source contract) -> latest frame as numpy array
    │   ├── base.py         #   Source ABC + SourceError
    │   ├── webcam.py       #   WebcamSource: live feed, applies the selfie mirror
    │   ├── image.py        #   ImageSource: loaded BGRA overlay image
    │   └── capture.py      #   WindowSource / MonitorSource: live Windows Graphics Capture (threaded)
    ├── scene.py            # Transform geometry + Layer (source+transform+z_index+selected) + Scene
    ├── compositor.py       # alpha-composite layers onto the webcam frame in place, by z-order
    ├── interaction.py      # grab/move/resize state machines over the selected layer
    └── sinks/              # frame consumers (Sink contract)
        ├── base.py         #   Sink ABC + SinkError
        ├── preview.py      #   PreviewSink: cv2.imshow window
        └── virtualcam.py   #   VirtualCamSink: pyvirtualcam broadcast (OBS backend)
```

## How to run
```bash
pip install -r requirements.txt
python -m src.main --mode skeleton
python -m src.main --mode image --image assets/test.png
python -m src.main --mode image --source window --window-title "Notepad"  # live window overlay
python -m src.main --mode image --source screen --monitor-index 1         # live monitor overlay (1-based)
```
Press `q` to quit. (A runtime key to toggle modes is a nice-to-have, not required.)

## Conventions
- **Keep every increment runnable.** After each subplan the app must launch and show *something* on the webcam. No half-wired states.
- Hand landmarks are passed around as **normalized coordinates (0–1)**; convert to pixels only at draw time.
- Keep modules single-purpose per the structure above; `main.py` owns the loop, everything else is a helper.
- Prefer clarity over cleverness — this is a debug tool. Print FPS to the console so we can see if tracking keeps up.
- Standard landmark indices: thumb tip = 4, index tip = 8. **Pinch** = normalized distance(4, 8) below a tuned threshold.

## Definition of done for the MVP
- `skeleton` mode: live webcam window with a stable hand skeleton tracking the hand at usable frame rate.
- `image` mode: a hand pinch grabs the image, moving the hand moves it, and there's a working resize gesture.

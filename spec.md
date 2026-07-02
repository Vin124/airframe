# CLAUDE.md

## Project
**shacam-win** — early MVP of a webcam hand-tracking tool (Shacam-style), for Windows, built to run on a laptop webcam.

This repo is **not** the production app. It's a debug harness to prove out real-time hand tracking and gesture-driven image manipulation before any of the heavy production pieces (virtual camera driver, DirectX compositing) get built.

## What we're building (MVP scope only)
Two runnable modes, selected at launch:

1. **`skeleton` mode** — open the webcam, run hand tracking, draw the 21-point hand skeleton live over the video feed. Pure debug view. This is the first thing to get working and must be testable on the laptop webcam.
2. **`image` mode** — same webcam + hand tracking, but with a loaded image on screen that the hand can **pinch to grab, move, and resize**. This is `skeleton` mode plus a gesture layer plus a draggable overlay.

Mode 2 is a superset of mode 1. Build mode 1 first, fully working, then layer mode 2 on top.

## Out of scope for MVP (do NOT build yet)
- Virtual camera output (no `pyvirtualcam` yet — we display in an OpenCV window)
- Window/screen capture
- iPhone/iPad mirroring
- Multi-hand or multi-object manipulation (single hand, single image is fine)
- Any packaging/installer

## Tech stack
- **Python 3.11**
- **OpenCV** (`opencv-python`) — webcam capture + window display + drawing
- **MediaPipe** — hand landmark detection (21 landmarks per hand). Prefer the Tasks API (`HandLandmarker`, `hand_landmarker.task` model) in LIVE_STREAM/VIDEO mode; fall back to the legacy `mp.solutions.hands` if the installed version makes that simpler. Whatever runs real-time.
- **NumPy** — array/coordinate math

## Project structure
```
shacam-win/
├── CLAUDE.md
├── plan.md
├── requirements.txt
├── assets/
│   └── test.png            # any test image for mode 2
└── src/
    ├── main.py             # entry point, arg parsing, mode dispatch, main loop
    ├── camera.py           # webcam open/read/release wrapper
    ├── hand_tracker.py     # MediaPipe wrapper -> normalized 21-landmark list
    ├── skeleton.py         # draw landmarks + connections (mode 1)
    ├── gestures.py         # pinch / grab detection from landmarks (mode 2)
    └── overlay.py          # image overlay + grab/move/resize logic (mode 2)
```

## How to run
```bash
pip install -r requirements.txt
python -m src.main --mode skeleton
python -m src.main --mode image --image assets/test.png
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

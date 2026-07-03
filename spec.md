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
windows-capture); multi-layer manipulation shipped in Phase 8 (add/remove/raise
layers at runtime, pinch grabs the top-most layer under the pinch point); a
Tkinter control panel + JSON settings persistence shipped in Phase 10
(`src/panel.py`, `src/config.py`, `--no-panel` to opt out).
Still out of scope:
- iPhone/iPad mirroring
- Any packaging/installer (Chunk 3)
- A full ONNX Runtime / TensorRT port of the hand model (Chunk 3). Phase 9 adds a
  best-effort `--gpu` flag that only requests MediaPipe's built-in GPU delegate
  and cleanly falls back to CPU; porting the model to a dedicated GPU runtime for
  guaranteed acceleration is deferred.

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
As of Phase 9 the pipeline stages run on separate threads (see "Threading model"
below): the webcam capture and the hand tracking each get their own thread, and
the main loop only composites + displays, paced to a target FPS.
```
shacam-win/
├── CLAUDE.md
├── plan.md
├── plan-2.md               # Chunk 2 phases (5-10)
├── requirements.txt
├── assets/
│   └── test.png            # any test image for image mode
└── src/
    ├── main.py             # entry point, arg parsing, mode dispatch, PipelineRuntime (main loop), thread wiring
    ├── camera.py           # low-level webcam open/read/release wrapper
    ├── hand_tracker.py     # MediaPipe wrapper -> normalized 21-landmark list; --gpu delegate + CPU fallback
    ├── tracking.py         # TrackingWorker: hand tracking + gesture hysteresis on its own thread (Phase 9)
    ├── timing.py           # pure stage-timer + loop-pacing + EMA-smoothing helpers (Phase 9)
    ├── skeleton.py         # draw landmarks + connections (debug overlay)
    ├── gestures.py         # pinch / grab detection from landmarks
    ├── hud.py              # heads-up display text + strip (incl. per-stage timing line)
    ├── fps.py              # frame-rate counter (main-loop rate)
    ├── panel.py            # Tkinter control panel + PanelActions controller (Phase 10)
    ├── config.py           # pure load/save/merge of the panel's last settings (handy_config.json)
    ├── sources/            # frame producers (Source contract) -> latest frame as numpy array
    │   ├── base.py         #   Source ABC + SourceError
    │   ├── webcam.py       #   WebcamSource: live feed, applies the selfie mirror (blocking read)
    │   ├── threaded.py     #   ThreadedSource: runs any blocking Source on a capture thread (Phase 9)
    │   ├── image.py        #   ImageSource: loaded BGRA overlay image
    │   └── capture.py      #   WindowSource / MonitorSource: live Windows Graphics Capture (threaded)
    ├── scene.py            # Transform geometry + Layer (source+transform+z_index+selected+layer_id) + Scene (multi-layer add/remove/raise + hit-test selection)
    ├── compositor.py       # alpha-composite layers onto the webcam frame in place by z-order + selected-layer highlight
    ├── interaction.py      # grab/move/resize state machines; step() resolves top-most selection on pinch start, drives the selected layer
    └── sinks/              # frame consumers (Sink contract)
        ├── base.py         #   Sink ABC + SinkError
        ├── preview.py      #   PreviewSink: cv2.imshow window
        └── virtualcam.py   #   VirtualCamSink: pyvirtualcam broadcast (OBS backend)
```

### Threading model (Phase 9)
Three threads, each the sole owner of its state; hand-offs are a single atomic
reference assignment (latest-frame-wins, GIL-safe, no locks):
- **Capture thread** — `ThreadedSource` wraps the blocking `WebcamSource` read and
  publishes each frame as an `(seq, frame)` packet. `PipelineRuntime` never blocks
  on the camera; if no newer frame exists it re-uses the last one. Published
  frames are READ-ONLY to consumers (the compositor draws on a private copy).
- **Tracking thread** — `TrackingWorker` owns the `HandTracker` and the
  `GestureDetector`s (their frame-to-frame pinch hysteresis). It consumes the
  newest webcam frame (dedup by `seq` so it never re-detects the same pixels),
  runs detect + gesture update at its own cadence, and publishes an immutable
  `TrackingResult` (hands, gesture states, per-update `track_ms`). All MediaPipe
  access is confined to this thread (VIDEO mode requires monotonic timestamps and
  single-threaded use). A tracker exception is printed once and freezes the last
  result — the app stays alive rather than dying silently.
- **Main loop** — `PipelineRuntime` is the single writer of scene/interaction. It
  reads the two threads' latest published values, composites + draws the HUD +
  feeds the sinks, and paces itself to `TARGET_FPS` (30) by handing the remaining
  frame budget to `cv2.waitKey` (which also renders the window and reads keys).
  The HUD shows per-stage timings (capture age, track, composite, sink, loop).

`PipelineRuntime` lives in `main.py` (rather than its own module) so it can share
the already-unit-tested loop helpers without a circular import; `run()` is a thin
setup/teardown shell around it.

## How to run
```bash
pip install -r requirements.txt
python -m src.main --mode skeleton
python -m src.main --mode image --image assets/test.png
python -m src.main --mode image --source window --window-title "Notepad"  # live window overlay
python -m src.main --mode image --source screen --monitor-index 1         # live monitor overlay (1-based)
python -m src.main --mode skeleton --gpu                                   # request GPU delegate (CPU fallback)
```
`--gpu` requests MediaPipe's built-in GPU delegate for the hand model and falls
back to CPU with a one-line notice when it's unavailable (the expected case for
MediaPipe's Python wheels on Windows). Any of the flags above combine with
`--virtualcam` to also broadcast the composited feed.
Press `q` to quit. Other runtime keys: `s` toggle skeleton, `r` reset transform,
`+/-` scale, `v` toggle virtual cam. Image-mode multi-layer keys: `a`/`w`/`m` add
an image / window / screen layer (reusing the launch args), `x` remove the
selected layer, `f` bring it to front. (A runtime key to toggle modes is a
nice-to-have, not required.)

### Control panel + config (Phase 10)
By default a small Tkinter control panel opens next to the preview so the app is
usable without memorizing hotkeys: choose a source (image via a file picker,
window by title, monitor by index) and add it as a layer, toggle the virtual
cam, reset the scene (removes all layers, closing each source), and read a live
FPS value. The panel drives the same `PipelineRuntime` methods the hotkeys use
(it is pumped once per main-loop tick on the main thread — Tkinter is not
thread-safe — so button callbacks run inline between frames and the main loop
stays the single writer of scene/interaction). Pass `--no-panel` to run
hotkey-only (headless / keyboard-only). The last source choice, image path,
window title, monitor index, and virtual-cam state persist to
`handy_config.json` (repo root, git-ignored) on exit and pre-fill the panel on
start. Startup precedence is **CLI flags > config > built-in defaults**; a
missing or corrupt config falls back to defaults with a stderr notice, never a
crash. See `src/config.py` (pure load/save/merge) and `src/panel.py`
(controller + Tk view).

## Conventions
- **Keep every increment runnable.** After each subplan the app must launch and show *something* on the webcam. No half-wired states.
- Hand landmarks are passed around as **normalized coordinates (0–1)**; convert to pixels only at draw time.
- Keep modules single-purpose per the structure above; `main.py` owns the loop, everything else is a helper.
- Prefer clarity over cleverness — this is a debug tool. Print FPS to the console so we can see if tracking keeps up.
- Standard landmark indices: thumb tip = 4, index tip = 8. **Pinch** = normalized distance(4, 8) below a tuned threshold.

## Definition of done for the MVP
- `skeleton` mode: live webcam window with a stable hand skeleton tracking the hand at usable frame rate.
- `image` mode: a hand pinch grabs the image, moving the hand moves it, and there's a working resize gesture.

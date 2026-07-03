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
python -m src.main --mode skeleton --gpu      # request GPU delegate (falls back to CPU on Windows)
```
Runtime keys: `q` quit, `s` toggle skeleton, `r` reset transform, `+/-` scale,
`v` toggle virtual cam. Image-mode multi-layer: `a`/`w`/`m` add an image / window
/ screen layer (reusing the launch args), `x` remove the selected layer, `f`
bring it to the front. A pinch grabs the top-most layer under the pinch point;
the selected layer is outlined.

## Structure
Pipeline: webcam **source** -> **scene** of layers -> **compositor** -> preview **sink**.
As of Phase 9 the stages are decoupled across threads (see Threading model below).
```
src/main.py           # entry: argparse, mode dispatch, PipelineRuntime (main loop), thread wiring
src/runtime note      #   PipelineRuntime lives in main.py (kept there to avoid a circular import
                      #   with the tested loop helpers); run() is a thin setup/teardown shell
src/camera.py         # low-level webcam open/read/release
src/hand_tracker.py   # MediaPipe wrapper -> 21 normalized landmarks per hand; --gpu delegate + CPU fallback
src/tracking.py       # TrackingWorker: hand tracking + gesture hysteresis on its own thread (Phase 9)
src/timing.py         # pure stage-timer + loop-pacing + EMA-smoothing helpers (Phase 9)
src/skeleton.py       # draw landmarks + connections (debug overlay)
src/gestures.py       # pinch/grab detection from landmarks
src/hud.py            # heads-up display text + strip (incl. per-stage timing line)
src/fps.py            # frame-rate counter (loop rate)
src/sources/          # frame producers (Source contract) -> latest frame as numpy array
    base.py           #   Source ABC + SourceError
    webcam.py         #   WebcamSource: live feed, applies the selfie mirror (blocking read)
    threaded.py       #   ThreadedSource: runs any blocking Source on a capture thread (Phase 9)
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

## Threading model (Phase 9)
Three threads, each a single owner of its state (no locks — one atomic reference
publish per hand-off, GIL-safe, latest-frame-wins):
- **Capture thread** (`ThreadedSource` around `WebcamSource`): does the blocking
  `cv2` read and publishes each frame as an `(seq, frame)` packet. Frames are
  READ-ONLY to consumers — copy before mutating.
- **Tracking thread** (`TrackingWorker`): OWNS the `HandTracker` and the
  `GestureDetector`s (their pinch hysteresis). Consumes the newest webcam frame
  (dedup by `seq`), runs detect + gesture update at its own cadence, publishes an
  immutable `TrackingResult`. All MediaPipe access stays on this one thread
  (VIDEO mode needs monotonic timestamps + single-threaded use). On a tracker
  exception it prints once and freezes the last result — the app stays alive.
- **Main loop** (`PipelineRuntime`): the SINGLE WRITER of scene/interaction. Reads
  the two threads' latest published values, composites + draws HUD + feeds sinks,
  paced to `TARGET_FPS` (30) via a post-work `cv2.waitKey` of the remaining
  budget. Never blocks on the camera (reuses the last frame if none is newer).

## Conventions
- Every phase in `plan.md` must leave the app runnable — no half-wired states.
- Pass landmarks as normalized coords (0-1); convert to pixels only at draw time.
- Landmark indices: thumb tip = 4, index tip = 8. Pinch = normalized distance(4,8) under a threshold defined as a constant in `gestures.py`.
- Mirror the webcam frame horizontally (selfie view).
- Print FPS to console so we can see if tracking keeps up (FPS = main-loop rate).
- Threading (Phase 9): one owner per piece of state, hand off with a single
  atomic reference publish (latest-frame-wins, no locks). The main loop is the
  only writer of scene/interaction; the tracking thread is the only toucher of
  the tracker + gesture detectors. Treat published frames as read-only.

## Gotchas
- Prefer MediaPipe Tasks API (`HandLandmarker`, `hand_landmarker.task`) in VIDEO/LIVE_STREAM mode; fall back to legacy `mp.solutions.hands` only if the installed version makes that simpler.
- `--gpu` requests MediaPipe's built-in GPU delegate and falls back to CPU with a one-line notice when it's unavailable — which is the expected case for MediaPipe's Python wheels on Windows. A full ONNX Runtime / TensorRT port of the hand model is deferred to Chunk 3.
- Virtual camera output is opt-in (`--virtualcam` flag / 'v' key) via pyvirtualcam and needs the OBS Virtual Camera backend (OBS Studio 26+).
- Live window/screen overlays (`--source window|screen`, image mode) use `windows-capture` (Windows Graphics Capture) on their own thread; frames are copied off the reused native buffer (latest-frame-wins, no lock). Monitor index is 1-based (1 = primary). A window that isn't found fails fast; a window closing mid-run leaves the app running (overlay disappears). iPhone/iPad mirroring is not planned.
- Multiple overlay layers are supported at runtime (Phase 8): add/remove layers,
  pinch to grab whichever is under the pinch point, bring-to-front, all live.
  Selection is single (exactly one active layer); layers carry a stable
  `layer_id` so remove/raise/fit target a layer across move/resize copies.

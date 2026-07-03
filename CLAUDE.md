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
python -m src.main --mode image --image assets/test.png --no-panel  # hotkeys only, no control panel
python -m src.main --mode skeleton --min-detection-confidence 0.7 --min-tracking-confidence 0.6  # startup-only MediaPipe confidences
python -m src.main --mode skeleton --speech    # + always-listening mic transcription printed to console (Phase 12)
python -m src.main --mode skeleton --speech --parse   # + local-LLM parse of each finalized phrase into a structured request (Phase 13, needs Ollama running)
```
A small **Tkinter control panel** opens alongside the preview by default (Phase
10): pick a source (image file picker / editable window-title dropdown / monitor
index) and add it as a layer, toggle the virtual cam, reset the scene, retune
pinch detection with two live sliders (pinch ENTER / EXIT threshold), watch a
live FPS readout. The window-title dropdown re-enumerates open top-level windows
each time it opens (`src/window_list.py`, ctypes `EnumWindows`, no new dep) and
stays editable so a custom substring still works. Its buttons/sliders drive the
same runtime the hotkeys use. Pass `--no-panel` to run hotkey-only. Last-used
panel settings persist to `handy_config.json` (repo root, git-ignored) on exit
and pre-fill on start; precedence is **CLI > config > defaults**.

Persisted config keys: `source`, `image_path`, `window_title`, `monitor_index`,
`virtualcam`, `speech` (Phase 12 opt-in, `--speech` store_true, remembered like
`virtualcam`), `parse` (Phase 13 opt-in, `--parse` store_true, remembered the
same way; only meaningful when `speech` is also on), `pinch_enter` /
`pinch_exit` (live slider thresholds), and `min_detection_confidence` /
`min_presence_confidence` / `min_tracking_confidence`
(startup-only MediaPipe knobs, also `--min-*-confidence` flags).

Runtime keys: `q` quit, `s` toggle skeleton, `r` reset transform, `+/-` scale,
`v` toggle virtual cam. Image-mode multi-layer: `a`/`w`/`m` add an image / window
/ screen layer (reusing the launch args), `x` remove the selected layer, `f`
bring it to the front. A pinch grabs the top-most layer under the pinch point;
the selected layer is outlined. Image-mode air-drawing (Phase 11): hold the draw
pose (index extended, other fingers curled) and move your fingertip to draw a
smoothed stroke; release the pose to freeze it into a grabbable/movable layer.
`c` clears the in-progress stroke, `k` cycles stroke color, `[` / `]` step the
stroke width down / up.

## Structure
Pipeline: webcam **source** -> **scene** of layers -> **compositor** -> preview **sink**.
As of Phase 9 the stages are decoupled across threads (see Threading model below).
```
src/main.py           # entry: argparse, mode dispatch, PipelineRuntime (main loop), thread wiring; re-exports the moved helpers/constants (layers, launch) for the test surface
src/layers.py         # scene/source-layer transitions (build/add/remove/fit) + the mode/source/layer-key vocabulary; camera-free + unit-testable
src/launch.py         # launch glue: CLI parsing, panel prefill, config <-> args, panel build (lazy Tk import)
src/runtime note      #   PipelineRuntime lives in main.py (kept there to avoid a circular import
                      #   with the tested loop helpers in layers.py); run() is a thin setup/teardown shell
src/camera.py         # low-level webcam open/read/release
src/hand_tracker.py   # MediaPipe wrapper -> 21 normalized landmarks per hand; --gpu delegate + CPU fallback
src/tracking.py       # TrackingWorker: hand tracking + gesture hysteresis on its own thread (Phase 9)
src/speech.py         # SpeechWorker: mic capture + energy-VAD phrase segmentation + ASR (Moonshine -> faster-whisper fallback) on its own thread (Phase 12); pure VAD + rolling-transcript helpers; deferred deps; speech_session() launch glue
src/parser.py          # RequestParser: one finalized phrase -> a strict-JSON structured ParsedRequest via a local Ollama model (Phase 13); pure prompt-build + JSON-extract + enum-normalize helpers are the security boundary; OllamaClient speaks Ollama's HTTP API over stdlib urllib (no new dep); wired as (part of) the speech thread's on_phrase callback, see src/launch.py's build_speech_on_phrase
src/timing.py         # pure stage-timer + loop-pacing + EMA-smoothing helpers (Phase 9)
src/skeleton.py       # draw landmarks + connections (debug overlay)
src/gestures.py       # pinch/grab/draw detection from landmarks; per-instance enter/exit thresholds (live-tunable); draw-pose N-frame confirmation
src/drawing.py        # air-drawing (Phase 11): immutable Stroke + smoothing + rasterize-to-BGRA + bbox->Transform placement + live overlay; pure numpy/cv2
src/tuning.py         # dependency-light GestureTuning + TrackerConfidence snapshots + clamp_tuning (the one hysteresis-gap boundary, shared by panel + config)
src/window_list.py    # list_window_titles(): open top-level window titles via ctypes EnumWindows (panel dropdown); injectable + failure-safe ([])
src/hud.py            # heads-up display text + strip (incl. per-stage timing line)
src/fps.py            # frame-rate counter (loop rate)
src/panel.py          # Tkinter control panel + PanelActions controller (Phase 10); pumped by PipelineRuntime
src/config.py         # pure load/save/merge of the panel's last settings (handy_config.json)
src/sources/          # frame producers (Source contract) -> latest frame as numpy array
    base.py           #   Source ABC + SourceError
    webcam.py         #   WebcamSource: live feed, applies the selfie mirror (blocking read)
    threaded.py       #   ThreadedSource: runs any blocking Source on a capture thread (Phase 9)
    image.py          #   ImageSource: loaded BGRA overlay image; ArraySource: in-memory BGRA (frozen air-drawing strokes)
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
Three threads (a fourth, the speech thread, is added optionally in Phase 12),
each a single owner of its state (no locks — one atomic reference publish per
hand-off, GIL-safe, latest-frame-wins):
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
- **Speech thread** (`SpeechWorker`, Phase 12, OPTIONAL — only when `--speech`):
  OWNS the mic stream + the ASR engine. Pulls ~0.2s mic blocks, segments phrases
  by energy-VAD, transcribes only finalized phrases, and publishes an immutable
  rolling `Transcript` (one atomic ref store, latest-wins) plus an optional
  `on_phrase` callback per phrase. Fully off the main loop, so a ~1s transcribe
  never drops render/tracking FPS. NOTE: `on_phrase` runs ON the speech thread.
  Bracketed by `speech_session()`; a missing mic / no engine degrades to no-op.

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
- Pinch thresholds are LIVE-tunable via the panel sliders: a slider publishes a
  frozen `GestureTuning` from the main thread to the tracking thread (one atomic
  ref store, `TrackingWorker.set_tuning`), which swaps it onto the detectors it
  owns — the main thread never touches detector internals. `clamp_tuning` (in
  `src/tuning.py`) is the ONE boundary enforcing `exit > enter` (a minimum gap),
  used by BOTH the slider path and config load, so an inverted/equal pair can
  never reintroduce pinch-label flicker.
- The three MediaPipe min-confidence params (`--min-detection-confidence`,
  `--min-presence-confidence`, `--min-tracking-confidence`) are STARTUP-ONLY:
  they bake into the `HandLandmarker` options, so changing them needs a tracker
  re-init. Hence they are CLI/config flags, NOT live sliders (unlike pinch
  thresholds). Persisted + resolved with the same CLI > config > default rule.
- `--gpu` requests MediaPipe's built-in GPU delegate and falls back to CPU with a one-line notice when it's unavailable — which is the expected case for MediaPipe's Python wheels on Windows. A full ONNX Runtime / TensorRT port of the hand model is deferred to Chunk 3.
- Virtual camera output is opt-in (`--virtualcam` flag / 'v' key) via pyvirtualcam and needs the OBS Virtual Camera backend (OBS Studio 26+).
- Speech (Phase 12, `--speech`, `src/speech.py`): `select_engine` tries **Moonshine
  first, then faster-whisper, then no-speech**, each fallback a one-line notice
  (degrade, never die — the app's failure policy). Every heavy dep
  (`moonshine_onnx`, `faster_whisper`, `sounddevice`) is imported LAZILY inside
  the class that needs it, so the base pipeline runs with NONE installed (mirrors
  the deferred `pyvirtualcam` import). Install: `pip install sounddevice
  faster-whisper` and, for the preferred engine, `pip install
  useful-moonshine-onnx` (the PyPI wheel — the moonshine repo's git-subdirectory
  install no longer builds). A missing/failed mic prints one stderr notice and the
  worker exits cleanly. The `on_phrase` callback (and the console `[speech]` print)
  runs ON the speech thread — keep it cheap/thread-safe. Energy-VAD constants live
  at the top of `src/speech.py` (`VAD_RMS_THRESHOLD`, `SILENCE_GAP_SECONDS`,
  `MAX_PHRASE_SECONDS`, `ROLLING_TRANSCRIPT_SIZE`); only finalized phrases are
  transcribed (no streaming) to keep CPU/latency low.
- LLM parser (Phase 13, `--parse`, `src/parser.py`): only meaningful together
  with `--speech` (`src.launch.build_speech_on_phrase` wires it into
  `SpeechWorker`'s `on_phrase`, so it runs ON THE SPEECH THREAD, same
  cheap/thread-safe rule as the transcript print). `RequestParser.parse` never
  raises: an `OllamaClient` HTTP failure (server down, model not pulled) or any
  malformed/non-JSON model reply prints one stderr notice and returns `None`
  for that phrase, degrading to Phase 12's transcript-only line — Ollama being
  unreachable can't crash the app. Model output is NEVER trusted directly:
  `parse_llm_json` extracts the first balanced `{...}` block (tolerating stray
  prose/markdown fences smaller models add despite the "JSON only" prompt) and
  `normalize_route` / `normalize_source` clamp every field to a fixed enum
  (unknown route -> `retrieve`, unknown source -> `wikimedia`, `generate`
  routes always force `source="generator"`) — this is the ONE validation
  boundary Phase 14/16 can rely on downstream. A non-`insert` action or a blank
  query both parse to `None` (ignored), matching "ignore non-actionable
  speech." No new pip dependency: `OllamaClient` speaks Ollama's REST API
  (`/api/generate`, `format="json"`, temperature 0) via stdlib `urllib.request`
  — only the separate Ollama application + a pulled small model are required
  at runtime (see `requirements.txt`).
- Live window/screen overlays (`--source window|screen`, image mode) use `windows-capture` (Windows Graphics Capture) on their own thread; frames are copied off the reused native buffer (latest-frame-wins, no lock). Monitor index is 1-based (1 = primary). A window that isn't found fails fast; a window closing mid-run leaves the app running (overlay disappears). iPhone/iPad mirroring is not planned.
- Air-drawing (Phase 11, image mode only): the DRAW-POSE detection + its N-frame
  confirmation hysteresis live in `GestureDetector` on the TRACKING thread (it
  publishes `is_draw` + the index-tip `draw_point` on the frozen `GestureState`);
  the main loop NEVER computes a gesture from raw landmarks. The accumulating
  `Stroke` is owned by the MAIN loop (single writer of scene/interaction), which
  appends `draw_point` while `is_draw` holds and, on release, rasterizes it
  (`src/drawing.py`) into an `ArraySource` layer placed via `bbox_to_transform`
  so it lands exactly where drawn. Gesture precedence: `layers.suppress_draw_pinch`
  masks `is_pinching=False` for any drawing hand BEFORE `interaction.step`, so a
  curled draw pose can't also start a pinch-grab. Keys c/k/[/] are image-mode only.
- Multiple overlay layers are supported at runtime (Phase 8): add/remove layers,
  pinch to grab whichever is under the pinch point, bring-to-front, all live.
  Selection is single (exactly one active layer); layers carry a stable
  `layer_id` so remove/raise/fit target a layer across move/resize copies.
- The control panel (Phase 10) is Tkinter, which is NOT thread-safe: it runs on
  the MAIN thread, created after the camera opens and pumped once per tick by
  `PipelineRuntime._pump_panel` (a non-blocking `root.update()`). Button
  callbacks therefore fire inline on the main thread between frames and call the
  same `PipelineRuntime` add/reset/toggle methods the hotkeys use, so the main
  loop stays the single writer of scene/interaction (no queues, no duplicated
  logic). Closing the panel window keeps the app running (hotkeys still work);
  the pump becomes a no-op. If Tk can't init (no display), the app prints a
  notice and runs hotkey-only. A `SourceError` from a panel add-layer shows in
  the panel's status label + stderr and never crashes/exits.

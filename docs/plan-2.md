# plan-2.md — airframe, Chunk 2 (POC → usable app)

Continues from `plan.md` (Phases 0–4 done: skeleton + image modes working). This chunk turns the demo into an app you can select as a camera in Zoom/Teams/Meet, overlaying live app windows you manipulate by hand.

Still Python + OpenCV + MediaPipe. New deps: `pyvirtualcam`, `windows-capture`. Native (C#/C++) rewrite is explicitly **not** in this chunk.

Build order is strict. Each phase must run before the next. Copy the **Prompt for Claude Code** block per phase.

---

## Phase 5 — Refactor to a pipeline

**Goal:** restructure the loose MVP into `sources → scene → compositor → sinks` so every later feature plugs in cleanly. No new user-facing features.

**Done when:** the existing image-overlay behavior works exactly as before, but now flows through the new architecture, and a preview window still shows it.

**Prompt for Claude Code:**
> Refactor per this architecture without changing current behavior. Create: `src/sources/` with a base `Source` (returns the latest frame as a numpy array) and `WebcamSource`, `ImageSource`; `src/scene.py` holding a list of `Layer` objects (each = a source + transform: x, y, scale, z_index, selected); `src/compositor.py` that draws the webcam as background then alpha-composites each layer by z-order; `src/sinks/` with a base `Sink` and `PreviewSink` (cv2.imshow). Move grab/move/resize from `overlay.py` into `src/interaction.py` operating on the selected layer. Rewrite `main.py` to wire webcam source → scene → compositor → preview sink, run the loop, keep FPS printing and `q` to quit. The image mode must behave identically to before. Update the structure map in CLAUDE.md and spec.md.

---

## Phase 6 — Virtual camera output (highest value)

**Goal:** broadcast the composited frame as a virtual webcam so it appears in video-call apps.

**Done when:** with OBS Virtual Camera installed, I launch the app and can select the virtual cam in Zoom/Teams/Meet and see my composited feed live.

**Prompt for Claude Code:**
> Add `VirtualCamSink` in `src/sinks/` using `pyvirtualcam`. On Windows it needs the OBS Virtual Camera backend installed — detect its absence and print a clear setup message. Initialize the cam at a fixed resolution and fps matching the pipeline; convert composited frames to the format pyvirtualcam expects (RGB) and send each frame. Add a `--virtualcam` flag (and a runtime key) to enable it alongside the preview sink. Keep both sinks fed from the same composited frame. Confirm the virtual camera shows up and updates live in a video-call app.

---

## Phase 7 — Window & screen capture sources

**Goal:** overlay a *live app window* (or screen region) instead of a static image, as a manipulable layer.

**Done when:** I can point the app at a running window by title, and it appears as a live, grab-able overlay that updates in real time.

**Prompt for Claude Code:**
> Add live capture sources using `windows-capture` (Windows Graphics Capture API). Create `WindowSource` (capture a specific window by title substring) and `MonitorSource` (capture a monitor or region), each running capture on its own callback/thread and exposing the latest frame as a numpy array like the other sources. Handle the target window closing or not being found gracefully. Add a `--source` option (`image` | `window` | `screen`) plus the needed args (window title / monitor index). A window/screen layer must be grab/move/resize-able through the existing interaction path. Confirm a live window overlay tracks the hand.

---

## Phase 8 — Scene: multi-layer + selection

**Goal:** more than one overlay at once, with z-order and "grab the one under the pinch."

**Done when:** I can add multiple layers, pinch to grab whichever is under the pinch point, move/resize it, bring it to front, and remove it — all at runtime.

**Prompt for Claude Code:**
> Extend `scene.py` and `interaction.py` for multiple layers. On pinch start, select the top-most layer whose bounds contain the pinch point and mark it active; move/resize applies only to the active layer. Add runtime keys to add a layer (image/window/screen), remove the active layer, and bring the active layer to front (raise z_index). Draw a thin highlight around the active layer. Composite strictly by z-order. Confirm two overlapping layers can be independently manipulated.

---

## Phase 9 — Performance pass

**Goal:** hold a usable frame rate at virtual-cam resolution with several live sources.

**Done when:** FPS stays smooth (target ~30) with a webcam + one live window layer + virtual cam on, on my laptop.

**Prompt for Claude Code:**
> Decouple the stages so a slow source can't stall render: run each capture source on its own thread publishing latest-frame-wins; run hand tracking at its own cadence (reuse last landmarks between updates if needed); composite + sink on the main loop at target fps. Add simple timing instrumentation per stage to the HUD. Then add an optional GPU path for the hand model (ONNX Runtime or TensorRT) with automatic CPU fallback, behind a flag. Confirm FPS holds with a live window layer and virtual cam enabled.

---

## Phase 10 — Minimal control UI + config

**Goal:** stop driving everything by CLI flags and hotkeys; a small panel to run it day to day.

**Done when:** I can pick a source, add it as a layer, toggle the virtual cam, and reset the scene from a small always-available control window.

**Prompt for Claude Code:**
> Add a lightweight control window (PySide6, or Tkinter if lighter) running alongside the preview: a dropdown/list to choose and add a source (image file picker, window-by-title, monitor), a toggle for the virtual camera, a reset-scene button, and a live FPS readout. It should drive the same scene/sink objects the hotkeys already use — no logic duplication. Persist last settings to a small JSON config on exit and reload on start. Confirm the app is fully operable from the panel.

---

## After this chunk (Chunk 3, not now)
- Packaging: PyInstaller build + installer, bundle/first-run check for OBS Virtual Camera.
- Polish: onboarding, gesture tuning UI, error surfaces.
- Native-rewrite decision: only if Python can't hold latency/quality at target resolution — revisit the C#/C++ + DirectX + Media Foundation path from the original crude plan.

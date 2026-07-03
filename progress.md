# progress.md — shacam-win build log

Chunk 1 (`plan.md`, Phases 0-4): built 2026-07-02. Chunk 2 (`plan-2.md`, Phases 5-10): built 2026-07-03 — see the second half of this file.
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

---

# Chunk 2 — plan-2.md (Phases 5-10), built 2026-07-03

POC → usable app: the demo now broadcasts as a virtual webcam for Zoom/Teams/Meet,
overlays live app windows/monitors you manipulate by hand, supports multiple layers,
holds ~30 FPS via a threaded pipeline, and is drivable from a small Tkinter panel.

## How to run (chunk 2)

```bash
# same 3.11 venv; two new deps: pyvirtualcam (needs OBS Studio 26+), windows-capture
.venv\Scripts\python.exe -m src.main --mode skeleton
.venv\Scripts\python.exe -m src.main --mode image --image assets/test.png
.venv\Scripts\python.exe -m src.main --mode image --source window --window-title "Notepad"
.venv\Scripts\python.exe -m src.main --mode image --source screen --monitor-index 1
# flags: --virtualcam (broadcast to OBS Virtual Camera), --gpu (best-effort, falls
# back to CPU on Windows), --no-panel (hotkeys only), --debug-skeleton
```

Keys: `q` quit · `s` skeleton overlay · `r` reset transform · `+/-` scale · `v` virtual cam
· `a`/`w`/`m` add image/window/screen layer · `x` remove selected · `f` bring to front.
A Tkinter control panel (on by default) mirrors all of it: source picker + file browser,
add layer, virtual-cam checkbox, reset scene, live FPS. Last settings persist to
`handy_config.json` (gitignored; precedence CLI > config > defaults).

## Phase-by-phase

| Phase | Commit | Shipped | Tests |
|-------|--------|---------|-------|
| 5 — Pipeline refactor | `fa57776` | sources/ (Source ABC, Webcam/Image) → scene.py (Transform/Layer/Scene) → compositor.py → sinks/ (Preview); interaction.py state machines; overlay.py dissolved; behavior identical | 128 |
| 6 — Virtual camera | `46adf00` | VirtualCamSink (pyvirtualcam/OBS, lazy init, deferred import), --virtualcam + 'v' key, fatal-at-launch vs degrade-at-runtime failure policy | 153 |
| 7 — Window/screen capture | `3c8b15f` | WindowSource + MonitorSource (windows-capture, own callback thread, latest-frame-wins, graceful window-death), --source/--window-title/--monitor-index, one-time fit to half frame | 197 |
| 8 — Multi-layer scene | `82ee3e7` | pinch selects top-most layer under the point (rising-edge only, no mid-gesture stealing), stable layer_ids, add/remove/raise keys, selection highlight, per-layer pending fit | 245 |
| 9 — Performance pass | `1625e3e` | capture thread + tracking thread (owns MediaPipe timestamps + gesture hysteresis) + 30fps-paced main loop; per-stage EMA timings on HUD; --gpu best-effort w/ CPU fallback; 10.5 → ~29.5 FPS | 295 |
| 10 — Control panel + config | `0345b6d` | Tkinter panel (main-thread pump, zero logic duplication — drives the same runtime methods as hotkeys), handy_config.json persistence, --no-panel | 332 |
| cleanup | `822da68` | dead-code sweep (vulture); main.py 1006 → 699 lines by extracting layers.py (scene/source transitions + vocabulary constants) and launch.py (config/panel bridge) | 332 |

Every phase: implemented → full pytest suite green → opus code review (all CRITICAL/HIGH
findings fixed before commit, MEDIUMs fixed or explicitly deferred to the phase that owns
them) → committed.

## Verification performed

- **Unit tests:** 332 passing (90 from chunk 1), all pure-logic/fakes — no webcam, display,
  MediaPipe, OBS, or capture API needed. New coverage: compositor z-order/clipping/scaling,
  scene selection/identity invariants, interaction multi-layer semantics (selection stealing,
  resize-end edges, background-pinch drag), threaded-source and tracking-worker lifecycle +
  crash policy, copy-every-tick frame isolation, timing/pacing math, virtualcam sink + pump
  state machine, capture source fakes, config roundtrip/corrupt/merge, panel controller.
- **Live smokes (bounded, headless-captured, this machine):** image mode ~29.5 FPS steady
  (vs 10.5 serial baseline; raw camera ~20-26 FPS depending on lighting); live Notepad
  window layer + virtual cam ON held ~29.4 FPS (Phase 9 done-when); window closed mid-run
  → single notice, app keeps running; missing window → clean exit 1 with actionable
  message; OBS Virtual Camera backend detected and fed (Phase 6); --gpu prints fallback
  notice and runs CPU; panel opens, pre-fills from config, syncs the 'v' key, writes
  handy_config.json on clean quit.
- **Not yet human-verified:** selecting the "OBS Virtual Camera" in an actual Zoom/Teams/
  Meet call (backend init + frame streaming is verified; the picker UX needs your eyes);
  gesture feel with multiple overlapping layers; panel ergonomics.

## Key decisions (chunk 2)

- Threading model: three single-owner threads — capture (ThreadedSource wraps the blocking
  cv2 read), tracking (owns HandLandmarker VIDEO-mode timestamps AND gesture hysteresis),
  main loop (sole writer of scene/interaction, paced to TARGET_FPS=30 by time.sleep;
  cv2.waitKey(1) only pumps the window). All hand-offs are single atomic reference
  publishes of immutable/owned snapshots — no locks.
- Failure policy is consistently "degrade, never die": tracker crash publishes an idle
  result (releases any grab) and freezes; camera death freezes the last frame with one
  stderr notice; runtime virtual-cam/window failures warn + disable; only launch-time
  failures exit(1). Join timeouts never free native resources under a live thread.
- pyvirtualcam and windows-capture are deferred imports — the base pipeline runs without
  the optional deps installed.
- GPU: MediaPipe GPU delegate is best-effort only (unsupported in Python on Windows);
  a full ONNX Runtime / TensorRT port is explicitly deferred to Chunk 3.
- Tkinter over PySide6 (stdlib, zero deps); panel pumped on the main thread each tick, so
  its callbacks interleave safely with the frame loop and reuse the exact hotkey code paths.
- Layer identity via monotonically increasing layer_id (never reused); selection changes
  only on a pinch rising edge while idle.

## Known limitations / follow-ups (chunk 3 candidates)

- Virtual-cam feed includes the HUD/debug overlays (intended for the harness; hide-for-call
  is future polish).
- Panel modal dialogs (file Browse) pause the render loop while open (documented; capture
  threads keep running).
- handy_config.json is CWD-relative and written non-atomically (self-heals on corrupt read).
- A CLI value equal to its built-in default can't override a remembered config value
  (argparse limitation, documented in launch.py).
- Hand slots can still swap between frames in two-hand tracking (chunk-1 note, unchanged).
- Chunk 3 (per plan-2.md): packaging/installer, onboarding + gesture tuning UI, and the
  native-rewrite decision only if Python can't hold latency at target resolution.

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

---

# Chunk 3 — plan-3.md (Phases 11-16), built 2026-07-03

The "speak-and-gesture AR presenter" layer: draw shapes in the air, and speak to pull
real images (or generate imagined ones) into the call. Everything produced becomes a
`Layer` in the existing scene and composites through to the virtual cam, reusing the
Chunk-2 hand tracking + threaded pipeline. New optional deps land per phase; the base app
still runs with none of them installed (all imported lazily).

## How to run (chunk 3)

```bash
# same 3.11 venv. New OPTIONAL deps (install only what you use):
#   speech:    pip install sounddevice faster-whisper useful-moonshine-onnx
#   parser:    Ollama running locally + `ollama pull qwen2.5:1.5b` (no pip dep; stdlib urllib)
#   retriever: pip install ddgs           (Wikimedia needs no dep; ddgs is the fallback)
#   generator: pip install torch --index-url https://download.pytorch.org/whl/cu124
#              pip install diffusers transformers accelerate safetensors   (CUDA GPU only)
.venv\Scripts\python.exe -m src.main --mode image --image assets/test.png            # air-drawing (no deps)
.venv\Scripts\python.exe -m src.main --mode image --image assets/test.png --speech   # + live transcript
.venv\Scripts\python.exe -m src.main --mode image --image assets/test.png --speech --parse            # + LLM router
.venv\Scripts\python.exe -m src.main --mode image --image assets/test.png --speech --parse --generate # + GPU generation
```

New runtime keys (image mode): air-drawing — hold the draw pose (index extended, other
fingers curled) to trace a stroke; release to freeze it into a grabbable layer. `c` clear
in-progress stroke, `k` cycle stroke color, `[`/`]` stroke width. Speak "insert CFTR" (with
`--speech --parse`) to dock a fetched image at the top edge; pinch it and pull down to spawn
it into the scene. `--speech`/`--parse`/`--generate` persist to `handy_config.json` like the
other flags (CLI > config > default).

## Phase-by-phase

| Phase | Commit | Shipped | Tests |
|-------|--------|---------|-------|
| 11 — Air-drawing | `6d6e98d` | `drawing.py` (immutable Stroke + moving-average smoothing w/ pinned endpoints + rasterize-to-BGRA + bbox→Transform); draw-pose detection w/ N-frame hysteresis on the tracking thread; freeze-to-`ArraySource`-layer on release; `c`/`k`/`[`/`]` keys; pinch-suppressed-while-drawing precedence | 427 |
| 12 — Speech layer | `bdbff31` | `speech.py` (`SpeechWorker` own thread owns mic + ASR; Moonshine active w/ faster-whisper fallback + engine warm-up validation; energy-VAD phrase segmentation; atomic rolling-transcript publish + per-phrase callback); `--speech`; heavy deps lazy/optional | 473 |
| 13 — LLM parser + router | `ddafbdc` | `parser.py` (Ollama qwen2.5:1.5b → strict JSON `{action,query,route,source}`; `parse_llm_json` is the untrusted-output boundary — bounded input, JSON-bomb `RecursionError` caught, query clamped, enum-validated w/ safe defaults); `--parse`; runs on the speech thread | 524 |
| 14 — Retriever | `749aa62` | `retriever.py` (WikimediaSource default + keyless DuckDuckGo `ddgs` fallback + optional local folder, behind one `fetch`; decode→BGRA asset; SSRF-safe: per-redirect-hop re-validation, http/https-only, 20 MB + 10 s caps, pre-decode 40 MP bomb guard, LRU temp cache) | 594 |
| 15 — Spawn gesture | `3654b56` | `spawn.py` (`PreviewInbox` — the ONE atomic speech→main hand-off; pure dock geometry + `advance_spawn` state machine IDLE→GRABBED→spawn/dismiss; `render_dock`); retrieved image docks top-edge, pinch-from-top + pull-down spawns it as a layer through to the virtual cam; suppress-pinch-while-dragging precedence | 630 |
| 16 — Generator (GPU) | `f4d5675` | `generator.py` (`DiffusionGenerator` — SDXL-Turbo default, **in-engine OOM fallback to sd-turbo**, lazy pipeline load, fp16 + cpu-offload for 8 GB; `make_placeholder` "materializing" card; RGB→BGRA); `--generate` opt-in; generate-routed request publishes placeholder→result via the same PreviewInbox spawn path; degrades to a notice with no GPU/torch | 667 |

Every phase: implemented (opus subagent) → full pytest suite green → independent opus code
review (all CRITICAL/HIGH fixed before commit; MEDIUMs fixed unless plan-deferred) →
committed. Orchestrator fixes applied on top of each agent's work before commit (see Key
decisions).

## Verification performed

- **Unit tests:** 667 passing (378 inherited from chunks 1-2, +289 for chunk 3), all
  pure-logic/faked — no webcam, display, MediaPipe, mic, ASR model, Ollama, network, or GPU
  touched. New coverage: stroke smoothing/rasterize/placement + draw-pose hysteresis +
  pinch-suppression precedence; VAD segmentation + engine-fallback + SpeechWorker crash/mic-
  fail lifecycle + bad-callback isolation; `parse_llm_json` on fenced/prose/garbage/bomb
  input + routing normalization + OllamaClient error mapping; Wikimedia response parsing +
  URL-scheme validation + decode/RGBA/downscale + decode-bomb + redirect-block + byte-cap +
  cache prune + source fallback; PreviewInbox hand-off + dock geometry + spawn state machine
  + mid-grab replacement + suppress-during-grab + runtime spawn lifecycle; generator
  placeholder + RGB→BGRA + no-GPU/no-torch degrade + never-raises + **OOM fallback** +
  placeholder-then-result publish order + `--generate` gating.
- **Live/real-backend checks (this machine, throwaway scripts):**
  - Phase 12: Moonshine engine initialized natively (faster-whisper `base.en` verified as
    fallback + cached); 13 mic input devices probed (no recording).
  - Phase 13: real Ollama round-trip hit the done-when first try — "yeah we're looking at
    CFTR" → `{action: insert, query: "CFTR protein structure", route: retrieve, source:
    wikimedia}`; chit-chat → `None`; "friendly robot mascot" → `route: generate`. Warm
    latency ~2.8 s/call.
  - Phase 14: real Wikimedia fetch for "CFTR protein structure" → 1024×715 BGRA, CC BY-SA
    4.0 attribution, 0.62 s; nonsense query → graceful ddgs fallback.
- **Phase 16 real GPU generation — NOT verified in this environment (disk-blocked).** torch
  2.6.0+cu124 (CUDA available) and diffusers 0.39.0 install fine, but the ~7 GB diffusion
  model can't be held: the disk has only ~8.6 GB free (925 GB drive, 100% full), and a
  HuggingFace download needs ~2× the model size transiently to finalize, so both SDXL-Turbo
  and sd-turbo downloaded to full size then failed to finalize and were cleaned up (GPU VRAM
  never rose above idle → no model ever loaded). Separately, SDXL-Turbo was observed to OOM
  the 8 GB card even with cpu-offload — which is exactly why the in-engine OOM→sd-turbo
  fallback was added. The generator CODE is complete, reviewed (PASS, no CRITICAL/HIGH), and
  fully unit-tested incl. the fallback; the runtime degrades cleanly (a load/OOM failure is
  caught → one notice → generate returns `None`, app keeps running). **To finish verifying:
  free ~15 GB of disk, then `--speech --parse --generate` and speak a "generate" request.**

## Key decisions (chunk 3)

- **One atomic channel per cross-thread hand-off, preserved throughout.** Speech runs on a
  4th single-owner thread; the parser/retriever/generator all run ON that speech thread
  (blocking network/GPU there is fine — never the render loop). The only speech→main channel
  is `PreviewInbox` (one reference store/clear, latest-wins, no lock), so the main loop stays
  the sole writer of scene/interaction. Every code review re-confirmed this invariant holds.
- **Untrusted-input boundaries hardened at review time.** The LLM output (`parse_llm_json`)
  and the remote-image fetch (`retriever`) are the two trust boundaries; reviews caught and
  fixed a JSON-bomb `RecursionError` escaping the parser's "never raises" contract, and — on
  the retriever — an SSRF-via-redirect scheme bypass and a decode-time pixel bomb, plus
  bounded the query, download size, timeout, and temp cache.
- **Degrade-never-die extended to every new dep.** Missing sounddevice/ASR, Ollama down,
  Wikimedia/ddgs miss or error, no CUDA/torch, model OOM or load failure — each returns
  `None` + one stderr notice and the app keeps running. All heavy imports are deferred so the
  base pipeline imports with none installed (proven: the suite passes without torch).
- **Model choice for the 8 GB target.** SDXL-Turbo (quality) is the default with an in-engine
  OOM fallback to the lighter sd-turbo (reliability), rather than shipping a default that
  silently fails on the stated hardware. qwen2.5:1.5b (over the installed qwen3:4b) for the
  parser — smaller, faster, JSON-only, no thinking-mode overhead.
- **main.py held at the 800-line cap** every phase by putting logic in the new modules and
  trimming docstrings; spawn/generate glue lives in `spawn.py`/`generator.py`/`launch.py`.

## Known limitations / follow-ups (chunk 3)

- **Phase 16 real generation unverified (disk space)** — see Verification above. Also, the
  first generate is slow on 8 GB (SDXL-Turbo load → OOM → sd-turbo reload) before it settles;
  making sd-turbo the default on ≤8 GB cards would avoid the one-time OOM tax.
- Speech transcription pauses while a generate request loads the model on the first call
  (both on the speech thread); acceptable, bounded by the worker's daemon + join timeout.
- The docked preview can be pinch-spawned while it's still a "materializing" placeholder
  (spawns the placeholder card); the real image re-docks on arrival — acceptable MVP edge.
- Stroke fragmentation on a tracking dropout and single-slot (primary-hand) drawing/spawn are
  deliberate MVP tradeoffs (documented), owned by plan-3.md's closing tuning pass.
- WebP images skip the pre-decode dimension probe (PNG/JPEG/GIF/BMP covered); a WebP bomb is
  still bounded by the 20 MB compressed cap + post-decode backstop.
- Deferred per plan-3.md: 3D objects (image-to-mesh), CLIP-embedding local library matching,
  and the gesture-threshold/trigger-phrase tuning pass.

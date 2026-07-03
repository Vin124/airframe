# plan-3.md — shacam-win, Chunk 3 (speak-and-gesture AR presenter)

Continues from `plan-2.md` (Chunk 2 done: pipeline refactor, virtual cam, window/screen capture, multi-layer scene). This chunk adds the "conjure things into frame" layer — draw shapes in the air, and speak to pull real images into the call.

Still Python + OpenCV + MediaPipe. New deps by phase: `moonshine` (or faster-whisper as fallback), a local LLM runtime (`ollama`), an image-search/Wikimedia client, and — only for the optional generation phase — a diffusion runtime + GPU.

Everything here produces an asset that becomes a **Layer** in the existing scene, and reuses the existing hand tracking + virtual-cam sink. Build order is strict; each phase must run before the next.

---

## Phase 11 — Air-drawing (no AI, quick win)

**Goal:** draw with your fingertip and leave a shape floating in frame.

**Done when:** I hold a "draw" gesture, move my index finger, and a stroke follows it; the stroke stays as an overlay I can then grab/move like any other layer.

**Prompt for Claude Code:**
> Add air-drawing. Reuse the hand tracker. Define a draw gesture (e.g. index extended while middle/ring/pinky curled) that, while held, samples the index fingertip (landmark 8) each frame and appends to a stroke path. Render the stroke as a smoothed polyline overlay. On gesture release, freeze the stroke into a new Layer in the scene (rasterize it to an RGBA image at its bounding box) so it can be grabbed/moved/resized by the existing interaction code. Add a runtime key to clear the current stroke and to set stroke color/width. Confirm I can draw an "A" in the air and it stays in frame.

---

## Phase 12 — Speech layer (always listening)

**Goal:** live transcription running alongside the video loop, off the mic.

**Done when:** what I say prints to the console as a rolling live transcript without stalling the video/tracking frame rate.

**Prompt for Claude Code:**
> Add `src/speech.py`. Capture mic audio in a background thread and stream it to Moonshine for live transcription (fall back to faster-whisper if Moonshine setup is problematic on this machine). Expose a rolling transcript buffer and a callback that fires on each finalized phrase. It must run fully off the main video loop — the render/tracking FPS must not drop when speech is active. Print the live transcript to console for now. No actions wired yet.

---

## Phase 13 — Local LLM parser + router

**Goal:** turn messy speech into a clean, structured request and decide what to do with it.

**Done when:** saying "yeah we're looking at CFTR" produces a structured result like `{action: insert, query: "CFTR protein structure", route: retrieve, source: wikimedia}` in the console.

**Prompt for Claude Code:**
> Add `src/parser.py`. Run a small local LLM via Ollama (a 1–3B model, e.g. Qwen or Phi) that takes a recent transcript snippet and returns strict JSON: whether the user is asking to insert/create something, the cleaned search query, whether it should be **retrieved** (a real, nameable thing) or **generated** (imagined/described), and a suggested source. Prompt it to output JSON only; parse safely and ignore non-actionable speech. Feed it the finalized phrases from Phase 12. Keep latency low (short outputs). Log the parsed result. No fetching yet.

---

## Phase 14 — Retriever

**Goal:** fetch a real image for the parsed query.

**Done when:** a parsed `retrieve` request for "CFTR" returns an actual image, downloaded and ready to show.

**Prompt for Claude Code:**
> Add `src/retriever.py` with a swappable source behind one interface: `WikimediaSource` (default — query Wikimedia Commons via its API, pick the best licensed image), a `WebSearchSource` fallback (an image-search API for misses), and an optional `LocalLibrarySource` (a preloaded folder matched to the query with CLIP embeddings). Given a parsed request, fetch the top image, download it to a temp cache as RGBA, and hand it back as a ready asset. Handle no-result and network errors gracefully. Wire it so a `retrieve` request from Phase 13 ends with an image in hand. Don't spawn it into the scene yet.

---

## Phase 15 — Spawn gesture (pull it into frame)

**Goal:** the moment that ties speech to the scene — pinch from the top edge and pull the fetched asset down into the call.

**Done when:** after I say "insert CFTR", the image appears as a preview at the top of the frame, and a pinch-from-top + pull-down drags it in as a normal, manipulable layer the viewer sees on the virtual cam.

**Prompt for Claude Code:**
> When the retriever returns an asset, show it as a docked preview at the top edge of the frame. Add a spawn gesture: a pinch starting near the top edge over the preview "grabs" it, and pulling downward instantiates it as a new Layer at the pinch point, after which the existing move/resize interaction takes over. If ignored for a few seconds, the preview dismisses. Make sure the spawned layer composites through to the virtual cam. This connects speech → parser → retriever → scene end to end.

---

## Phase 16 — Generator (optional, GPU)

**Goal:** for made-up/described things, generate an image instead of retrieving one.

**Done when:** a parsed `generate` request for "a simple A shape" or "a friendly robot" produces a generated image that spawns the same way via Phase 15.

**Prompt for Claude Code:**
> Add `src/generator.py` for `generate`-routed requests: run a fast few-step text-to-image model (Z-Image-Turbo, or SDXL-Lightning) on GPU, with a clear message if no suitable GPU is present. Show a "materializing" placeholder in the top preview while it runs, then swap in the result and hand it to the same spawn path as the retriever. Keep it fully behind the router's `generate` branch so retrieval stays the default for real things. Confirm a described object can be summoned into frame.

---

## After this chunk (later)
- **3D objects:** Stable Fast 3D (image-to-mesh, ~sub-second on a consumer GPU) so summoned objects can be rotated, not just flat images. Heavy rendering work — deferred until the 2D flow feels good.
- Tuning pass: gesture thresholds for draw vs pinch vs spawn, transcript trigger phrasing, source preferences.

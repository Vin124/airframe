# plan.md — airframe MVP

Build order is strict: each phase must run before starting the next. Copy the **Prompt for Claude Code** block into Claude Code for each phase.

---

## Phase 0 — Scaffold + environment

**Goal:** repo skeleton, dependencies, a webcam window that shows the raw feed.

**Done when:** `python -m src.main --mode skeleton` opens the webcam and shows the live feed (no tracking yet), `q` quits, console prints FPS.

**Prompt for Claude Code:**
> Set up the project per CLAUDE.md. Create `requirements.txt` (opencv-python, mediapipe, numpy), the `src/` package, and `assets/`. Implement `src/camera.py` (open default webcam, read frames, release cleanly) and a minimal `src/main.py` with argparse for `--mode` (choices: skeleton, image) and `--image`. For now, in the main loop just capture frames, mirror them horizontally (selfie view), show them in an OpenCV window, print FPS to console, and quit on `q`. Don't add hand tracking yet. Confirm it runs.

---

## Phase 1 — Mode 1: live hand skeleton

**Goal:** the core debug view — 21-point hand skeleton drawn live over the webcam.

**Done when:** `--mode skeleton` shows my hand skeleton tracking in real time with acceptable FPS.

**Prompt for Claude Code:**
> Implement hand tracking. In `src/hand_tracker.py`, wrap MediaPipe (prefer the Tasks API `HandLandmarker` with `hand_landmarker.task` in VIDEO/LIVE_STREAM mode; fall back to `mp.solutions.hands` if simpler for the installed version). Expose a class that takes a frame and returns a list of hands, each as 21 normalized (x, y, z) landmarks, or empty if none. Auto-download the `.task` model if needed. In `src/skeleton.py`, draw the landmarks as dots and the standard hand connections as lines, converting normalized coords to pixels. Wire `skeleton` mode in `main.py` to run the tracker each frame and draw the skeleton over the feed. Keep FPS printing. Make sure it runs on a laptop webcam in real time.

---

## Phase 2 — Gesture layer (pinch + grab)

**Goal:** turn raw landmarks into gesture state. No visuals change yet except a debug label.

**Done when:** an on-screen text label reads "PINCH" when I touch thumb+index together and "OPEN" otherwise, responsive and not too jittery.

**Prompt for Claude Code:**
> Add `src/gestures.py`. Implement pinch detection: normalized Euclidean distance between landmark 4 (thumb tip) and 8 (index tip); below a tuned threshold = pinching. Return a small state object per hand: `is_pinching` (bool), `pinch_point` (midpoint of 4 and 8, normalized), and optionally `is_grab` (fist — most fingertips curled toward the palm). Add light smoothing/hysteresis so the state doesn't flicker. In skeleton mode, draw a debug label showing the current gesture and a dot at the pinch point so I can eyeball the tuning. Don't touch image mode yet.

---

## Phase 3 — Mode 2: image grab / move / resize

**Goal:** pinch to grab a loaded image, move it with the hand, resize it.

**Done when:** `--mode image --image assets/test.png` shows the image over the feed; pinching on it grabs it, moving my hand moves it, and a resize gesture scales it.

**Prompt for Claude Code:**
> Implement `src/overlay.py`: load an image with alpha, hold its position/scale as state, and alpha-composite it onto each frame at its current transform. Wire up `image` mode in `main.py`: reuse the tracker + gestures from Phases 1–2. Interaction: when a pinch starts with the pinch point over the image, "grab" it; while held, the image follows the pinch point (move); release on un-pinch. For resize, use a two-stage approach — start with a simple keyboard `+`/`-` scale so the pipeline works, then implement a two-hand pinch-distance resize (distance between the two hands' pinch points controls scale) if two hands are available. Keep the skeleton optionally drawable via a debug flag. Make sure both modes still run.

---

## Phase 4 — Cleanup + tuning

**Goal:** make it pleasant to demo and debug.

**Done when:** thresholds feel right, FPS is visible, and switching modes is smooth.

**Prompt for Claude Code:**
> Polish pass. Expose the pinch threshold and smoothing as constants at the top of `gestures.py` with comments on how to tune. Add a runtime key to toggle the skeleton overlay on/off, and a key to reset the image transform in image mode. Draw a small HUD (mode name, FPS, gesture state). Handle no-webcam and no-image errors gracefully with clear messages. Confirm both modes run end to end.

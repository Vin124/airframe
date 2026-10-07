# Contributing to Airframe

Thanks for your interest! Issues and pull requests are welcome.

## Dev setup

```bash
git clone https://github.com/Vin124/airframe.git
cd airframe
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q          # should pass before you start
```

Requires Windows 10/11 and Python 3.11+. The test suite is fully headless —
no webcam, display, microphone, or network needed.

## Project conventions

These are the rules the codebase is built on; PRs should follow them:

- **Every change leaves the app runnable.** No half-wired states.
- **Degrade, never die.** A missing optional dependency, an unreachable
  service, or a crashed worker prints one clear notice and the app keeps
  running. Heavy/optional imports are deferred into the class that needs them.
- **One owner per piece of state.** Threads hand off with a single atomic
  reference publish (latest-wins, no locks). The main loop is the only writer
  of the scene; the tracking thread is the only toucher of the tracker and
  gesture detectors. Treat published frames as read-only.
- **Normalized coordinates.** Landmarks stay in 0–1 space; convert to pixels
  only at draw time.
- **Pure logic, tested.** Gesture math, geometry, smoothing, config
  validation, and parsing are pure functions/classes with unit tests; I/O
  (camera, MediaPipe, Tk, network) stays behind small injectable wrappers.
- **Small, focused modules** over large ones.

Architecture details, threading model, and per-module notes live in
[`CLAUDE.md`](CLAUDE.md); the original spec and phased build plans are in
[`docs/`](docs/).

## Pull requests

1. Fork and create a feature branch.
2. Add or update tests for anything with testable logic (`tests/` mirrors
   `src/` one-to-one).
3. Make sure `python -m pytest -q` passes.
4. Use [conventional-commit](https://www.conventionalcommits.org/) style
   messages: `feat: ...`, `fix: ...`, `refactor: ...`, `docs: ...`,
   `test: ...`, `chore: ...`, `perf: ...`.
5. Describe *what* and *why* in the PR body; include repro steps for bug
   fixes.

For larger features, please open an issue first so the approach can be
discussed.

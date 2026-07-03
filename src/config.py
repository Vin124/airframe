"""Small JSON config: persist the control panel's last-used settings.

Pure load/save/merge helpers so the panel and the runtime can round-trip a
handful of settings -- last source choice, image path, window title, monitor
index, and virtual-cam on/off -- across runs without needing a UI or a display.

Design points:
  - Everything here is a pure function of its inputs (plus the file at ``path``);
    nothing imports ``src.main`` or touches Tkinter, so it's trivially unit-
    testable with ``tmp_path``.
  - A missing file (first run) or a corrupt/unreadable one falls back to the
    built-in defaults with a one-line ``stderr`` notice -- never a crash.
  - ``merge_config`` is the single validation boundary: unknown keys are
    dropped, wrong-typed / out-of-range values fall back to their default, so a
    hand-edited or stale file can never inject a bad value downstream.

Precedence for a launched app is **CLI > config > built-in defaults**: an
explicit CLI flag overrides a persisted value, which overrides the hard-coded
default here. That precedence is resolved in ``src.main`` (see
``resolve_panel_prefill``); this module only owns the "config" tier.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict

# Written to the current working directory (the repo root when launched via
# ``python -m src.main``). Git-ignored -- it's per-user local state.
CONFIG_FILENAME = "handy_config.json"

# Serialization values for the source choice. These intentionally mirror
# ``src.main``'s ``SOURCE_*`` CLI choices (same literals) but are declared here
# so the persistence layer stays independent of the CLI module (no import cycle).
SOURCE_IMAGE = "image"
SOURCE_WINDOW = "window"
SOURCE_SCREEN = "screen"
_VALID_SOURCES = (SOURCE_IMAGE, SOURCE_WINDOW, SOURCE_SCREEN)

# windows-capture indexes monitors 1-based (1 = primary); anything < 1 is invalid.
_DEFAULT_MONITOR_INDEX = 1

_DEFAULTS: Dict[str, Any] = {
    "source": SOURCE_IMAGE,
    "image_path": "",
    "window_title": "",
    "monitor_index": _DEFAULT_MONITOR_INDEX,
    "virtualcam": False,
}


def default_config() -> Dict[str, Any]:
    """Return a fresh copy of the built-in defaults (never the shared dict)."""
    return dict(_DEFAULTS)


def merge_config(loaded: Any) -> Dict[str, Any]:
    """Overlay the known, type-valid keys of ``loaded`` onto the defaults.

    The single validation boundary: a non-dict input, an unknown key, a wrong
    type, or an out-of-range value each falls back to the default for that key.
    Returns a fresh dict and never mutates ``loaded`` -- so a stale or
    hand-edited config can't smuggle a bad value into the app.
    """
    config = default_config()
    if not isinstance(loaded, dict):
        return config

    source = loaded.get("source")
    if source in _VALID_SOURCES:
        config["source"] = source

    image_path = loaded.get("image_path")
    if isinstance(image_path, str):
        config["image_path"] = image_path

    window_title = loaded.get("window_title")
    if isinstance(window_title, str):
        config["window_title"] = window_title

    monitor_index = loaded.get("monitor_index")
    # bool is an int subclass -- reject True/False sneaking in as an index.
    if isinstance(monitor_index, int) and not isinstance(monitor_index, bool) and monitor_index >= 1:
        config["monitor_index"] = monitor_index

    virtualcam = loaded.get("virtualcam")
    if isinstance(virtualcam, bool):
        config["virtualcam"] = virtualcam

    return config


def load_config(path: Any) -> Dict[str, Any]:
    """Load and validate the config at ``path``; defaults on missing/corrupt.

    A missing file is silent (the first-run case). A present-but-unreadable file
    or invalid JSON prints a one-line ``stderr`` notice and returns the defaults
    -- this never raises, so a bad config can't stop the app from starting.
    """
    p = Path(path)
    if not p.exists():
        return default_config()
    try:
        loaded = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"Warning: could not read config {p} ({exc}); using defaults.", file=sys.stderr)
        return default_config()
    return merge_config(loaded)


def save_config(path: Any, config: Dict[str, Any]) -> None:
    """Write ``config`` (validated onto the defaults first) to ``path`` as JSON.

    Merges through ``merge_config`` so an incomplete or slightly-off dict still
    round-trips cleanly. A write failure prints a ``stderr`` notice instead of
    raising, so persisting on exit can't crash teardown.
    """
    merged = merge_config(config)
    try:
        Path(path).write_text(json.dumps(merged, indent=2), encoding="utf-8")
    except OSError as exc:
        print(f"Warning: could not write config {path} ({exc}).", file=sys.stderr)

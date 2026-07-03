"""Small JSON config: persist the control panel's last-used settings.

Pure load/save/merge helpers so the panel and the runtime can round-trip a
handful of settings -- last source choice, image path, window title, monitor
index, virtual-cam on/off, the live pinch thresholds (``pinch_enter`` /
``pinch_exit``, panel sliders), and the startup-only MediaPipe min-confidence
knobs (``min_detection_confidence`` / ``min_presence_confidence`` /
``min_tracking_confidence``) -- across runs without needing a UI or a display.

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

from src.tuning import (
    CONFIDENCE_MAX,
    CONFIDENCE_MIN,
    DEFAULT_MEDIAPIPE_CONFIDENCE,
    DEFAULT_PINCH_ENTER,
    DEFAULT_PINCH_EXIT,
    PINCH_THRESHOLD_MAX,
    PINCH_THRESHOLD_MIN,
    clamp_tuning,
)

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

# MediaPipe's three startup-only min-confidence knobs, persisted so a CLI-set
# value round-trips. See ``src.tuning.TrackerConfidence`` / ``resolve_panel_prefill``.
_CONFIDENCE_KEYS = (
    "min_detection_confidence",
    "min_presence_confidence",
    "min_tracking_confidence",
)

_DEFAULTS: Dict[str, Any] = {
    "source": SOURCE_IMAGE,
    "image_path": "",
    "window_title": "",
    "monitor_index": _DEFAULT_MONITOR_INDEX,
    "virtualcam": False,
    # Speech layer opt-in (Phase 12, --speech). Like virtualcam: a store_true CLI
    # flag whose last state is remembered here.
    "speech": False,
    # LLM parser opt-in (Phase 13, --parse; only meaningful with --speech). Same
    # store_true-flag-remembered-here shape as speech/virtualcam.
    "parse": False,
    # Image-generator opt-in (Phase 16, --generate; only meaningful with --speech
    # --parse, GPU-only). Same store_true-flag-remembered-here shape as parse.
    "generate": False,
    # Live pinch tuning (panel sliders); clamped as a pair so exit > enter.
    "pinch_enter": DEFAULT_PINCH_ENTER,
    "pinch_exit": DEFAULT_PINCH_EXIT,
    # Startup-only MediaPipe confidences (CLI/config, not sliders).
    "min_detection_confidence": DEFAULT_MEDIAPIPE_CONFIDENCE,
    "min_presence_confidence": DEFAULT_MEDIAPIPE_CONFIDENCE,
    "min_tracking_confidence": DEFAULT_MEDIAPIPE_CONFIDENCE,
}


def default_config() -> Dict[str, Any]:
    """Return a fresh copy of the built-in defaults (never the shared dict)."""
    return dict(_DEFAULTS)


def _number_in_range(value: Any, low: float, high: float) -> bool:
    """True iff ``value`` is a real (non-bool) number within ``[low, high]``."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and low <= value <= high
    )


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

    speech = loaded.get("speech")
    if isinstance(speech, bool):
        config["speech"] = speech

    parse = loaded.get("parse")
    if isinstance(parse, bool):
        config["parse"] = parse

    generate = loaded.get("generate")
    if isinstance(generate, bool):
        config["generate"] = generate

    # Pinch tuning: accept each value only if it's a number inside the slider
    # range (else fall back to that key's default), then run the PAIR through the
    # shared clamp so the hysteresis gap (exit > enter) is always enforced --
    # exactly the same validation boundary the panel slider path uses.
    pinch_enter = loaded.get("pinch_enter")
    pinch_exit = loaded.get("pinch_exit")
    enter = (
        pinch_enter
        if _number_in_range(pinch_enter, PINCH_THRESHOLD_MIN, PINCH_THRESHOLD_MAX)
        else DEFAULT_PINCH_ENTER
    )
    exit = (
        pinch_exit
        if _number_in_range(pinch_exit, PINCH_THRESHOLD_MIN, PINCH_THRESHOLD_MAX)
        else DEFAULT_PINCH_EXIT
    )
    tuning = clamp_tuning(enter, exit)
    config["pinch_enter"] = tuning.enter
    config["pinch_exit"] = tuning.exit

    # Startup-only MediaPipe confidences: each a float in [0, 1] or its default.
    for key in _CONFIDENCE_KEYS:
        value = loaded.get(key)
        if _number_in_range(value, CONFIDENCE_MIN, CONFIDENCE_MAX):
            config[key] = float(value)

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

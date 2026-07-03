"""Runtime tuning knobs shared across layers, dependency-light on purpose.

Two immutable snapshots plus the single clamp that guards the pinch hysteresis
invariant:

- ``GestureTuning(enter, exit)`` -- the live pinch thresholds. Published from the
  MAIN thread to the TRACKING thread via one atomic reference store (see
  ``src.tracking.TrackingWorker.set_tuning``); the panel sliders drive it.
- ``TrackerConfidence(detection, presence, tracking)`` -- MediaPipe's three
  min-confidence knobs. Startup-only (they need a tracker re-init), so they are
  CLI/config resolved, never live-tuned.

Kept free of any heavy import (stdlib ``dataclasses`` only) so BOTH the gesture
layer (``src.gestures`` -> ``src.hand_tracker`` -> MediaPipe) and the import-light
persistence layer (``src.config``) can share ``clamp_tuning`` without dragging
MediaPipe into config. ``clamp_tuning`` is the ONE validation boundary for pinch
tuning, used by the panel path and by config load alike.
"""

from dataclasses import dataclass

# Slider / config range for both pinch thresholds (normalized index-thumb
# distance, a fraction of the frame -- not pixels). The panel Scale widgets span
# this window and ``src.config.merge_config`` rejects anything outside it.
PINCH_THRESHOLD_MIN = 0.01
PINCH_THRESHOLD_MAX = 0.15

# Built-in pinch defaults, tuned by eye at arm's length (see the HOW TO TUNE
# block in ``src.gestures``). ``src.gestures`` re-exports these as the
# ``PINCH_ENTER_THRESHOLD`` / ``PINCH_EXIT_THRESHOLD`` names it has always used.
DEFAULT_PINCH_ENTER = 0.05
DEFAULT_PINCH_EXIT = 0.07

# Minimum enter/exit separation. Hysteresis only works when exit sits strictly
# ABOVE enter: entering a pinch needs the tight ENTER distance, but once pinching
# the fingers may drift out to the looser EXIT distance before releasing. If the
# two collapse (equal or inverted) the pinch label flickers on/off as the fingers
# hover at a single boundary. ``clamp_tuning`` guarantees at least this gap.
MIN_HYSTERESIS_GAP = 0.005

# MediaPipe's own default for all three min-confidence knobs, and their valid
# range. Startup-only (a change needs a tracker re-init) -- CLI/config, no slider.
DEFAULT_MEDIAPIPE_CONFIDENCE = 0.5
CONFIDENCE_MIN = 0.0
CONFIDENCE_MAX = 1.0


@dataclass(frozen=True)
class GestureTuning:
    """Immutable pinch-threshold snapshot (main thread -> tracking thread)."""

    enter: float
    exit: float


@dataclass(frozen=True)
class TrackerConfidence:
    """Immutable MediaPipe min-confidence trio (startup-only; needs re-init)."""

    detection: float = DEFAULT_MEDIAPIPE_CONFIDENCE
    presence: float = DEFAULT_MEDIAPIPE_CONFIDENCE
    tracking: float = DEFAULT_MEDIAPIPE_CONFIDENCE


def _clamp(value: float, low: float, high: float) -> float:
    """Clamp ``value`` into the inclusive ``[low, high]`` range."""
    return max(low, min(high, value))


def clamp_tuning(enter: float, exit: float) -> GestureTuning:
    """Clamp a pinch (enter, exit) pair into range and enforce the hysteresis gap.

    The single validation boundary for pinch tuning, shared by the panel slider
    path and config load. Both values are clamped into
    ``[PINCH_THRESHOLD_MIN, PINCH_THRESHOLD_MAX]``, then ``exit`` is lifted to at
    least ``enter + MIN_HYSTERESIS_GAP`` so the invariant ``exit > enter`` always
    holds -- an equal or inverted pair from a hand-edited config or a slider drag
    can never disable hysteresis and reintroduce pinch-label flicker. If lifting
    ``exit`` would push it past the max, the pair is pulled down together so both
    stay in range while preserving the gap.
    """
    enter = _clamp(float(enter), PINCH_THRESHOLD_MIN, PINCH_THRESHOLD_MAX)
    exit = _clamp(float(exit), PINCH_THRESHOLD_MIN, PINCH_THRESHOLD_MAX)
    if exit < enter + MIN_HYSTERESIS_GAP:
        exit = enter + MIN_HYSTERESIS_GAP
    if exit > PINCH_THRESHOLD_MAX:
        exit = PINCH_THRESHOLD_MAX
        enter = PINCH_THRESHOLD_MAX - MIN_HYSTERESIS_GAP
    return GestureTuning(enter=enter, exit=exit)

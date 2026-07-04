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

# Slider / config range for both pinch thresholds. Since the palm-normalized
# metric landed, a threshold is a RATIO: index-thumb distance divided by the
# palm length (wrist -> middle-finger MCP), so it's invariant to how far the
# hand is from the camera. Touching fingertips read ~0.05-0.20; an open hand
# reads well above 1.0. The panel Scale widgets span this window and
# ``src.config.merge_config`` rejects anything outside it (which also cleanly
# retires stale configs persisted under the old frame-fraction metric, whose
# 0.01-0.15 values mostly fall below this minimum).
PINCH_THRESHOLD_MIN = 0.10
PINCH_THRESHOLD_MAX = 0.90

# Built-in pinch defaults (palm-relative ratio: thumb-index gap / palm length).
# A firm pinch reads ~0.05-0.20 and an open hand well above 1.0, so ENTER at
# 0.40 fires on a natural, LOOSE pinch (fingertips a few cm apart) rather than
# demanding a hard press -- the earlier 0.30 needed near-touching fingers, which
# is the main reason grabs felt hard to trigger. EXIT at 0.58 gives a wide 0.18
# hysteresis band so a grab HOLDS while the fingers drift during a drag. Both are
# live-tunable from the panel sliders. ``src.gestures`` re-exports these as
# ``PINCH_ENTER_THRESHOLD`` / ``PINCH_EXIT_THRESHOLD``.
DEFAULT_PINCH_ENTER = 0.40
DEFAULT_PINCH_EXIT = 0.58

# Minimum enter/exit separation. Hysteresis only works when exit sits strictly
# ABOVE enter: entering a pinch needs the tight ENTER distance, but once pinching
# the fingers may drift out to the looser EXIT distance before releasing. If the
# two collapse (equal or inverted) the pinch label flickers on/off as the fingers
# hover at a single boundary. ``clamp_tuning`` guarantees at least this gap.
# Scaled with the ratio metric's range (tremor near the boundary is ~0.02 of a
# palm length, vs ~0.003 of a frame under the old metric).
MIN_HYSTERESIS_GAP = 0.03

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

"""Pinch/grab gesture detection from normalized hand landmarks.

Pure math on normalized ``(x, y, z)`` landmark tuples (see ``src.hand_tracker``
for the ``Hand`` type). Pixel conversion happens only at draw time, per
project convention.
"""

from dataclasses import dataclass
from math import hypot
from typing import Optional, Tuple

from src.hand_tracker import Hand
from src.tuning import DEFAULT_PINCH_ENTER, DEFAULT_PINCH_EXIT

# Landmark indices (MediaPipe hand topology).
WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20

# Curl heuristic: for grab detection, compare each fingertip's distance from
# the wrist against its PIP joint's distance from the wrist. Index/middle/
# ring/pinky, in the same order as their tips above.
_CURL_PIP_JOINTS = (6, 10, 14, 18)
_CURL_FINGER_TIPS = (INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)

# A fist rarely curls all four fingers with perfect symmetry (tracking noise,
# thumb occlusion, etc.), so require a majority rather than unanimity.
_MIN_CURLED_FINGERS_FOR_GRAB = 3

# Pinch thresholds, normalized Euclidean distance (x, y only -- z is noisy
# depth from MediaPipe and not reliable enough to include here).
#
# Hysteresis: entering pinch requires getting close (tight ENTER threshold);
# once pinching, fingers can drift a bit farther apart (looser EXIT
# threshold) before it releases. That gap is what stops rapid PINCH/OPEN
# flicker when your fingers hover right at the boundary.
#
# HOW TO TUNE:
#   - LIVE: the control panel has two sliders (pinch ENTER / pinch EXIT) that
#     retune tracking on the fly -- moving one publishes a ``GestureTuning``
#     snapshot to the tracking thread, which swaps the thresholds onto the
#     detectors it owns (see ``GestureDetector.set_thresholds`` +
#     ``src.tracking.TrackingWorker.set_tuning``). The slider values persist to
#     ``handy_config.json`` (keys ``pinch_enter`` / ``pinch_exit``) and pre-fill
#     the sliders next launch. Use the sliders to dial these in interactively.
#   - The two numbers below are only the built-in DEFAULTS (re-exported from
#     ``src.tuning``) used before any slider move / persisted config. Guidance:
#   - Pinch triggers too EARLY (fires before your fingers actually touch):
#     lower the ENTER threshold.
#   - Pinch triggers too LATE (you have to press fingers hard together
#     before it registers): raise the ENTER threshold.
#   - Gesture FLICKERS between PINCH and OPEN while held steady: widen the
#     enter/exit gap -- raise EXIT and/or lower ENTER. A wider gap absorbs
#     small hand tremor near the boundary instead of it crossing back over a
#     single threshold.
#   - Pinch releases too EAGERLY on small movement while held: raise the EXIT
#     threshold so it tolerates more drift before letting go.
#   - EXIT must stay strictly above ENTER or hysteresis has no effect and
#     flicker comes back; ``src.tuning.clamp_tuning`` enforces a minimum gap on
#     BOTH the slider and the config-load path so an inverted pair can't slip in.
#   - These are normalized distances (fraction of the frame, not pixels), so
#     they scale with hand distance from the camera automatically -- tune at
#     your typical demo distance from the webcam.
#
# Tuned by eye against typical index-thumb spacing at arm's length from a
# webcam.
PINCH_ENTER_THRESHOLD = DEFAULT_PINCH_ENTER
PINCH_EXIT_THRESHOLD = DEFAULT_PINCH_EXIT

Point2D = Tuple[float, float]


@dataclass(frozen=True)
class GestureState:
    """Immutable per-frame gesture result for one tracked hand."""

    is_pinching: bool
    pinch_point: Point2D
    is_grab: bool


# Returned when there is no hand to analyze this frame.
_NO_HAND_PINCH_POINT: Point2D = (0.0, 0.0)


def pinch_distance_and_midpoint(hand: Hand) -> Tuple[float, Point2D]:
    """Return (distance, midpoint) between thumb tip and index tip.

    Uses only x, y -- z (depth) is unreliable from MediaPipe's monocular
    estimate and would distort the distance, so it's ignored here.
    """
    tx, ty, _tz = hand[THUMB_TIP]
    ix, iy, _iz = hand[INDEX_TIP]
    distance = hypot(ix - tx, iy - ty)
    midpoint = ((tx + ix) / 2.0, (ty + iy) / 2.0)
    return distance, midpoint


def detect_grab(hand: Hand) -> bool:
    """Simple fist heuristic: most fingertips are curled toward the palm.

    A fingertip counts as "curled" when it's closer to the wrist than its
    own PIP joint is -- i.e. the tip has folded back past the knuckle,
    which is roughly what happens when a finger closes into a fist. This
    ignores the thumb (its geometry doesn't curl the same way) and only
    requires a majority of the other four fingers to be curled, since
    tracking noise or an partially-open pinky shouldn't prevent a grab.
    """
    wx, wy, _wz = hand[WRIST]
    curled_count = 0
    for tip_idx, pip_idx in zip(_CURL_FINGER_TIPS, _CURL_PIP_JOINTS):
        tipx, tipy, _ = hand[tip_idx]
        pipx, pipy, _ = hand[pip_idx]
        tip_to_wrist = hypot(tipx - wx, tipy - wy)
        pip_to_wrist = hypot(pipx - wx, pipy - wy)
        if tip_to_wrist < pip_to_wrist:
            curled_count += 1
    return curled_count >= _MIN_CURLED_FINGERS_FOR_GRAB


class GestureDetector:
    """Tracks pinch hysteresis state across frames for one hand.

    Single-hand MVP: one instance is enough. Call ``update`` once per frame
    with the current hand's landmarks (or ``None`` if no hand was detected).

    The enter/exit pinch thresholds are per-instance (defaulting to the module
    constants) so live panel tuning can retune detection: the owning tracking
    thread calls ``set_thresholds`` with the latest ``GestureTuning`` between
    frames. The hysteresis latch (``_is_pinching``) is preserved across a
    threshold swap -- only the comparison bounds change, so an in-flight pinch
    isn't dropped just because you nudged a slider.
    """

    def __init__(
        self,
        enter_threshold: float = PINCH_ENTER_THRESHOLD,
        exit_threshold: float = PINCH_EXIT_THRESHOLD,
    ) -> None:
        self._is_pinching = False
        self._enter_threshold = enter_threshold
        self._exit_threshold = exit_threshold

    def set_thresholds(self, enter_threshold: float, exit_threshold: float) -> None:
        """Swap in new pinch thresholds (owning tracking thread only).

        Mutates in place because this detector has a single owner (the tracking
        thread); the caller passes already-clamped values (see
        ``src.tuning.clamp_tuning``), so the enter < exit invariant holds. The
        hysteresis latch is intentionally left untouched.
        """
        self._enter_threshold = enter_threshold
        self._exit_threshold = exit_threshold

    def update(self, hand: Optional[Hand]) -> GestureState:
        """Compute this frame's gesture state, applying pinch hysteresis.

        No-hand case: returns a non-pinching state and resets the hysteresis
        latch (rather than holding the last state), since a lost hand means
        we have no evidence the pinch is still happening -- holding stale
        "pinching" state would risk a phantom gesture surviving a tracking
        dropout.
        """
        if hand is None:
            self._is_pinching = False
            return GestureState(
                is_pinching=False,
                pinch_point=_NO_HAND_PINCH_POINT,
                is_grab=False,
            )

        distance, midpoint = pinch_distance_and_midpoint(hand)

        if self._is_pinching:
            self._is_pinching = distance < self._exit_threshold
        else:
            self._is_pinching = distance < self._enter_threshold

        return GestureState(
            is_pinching=self._is_pinching,
            pinch_point=midpoint,
            is_grab=detect_grab(hand),
        )

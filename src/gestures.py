"""Pinch/grab gesture detection from normalized hand landmarks.

Pure math on normalized ``(x, y, z)`` landmark tuples (see ``src.hand_tracker``
for the ``Hand`` type). Pixel conversion happens only at draw time, per
project convention.
"""

from dataclasses import dataclass
from math import hypot
from typing import Optional, Tuple

from src.hand_tracker import Hand

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
# HOW TO TUNE (edit only the two numbers below):
#   - Pinch triggers too EARLY (fires before your fingers actually touch):
#     lower PINCH_ENTER_THRESHOLD.
#   - Pinch triggers too LATE (you have to press fingers hard together
#     before it registers): raise PINCH_ENTER_THRESHOLD.
#   - Gesture FLICKERS between PINCH and OPEN while held steady: widen the
#     enter/exit gap -- raise PINCH_EXIT_THRESHOLD and/or lower
#     PINCH_ENTER_THRESHOLD. A wider gap absorbs small hand tremor near the
#     boundary instead of it crossing back over a single threshold.
#   - Pinch releases too EAGERLY on small movement while held: raise
#     PINCH_EXIT_THRESHOLD so it tolerates more drift before letting go.
#   - Always keep PINCH_EXIT_THRESHOLD > PINCH_ENTER_THRESHOLD; if they're
#     equal (or inverted) hysteresis has no effect and flicker comes back.
#   - These are normalized distances (fraction of the frame, not pixels), so
#     they scale with hand distance from the camera automatically -- tune at
#     your typical demo distance from the webcam.
#
# Tuned by eye against typical index-thumb spacing at arm's length from a
# webcam.
PINCH_ENTER_THRESHOLD = 0.05
PINCH_EXIT_THRESHOLD = 0.07

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
    """

    def __init__(self) -> None:
        self._is_pinching = False

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
            self._is_pinching = distance < PINCH_EXIT_THRESHOLD
        else:
            self._is_pinching = distance < PINCH_ENTER_THRESHOLD

        return GestureState(
            is_pinching=self._is_pinching,
            pinch_point=midpoint,
            is_grab=detect_grab(hand),
        )

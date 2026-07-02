"""Draw a 21-point hand skeleton over a frame.

Pure drawing: hands arrive as normalized ``(x, y, z)`` landmarks and are
converted to pixel coordinates here (never before). Draws dots at each
landmark and lines along the standard MediaPipe hand-connection topology.
"""

from typing import List, Sequence, Tuple

import cv2

from src.gestures import GestureState
from src.hud import gesture_label as hud_gesture_label

# Standard MediaPipe 21-landmark hand topology, as (start, end) index pairs.
# Landmark map: 0 = wrist; thumb 1-4; index 5-8; middle 9-12; ring 13-16;
# pinky 17-20. Defined here as a module constant because mediapipe's
# `solutions.hands.HAND_CONNECTIONS` is unavailable in this build.
HAND_CONNECTIONS: Tuple[Tuple[int, int], ...] = (
    # Palm
    (0, 1), (0, 5), (0, 17), (5, 9), (9, 13), (13, 17),
    # Thumb
    (1, 2), (2, 3), (3, 4),
    # Index finger
    (5, 6), (6, 7), (7, 8),
    # Middle finger
    (9, 10), (10, 11), (11, 12),
    # Ring finger
    (13, 14), (14, 15), (15, 16),
    # Pinky
    (17, 18), (18, 19), (19, 20),
)

_LANDMARK_COLOR = (0, 0, 255)      # red dots (BGR)
_CONNECTION_COLOR = (0, 255, 0)    # green lines (BGR)
_LANDMARK_RADIUS = 4
_CONNECTION_THICKNESS = 2

# Gesture debug overlay (skeleton mode only).
_GESTURE_TEXT_COLOR = (255, 255, 255)   # white (BGR)
_GESTURE_TEXT_ORIGIN = (10, 30)         # top-left, in pixels
_GESTURE_FONT = cv2.FONT_HERSHEY_SIMPLEX
_GESTURE_FONT_SCALE = 1.0
_GESTURE_TEXT_THICKNESS = 2
_PINCH_POINT_COLOR = (0, 255, 255)      # yellow (BGR)
_PINCH_POINT_RADIUS = 8

def normalized_to_pixel(x: float, y: float, width: int, height: int) -> Tuple[int, int]:
    """Convert a normalized ``(x, y)`` in [0, 1] to integer pixel coordinates."""
    return int(round(x * width)), int(round(y * height))


def draw_skeleton(frame, hands: Sequence[List[Tuple[float, float, float]]]):
    """Draw every hand's landmarks and connections onto ``frame`` in place.

    ``hands`` is a list of hands, each a list of 21 normalized ``(x, y, z)``
    landmarks. Returns the same frame for convenient chaining.
    """
    height, width = frame.shape[:2]

    for hand in hands:
        points = [normalized_to_pixel(x, y, width, height) for (x, y, _z) in hand]

        for start_idx, end_idx in HAND_CONNECTIONS:
            if start_idx < len(points) and end_idx < len(points):
                cv2.line(
                    frame,
                    points[start_idx],
                    points[end_idx],
                    _CONNECTION_COLOR,
                    _CONNECTION_THICKNESS,
                )

        for point in points:
            cv2.circle(frame, point, _LANDMARK_RADIUS, _LANDMARK_COLOR, thickness=-1)

    return frame


def _gesture_label(gesture: GestureState) -> str:
    """Pick the debug label, delegating to hud's single label-priority rule.

    ``hand_present=True`` because this debug overlay is only drawn on frames
    where a hand was detected (see main.py).
    """
    return hud_gesture_label(
        is_grab=gesture.is_grab, is_pinching=gesture.is_pinching, hand_present=True
    )


def draw_gesture_debug(frame, gesture: GestureState):
    """Draw the current gesture label and a dot at the pinch point, in place.

    ``gesture.pinch_point`` is normalized (0-1); converted to pixels here,
    same as landmark drawing. Returns the same frame for convenient chaining.
    """
    height, width = frame.shape[:2]

    cv2.putText(
        frame,
        _gesture_label(gesture),
        _GESTURE_TEXT_ORIGIN,
        _GESTURE_FONT,
        _GESTURE_FONT_SCALE,
        _GESTURE_TEXT_COLOR,
        _GESTURE_TEXT_THICKNESS,
    )

    pinch_x, pinch_y = gesture.pinch_point
    point = normalized_to_pixel(pinch_x, pinch_y, width, height)
    cv2.circle(frame, point, _PINCH_POINT_RADIUS, _PINCH_POINT_COLOR, thickness=-1)

    return frame

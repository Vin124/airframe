"""Trash-corner remove gesture: drag a grabbed layer's pinch into the bottom-right
corner and release to delete it.

Replaces the old fist-to-remove (MediaPipe's fist detection flickered, so the
hold never landed cleanly). This reuses the reliable pinch-grab -- the only new
piece is a fixed screen zone and a hit-test -- so nothing new has to be detected.

Pure geometry + one cv2 draw (no camera/threads), unit-testable headlessly. The
MAIN loop owns the remove decision: it tracks the active grab's pinch point and,
on release inside the zone, removes the layer via the normal source-closing path.
"""

import numpy as np

from src.scene import Point2D, Size

# The trash zone is a square in the BOTTOM-RIGHT corner, this fraction of the
# frame's shorter side on each edge. Big enough to hit deliberately, small enough
# to stay out of the way of normal manipulation.
TRASH_ZONE_FRACTION = 0.16

# Zone chrome (BGR + alpha for the fill). Red = delete; brighter when armed.
_ZONE_FILL = (40, 40, 200)
_ZONE_FILL_ARMED = (60, 60, 255)
_ZONE_BORDER = (60, 60, 220)
_ZONE_ALPHA = 0.35          # idle fill opacity
_ZONE_ALPHA_ARMED = 0.6     # opacity while a grabbed pinch is over it
_LABEL = "TRASH"
_LABEL_COLOR = (255, 255, 255)
_LABEL_SCALE = 0.5
_LABEL_THICKNESS = 1


def trash_rect(frame_size: Size):
    """Pixel ``(x0, y0, x1, y1)`` of the trash zone in the bottom-right corner."""
    frame_w, frame_h = frame_size
    side = int(round(min(frame_w, frame_h) * TRASH_ZONE_FRACTION))
    return frame_w - side, frame_h - side, frame_w, frame_h


def in_trash_zone(point: Point2D, frame_size: Size) -> bool:
    """True when normalized ``point`` falls inside the trash zone. Never raises."""
    frame_w, frame_h = frame_size
    x0, y0, x1, y1 = trash_rect(frame_size)
    px, py = point[0] * frame_w, point[1] * frame_h
    return x0 <= px <= x1 and y0 <= py <= y1


def draw_trash_zone(frame: np.ndarray, armed: bool = False) -> None:
    """Draw the trash zone (a red corner box + label) onto ``frame`` in place.

    ``armed`` brightens/strengthens it while a grabbed pinch is hovering inside,
    so it's clear a release will delete. Main-thread draw on the private
    composited copy (so it shows in the preview + virtual cam).
    """
    frame_h, frame_w = frame.shape[:2]
    x0, y0, x1, y1 = trash_rect((frame_w, frame_h))
    fill = _ZONE_FILL_ARMED if armed else _ZONE_FILL
    alpha = _ZONE_ALPHA_ARMED if armed else _ZONE_ALPHA
    region = frame[y0:y1, x0:x1]
    if region.size:
        overlay = np.empty_like(region)
        overlay[:, :] = fill
        frame[y0:y1, x0:x1] = (alpha * overlay + (1.0 - alpha) * region).astype(frame.dtype)
    cv2_rectangle(frame, x0, y0, x1, y1)


def cv2_rectangle(frame: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> None:
    """Draw the zone border + centered label (split out so the blend stays pure-ish)."""
    import cv2  # local import keeps the module import light for pure-geometry tests

    cv2.rectangle(frame, (x0, y0), (x1 - 1, y1 - 1), _ZONE_BORDER, 2)
    (text_w, text_h), _baseline = cv2.getTextSize(
        _LABEL, cv2.FONT_HERSHEY_SIMPLEX, _LABEL_SCALE, _LABEL_THICKNESS
    )
    tx = x0 + max(0, ((x1 - x0) - text_w) // 2)
    ty = y0 + ((y1 - y0) + text_h) // 2
    cv2.putText(
        frame, _LABEL, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
        _LABEL_SCALE, _LABEL_COLOR, _LABEL_THICKNESS, cv2.LINE_AA,
    )

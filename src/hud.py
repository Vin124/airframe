"""Heads-up display: mode, FPS, gesture state, and key legend.

Text composition is pure (``gesture_label``, ``resize_or_grab_status``,
``build_hud_lines``) so it's unit-testable without a drawing surface; only
``draw_hud`` touches cv2. Drawn as a semi-transparent dark strip in the
frame's bottom-left corner -- deliberately away from both the frame center
(where hand/image interaction happens) and the top-left gesture-debug label
drawn by ``src.skeleton`` in skeleton-overlay mode.
"""

from typing import List, Optional

import cv2
import numpy as np

# Layout.
_MARGIN_PX = 10
_PADDING_PX = 8
_LINE_HEIGHT_PX = 22
_STRIP_WIDTH_PX = 260

# Style.
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.5
_FONT_THICKNESS = 1
_TEXT_COLOR = (255, 255, 255)  # white (BGR)
_BG_COLOR = (0, 0, 0)  # black (BGR)
_BG_ALPHA = 0.5  # strip opacity; frame shows through underneath

GESTURE_LABEL_GRAB = "GRAB"
GESTURE_LABEL_PINCH = "PINCH"
GESTURE_LABEL_OPEN = "OPEN"
GESTURE_LABEL_NONE = "--"  # no hand currently tracked

STATUS_GRABBED = "GRABBED"
STATUS_RESIZING = "RESIZING"

KEY_LEGEND = "q quit | s skeleton | r reset | +/- scale"

_NO_FPS_TEXT = "--"


def gesture_label(is_grab: bool, is_pinching: bool, hand_present: bool) -> str:
    """Pick the HUD's one-word gesture label.

    Priority GRAB > PINCH > OPEN mirrors ``src.skeleton``'s debug label so
    the two never disagree; returns ``GESTURE_LABEL_NONE`` when no hand is
    currently tracked.
    """
    if not hand_present:
        return GESTURE_LABEL_NONE
    if is_grab:
        return GESTURE_LABEL_GRAB
    if is_pinching:
        return GESTURE_LABEL_PINCH
    return GESTURE_LABEL_OPEN


def resize_or_grab_status(is_grabbed: bool, is_resizing: bool) -> Optional[str]:
    """Return the image-mode interaction status label, or None when idle.

    Resize takes priority in the (impossible in practice, but cheap to be
    defensive about) case both are true, matching ``src.overlay.step``'s own
    priority of two-hand resize over single-hand grab.
    """
    if is_resizing:
        return STATUS_RESIZING
    if is_grabbed:
        return STATUS_GRABBED
    return None


def build_hud_lines(
    mode_name: str,
    fps: Optional[float],
    gesture: str,
    status: Optional[str] = None,
) -> List[str]:
    """Compose the HUD's text lines, in top-to-bottom draw order.

    ``fps`` is the last reported FpsCounter value (may be None before the
    first one-second window elapses). ``status`` is extra image-mode state
    (e.g. "GRABBED", "RESIZING"); omitted from the HUD when None.
    """
    fps_text = f"{fps:.1f}" if fps is not None else _NO_FPS_TEXT
    lines = [
        f"Mode: {mode_name}",
        f"FPS: {fps_text}",
        f"Gesture: {gesture}",
    ]
    if status:
        lines.append(f"Status: {status}")
    lines.append(KEY_LEGEND)
    return lines


def draw_hud(frame, lines: List[str]) -> None:
    """Draw ``lines`` on a semi-transparent strip in the bottom-left corner, in place.

    Strip height grows with the number of lines; width and position are
    clamped to the frame so this never raises on very small frames.
    """
    if not lines:
        return

    frame_h, frame_w = frame.shape[:2]
    strip_width = max(1, min(_STRIP_WIDTH_PX, frame_w - 2 * _MARGIN_PX))
    strip_height = _PADDING_PX * 2 + _LINE_HEIGHT_PX * len(lines)

    x0 = _MARGIN_PX
    y1 = frame_h - _MARGIN_PX
    y0 = max(0, y1 - strip_height)
    x1 = min(frame_w, x0 + strip_width)
    if x1 <= x0 or y1 <= y0:
        return

    # Blend only the strip region (a view into the frame) -- copying and
    # re-blending the whole frame here costs a visible chunk of FPS.
    region = frame[y0:y1, x0:x1]
    strip = np.empty_like(region)
    strip[:] = _BG_COLOR
    cv2.addWeighted(strip, _BG_ALPHA, region, 1.0 - _BG_ALPHA, 0.0, dst=region)

    for i, line in enumerate(lines):
        baseline_y = y0 + _PADDING_PX + _LINE_HEIGHT_PX * (i + 1) - 6
        cv2.putText(
            frame,
            line,
            (x0 + _PADDING_PX, baseline_y),
            _FONT,
            _FONT_SCALE,
            _TEXT_COLOR,
            _FONT_THICKNESS,
            cv2.LINE_AA,
        )

"""Image-overlay state and interaction for image mode.

Loads an image with alpha, holds its position/scale as an immutable
``Transform`` (center in NORMALIZED 0-1 coords + a scale multiplier on the
image's native pixel size), and alpha-composites it onto each frame at that
transform. Pixel conversion happens only at composite/hit-test time, per
project convention.

Interaction is modelled as small pure state machines over immutable frozen
dataclasses so the whole thing is testable without a camera:

    - ``GrabState``   : pinch-start-on-image grabs; move follows the pinch
                        point (keeping the grab offset); un-pinch / hand-loss
                        releases.
    - ``ResizeState`` : two pinching hands scale the image by the ratio of the
                        current inter-pinch distance to the distance captured
                        when the two-hand gesture began.
    - ``InteractionState`` : bundles transform + grab + resize; ``step`` advances
                        it one frame from the two hands' gesture states.
"""

from dataclasses import dataclass, replace
from math import hypot
from typing import List, Protocol, Tuple

import cv2
import numpy as np

Point2D = Tuple[float, float]
Size = Tuple[int, int]  # (width, height) in pixels


class GestureLike(Protocol):
    """Structural type for what the interaction machines read off a gesture.

    ``src.gestures.GestureState`` satisfies it; tests can pass any object with
    these two attributes.
    """

    is_pinching: bool
    pinch_point: Point2D

# Scale is a multiplier on the image's native pixel size. Clamped so the image
# can neither vanish nor swallow the whole frame.
MIN_SCALE = 0.1
MAX_SCALE = 6.0
DEFAULT_SCALE = 1.0

# Center is clamped to the frame so the image can't be dragged fully off-screen:
# the center point always stays inside the frame, keeping part of the image
# visible.
CENTER_MIN = 0.0
CENTER_MAX = 1.0

# Below this normalized inter-pinch distance a two-hand resize is treated as
# degenerate (hands effectively coincident) and no scaling is applied, avoiding
# division by ~zero.
MIN_RESIZE_DISTANCE = 1e-4


class OverlayError(RuntimeError):
    """Raised when the overlay image can't be loaded or read."""


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass(frozen=True)
class Transform:
    """Immutable overlay transform: center (normalized) + scale multiplier.

    Values are clamped on construction, so every instance is valid and the
    functional update helpers below don't need to re-clamp.
    """

    center_x: float = 0.5
    center_y: float = 0.5
    scale: float = DEFAULT_SCALE

    def __post_init__(self) -> None:
        object.__setattr__(self, "center_x", _clamp(self.center_x, CENTER_MIN, CENTER_MAX))
        object.__setattr__(self, "center_y", _clamp(self.center_y, CENTER_MIN, CENTER_MAX))
        object.__setattr__(self, "scale", _clamp(self.scale, MIN_SCALE, MAX_SCALE))

    def moved_to(self, center_x: float, center_y: float) -> "Transform":
        """Return a copy centered at ``(center_x, center_y)`` (normalized, clamped)."""
        return Transform(center_x, center_y, self.scale)

    def scaled_by(self, factor: float) -> "Transform":
        """Return a copy with the scale multiplied by ``factor`` (clamped)."""
        return Transform(self.center_x, self.center_y, self.scale * factor)

    def with_scale(self, scale: float) -> "Transform":
        """Return a copy with an absolute ``scale`` (clamped)."""
        return Transform(self.center_x, self.center_y, scale)


def scaled_size(base_size: Size, scale: float) -> Size:
    """Return the image's pixel size at ``scale``, floored to at least 1x1."""
    base_w, base_h = base_size
    return max(1, int(round(base_w * scale))), max(1, int(round(base_h * scale)))


def top_left_px(transform: Transform, scaled: Size, frame_size: Size) -> Tuple[int, int]:
    """Return the top-left pixel of the scaled image for ``transform``."""
    scaled_w, scaled_h = scaled
    frame_w, frame_h = frame_size
    cx = transform.center_x * frame_w
    cy = transform.center_y * frame_h
    return int(round(cx - scaled_w / 2.0)), int(round(cy - scaled_h / 2.0))


def _ensure_bgra(image: np.ndarray) -> np.ndarray:
    """Return ``image`` as an HxWx4 BGRA array, adding an opaque alpha if absent."""
    if image.ndim == 2:  # grayscale -> BGR
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 3:
        alpha = np.full(image.shape[:2] + (1,), 255, dtype=image.dtype)
        image = np.concatenate([image, alpha], axis=2)
    return image


class Overlay:
    """A loaded BGRA image plus the pure rendering/hit-test operations on it.

    The image itself is immutable state; the evolving position/scale lives in a
    separate ``Transform`` that callers thread through ``composite`` /
    ``contains`` each frame.
    """

    def __init__(self, image_bgra: np.ndarray) -> None:
        self.image = _ensure_bgra(image_bgra)

    @classmethod
    def load(cls, path: str) -> "Overlay":
        """Load an image (with alpha) from ``path``.

        Raises OverlayError with a clear message if the file is missing or not a
        readable image. A missing alpha channel is filled in opaque.
        """
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise OverlayError(
                f"Could not load overlay image at '{path}'. Check that the path "
                "exists and points to a readable image file."
            )
        return cls(image)

    @property
    def base_size(self) -> Size:
        """Native (width, height) of the image in pixels."""
        return self.image.shape[1], self.image.shape[0]

    def rect_px(self, transform: Transform, frame_size: Size) -> Tuple[int, int, int, int]:
        """Return the image's on-frame rect as ``(x0, y0, width, height)`` in pixels."""
        scaled_w, scaled_h = scaled_size(self.base_size, transform.scale)
        x0, y0 = top_left_px(transform, (scaled_w, scaled_h), frame_size)
        return x0, y0, scaled_w, scaled_h

    def contains(self, transform: Transform, point: Point2D, frame_size: Size) -> bool:
        """Hit-test: is the normalized ``point`` inside the image's current rect?"""
        x0, y0, w, h = self.rect_px(transform, frame_size)
        frame_w, frame_h = frame_size
        px = point[0] * frame_w
        py = point[1] * frame_h
        return x0 <= px < x0 + w and y0 <= py < y0 + h

    def composite(self, frame: np.ndarray, transform: Transform) -> np.ndarray:
        """Alpha-blend the scaled image onto ``frame`` in place at ``transform``.

        Clips at all four frame edges (partially off-screen is fine; fully
        off-screen is a no-op) and guards against a zero-size scaled image.
        Returns the same frame for chaining.
        """
        frame_h, frame_w = frame.shape[:2]
        scaled_w, scaled_h = scaled_size(self.base_size, transform.scale)
        if scaled_w < 1 or scaled_h < 1:
            return frame

        interpolation = cv2.INTER_AREA if transform.scale < 1.0 else cv2.INTER_LINEAR
        scaled = cv2.resize(self.image, (scaled_w, scaled_h), interpolation=interpolation)

        x0, y0 = top_left_px(transform, (scaled_w, scaled_h), (frame_w, frame_h))

        # Intersection of the image rect with the frame, in frame pixels.
        fx0, fy0 = max(0, x0), max(0, y0)
        fx1, fy1 = min(frame_w, x0 + scaled_w), min(frame_h, y0 + scaled_h)
        if fx0 >= fx1 or fy0 >= fy1:
            return frame  # fully off-screen

        # Matching region inside the scaled overlay.
        ox0, oy0 = fx0 - x0, fy0 - y0
        ox1, oy1 = ox0 + (fx1 - fx0), oy0 + (fy1 - fy0)

        overlay_region = scaled[oy0:oy1, ox0:ox1]
        frame_region = frame[fy0:fy1, fx0:fx1]

        alpha = overlay_region[:, :, 3:4].astype(np.float32) / 255.0
        overlay_bgr = overlay_region[:, :, :3].astype(np.float32)
        blended = alpha * overlay_bgr + (1.0 - alpha) * frame_region.astype(np.float32)
        frame[fy0:fy1, fx0:fx1] = blended.astype(frame.dtype)
        return frame


@dataclass(frozen=True)
class GrabState:
    """Single-hand grab machine state.

    ``offset`` is the normalized (center - pinch_point) captured at grab time so
    the image tracks the hand without jump-centering. ``was_pinching`` is the
    previous frame's pinch flag, used to detect a pinch START (rising edge).
    """

    is_grabbed: bool = False
    offset: Point2D = (0.0, 0.0)
    was_pinching: bool = False


@dataclass(frozen=True)
class ResizeState:
    """Two-hand resize machine state, capturing the gesture's starting frame."""

    is_resizing: bool = False
    start_distance: float = 0.0
    start_scale: float = DEFAULT_SCALE


@dataclass(frozen=True)
class InteractionState:
    """Everything image mode threads frame-to-frame."""

    transform: Transform = Transform()
    grab: GrabState = GrabState()
    resize: ResizeState = ResizeState()

    @classmethod
    def initial(cls) -> "InteractionState":
        """A centered, un-grabbed, un-resizing starting state."""
        return cls()

    def with_transform(self, transform: Transform) -> "InteractionState":
        """Return a copy with a replaced transform (for e.g. keyboard scaling)."""
        return replace(self, transform=transform)


def update_grab(
    grab: GrabState,
    transform: Transform,
    overlay: Overlay,
    is_pinching: bool,
    pinch_point: Point2D,
    frame_size: Size,
) -> Tuple[GrabState, Transform]:
    """Advance the single-hand grab machine one frame.

    IDLE -> GRABBED only on a pinch that STARTS over the image (a pinch that
    starts off-image is consumed and never grabs, even if later dragged over).
    While GRABBED the center follows the pinch point at the captured offset.
    Un-pinch (which also covers hand loss, since a lost hand reports
    not-pinching) releases back to IDLE.
    """
    if grab.is_grabbed:
        if not is_pinching:
            return GrabState(is_grabbed=False, offset=(0.0, 0.0), was_pinching=False), transform
        new_center = (pinch_point[0] + grab.offset[0], pinch_point[1] + grab.offset[1])
        moved = transform.moved_to(*new_center)
        return GrabState(is_grabbed=True, offset=grab.offset, was_pinching=True), moved

    pinch_started = is_pinching and not grab.was_pinching
    if pinch_started and overlay.contains(transform, pinch_point, frame_size):
        offset = (transform.center_x - pinch_point[0], transform.center_y - pinch_point[1])
        return GrabState(is_grabbed=True, offset=offset, was_pinching=True), transform
    return GrabState(is_grabbed=False, offset=grab.offset, was_pinching=is_pinching), transform


def update_resize(
    resize: ResizeState,
    transform: Transform,
    both_pinching: bool,
    pinch_point_a: Point2D,
    pinch_point_b: Point2D,
) -> Tuple[ResizeState, Transform]:
    """Advance the two-hand resize machine one frame.

    When both hands pinch, the first such frame captures the inter-pinch
    distance and current scale; subsequent frames set scale to
    ``start_scale * (current_distance / start_distance)``. Losing either pinch
    ends the gesture, so it re-captures a fresh baseline next time (handles
    hands appearing/disappearing mid-resize gracefully).
    """
    if not both_pinching:
        return ResizeState(), transform

    distance = hypot(pinch_point_b[0] - pinch_point_a[0], pinch_point_b[1] - pinch_point_a[1])
    if not resize.is_resizing:
        started = ResizeState(is_resizing=True, start_distance=distance, start_scale=transform.scale)
        return started, transform
    if resize.start_distance < MIN_RESIZE_DISTANCE:
        return resize, transform
    factor = distance / resize.start_distance
    return resize, transform.with_scale(resize.start_scale * factor)


def step(
    state: InteractionState,
    overlay: Overlay,
    gesture_states: List[GestureLike],
    frame_size: Size,
) -> InteractionState:
    """Advance the whole interaction one frame from both hands' gesture states.

    ``gesture_states`` is a length-2 list (slot 0 = primary hand); each item
    only needs ``.is_pinching`` and ``.pinch_point``. Two-hand resize takes
    priority over single-hand grab when both hands pinch, so the primary hand's
    pinch can't fight the resize; grab is idled but its edge memory is kept
    coherent so ending a resize doesn't spuriously auto-grab.
    """
    primary = gesture_states[0]
    secondary = gesture_states[1]
    both_pinching = bool(primary.is_pinching and secondary.is_pinching)

    if both_pinching:
        resize, transform = update_resize(
            state.resize, state.transform, True, primary.pinch_point, secondary.pinch_point
        )
        grab = GrabState(is_grabbed=False, offset=(0.0, 0.0), was_pinching=True)
        return InteractionState(transform=transform, grab=grab, resize=resize)

    resize, transform = update_resize(
        state.resize, state.transform, False, primary.pinch_point, secondary.pinch_point
    )
    grab, transform = update_grab(
        state.grab, transform, overlay, primary.is_pinching, primary.pinch_point, frame_size
    )
    return InteractionState(transform=transform, grab=grab, resize=resize)

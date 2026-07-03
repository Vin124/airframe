"""Gesture-driven grab / move / resize state machines over the selected layer.

Image mode's interaction is modelled as small pure state machines over frozen
dataclasses so the whole thing is testable without a camera:

    - ``GrabState``   : a pinch that STARTS over the selected layer grabs it;
                        while held, the layer's center follows the pinch point
                        (keeping the grab offset); un-pinch / hand-loss releases.
    - ``ResizeState`` : two pinching hands scale the layer by the ratio of the
                        current inter-pinch distance to the distance captured
                        when the two-hand gesture began.
    - ``InteractionState`` : bundles grab + resize; ``step`` advances it one
                        frame from the two hands' gesture states and returns the
                        updated interaction plus the updated ``Scene``.

``step`` takes the whole ``Scene``: on a pinch rising edge it selects the
top-most layer under the pinch point (so you grab whichever overlay is on top
there), then advances grab/resize on the selected layer and writes it back. The
layer owns the ``Transform`` (its geometry); this module reads it, advances the
machines, and writes the new transform back onto the layer. Two-hand resize
outranks single-hand grab, matching the pre-refactor behavior exactly.
"""

from dataclasses import dataclass
from math import hypot
from typing import List, Optional, Protocol, Tuple

from src.scene import DEFAULT_SCALE, Layer, Scene, Size, Transform, contains

Point2D = Tuple[float, float]


class GestureLike(Protocol):
    """Structural type for what the interaction machines read off a gesture.

    ``src.gestures.GestureState`` satisfies it; tests can pass any object with
    these two attributes.
    """

    is_pinching: bool
    pinch_point: Point2D


# Below this normalized inter-pinch distance a two-hand resize is treated as
# degenerate (hands effectively coincident) and no scaling is applied, avoiding
# division by ~zero.
MIN_RESIZE_DISTANCE = 1e-4


@dataclass(frozen=True)
class GrabState:
    """Single-hand grab machine state.

    ``offset`` is the normalized (center - pinch_point) captured at grab time so
    the layer tracks the hand without jump-centering. ``was_pinching`` is the
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
    """Transient interaction state threaded frame-to-frame (grab + resize).

    The layer's ``Transform`` is deliberately NOT stored here -- it lives on the
    ``Layer`` in the scene. This holds only the machine state that has no home
    in the scene graph.
    """

    grab: GrabState = GrabState()
    resize: ResizeState = ResizeState()

    @classmethod
    def initial(cls) -> "InteractionState":
        """An un-grabbed, un-resizing starting state."""
        return cls()


def update_grab(
    grab: GrabState,
    transform: Transform,
    base_size: Optional[Size],
    is_pinching: bool,
    pinch_point: Point2D,
    frame_size: Size,
) -> Tuple[GrabState, Transform]:
    """Advance the single-hand grab machine one frame.

    IDLE -> GRABBED only on a pinch that STARTS over the layer (a pinch that
    starts off-layer is consumed and never grabs, even if later dragged over).
    While GRABBED the center follows the pinch point at the captured offset.
    Un-pinch (which also covers hand loss, since a lost hand reports
    not-pinching) releases back to IDLE. ``base_size`` is the layer's native
    size, used for the hit-test; ``None`` (a source with no frame yet) means
    the layer has no rect, so the pinch is consumed without grabbing.
    """
    if grab.is_grabbed:
        if not is_pinching:
            return GrabState(is_grabbed=False, offset=(0.0, 0.0), was_pinching=False), transform
        new_center = (pinch_point[0] + grab.offset[0], pinch_point[1] + grab.offset[1])
        moved = transform.moved_to(*new_center)
        return GrabState(is_grabbed=True, offset=grab.offset, was_pinching=True), moved

    pinch_started = is_pinching and not grab.was_pinching
    if (
        pinch_started
        and base_size is not None
        and contains(transform, base_size, pinch_point, frame_size)
    ):
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


def resolve_selection(
    state: InteractionState,
    scene: Scene,
    primary: GestureLike,
    frame_size: Size,
) -> Scene:
    """Pick the layer a fresh pinch grabs: the top-most under the pinch point.

    Runs only on a pinch RISING EDGE and only while idle: a pinch that starts
    mid-grab / mid-resize can't steal the selection (the active gesture owns it),
    and a pinch that starts over bare background (no layer under it) leaves the
    current selection untouched -- the edge is still consumed downstream, so it
    grabs nothing. Otherwise the top-most containing layer becomes selected, so
    the very same frame's ``update_grab`` grabs the layer we just picked.
    """
    in_gesture = state.grab.is_grabbed or state.resize.is_resizing
    pinch_started = primary.is_pinching and not state.grab.was_pinching
    if not pinch_started or in_gesture:
        return scene
    hit = scene.top_layer_at(primary.pinch_point, frame_size)
    if hit is None:
        return scene
    return scene.select_layer(hit)


def _advance_selected(
    state: InteractionState,
    layer: Layer,
    primary: GestureLike,
    secondary: GestureLike,
    frame_size: Size,
) -> Tuple[InteractionState, Layer]:
    """Advance grab/resize on the selected ``layer`` (pre-Phase-8 single-layer body).

    Two-hand resize takes priority over single-hand grab when both hands pinch,
    so the primary hand's pinch can't fight the resize; grab is idled but its
    edge memory is kept coherent so ending a resize doesn't spuriously auto-grab.
    """
    transform = layer.transform
    base_size = layer.base_size
    both_pinching = bool(primary.is_pinching and secondary.is_pinching)

    if both_pinching:
        resize, transform = update_resize(
            state.resize, transform, True, primary.pinch_point, secondary.pinch_point
        )
        grab = GrabState(is_grabbed=False, offset=(0.0, 0.0), was_pinching=True)
        return InteractionState(grab=grab, resize=resize), layer.with_transform(transform)

    resize, transform = update_resize(
        state.resize, transform, False, primary.pinch_point, secondary.pinch_point
    )
    grab, transform = update_grab(
        state.grab, transform, base_size, primary.is_pinching, primary.pinch_point, frame_size
    )
    return InteractionState(grab=grab, resize=resize), layer.with_transform(transform)


def step(
    state: InteractionState,
    scene: Scene,
    gesture_states: List[GestureLike],
    frame_size: Size,
) -> Tuple[InteractionState, Scene]:
    """Advance interaction one frame over ``scene`` from both hands.

    First resolves selection (``resolve_selection``): a pinch that starts over a
    layer selects the top-most one under it, so the same frame grabs exactly the
    layer the hand is on; an in-progress gesture can't be stolen. Then advances
    grab/resize on the selected layer and writes it back into the scene.

    ``gesture_states`` is a length-2 list (slot 0 = primary hand); each item only
    needs ``.is_pinching`` and ``.pinch_point``. Tolerates an empty / unselected
    scene (legal once the last layer is removed at runtime): nothing moves, but
    the pinch edge memory is still advanced so a later pinch is a clean edge.

    Returns the updated ``InteractionState`` and the updated ``Scene``.
    """
    primary = gesture_states[0]
    secondary = gesture_states[1]

    scene = resolve_selection(state, scene, primary, frame_size)

    layer = scene.selected_layer
    if layer is None:
        grab = GrabState(
            is_grabbed=False, offset=state.grab.offset, was_pinching=primary.is_pinching
        )
        return InteractionState(grab=grab, resize=ResizeState()), scene

    new_state, layer = _advance_selected(state, layer, primary, secondary, frame_size)
    return new_state, scene.replace_selected(layer)

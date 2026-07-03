"""Unit tests for Phase 11 air-drawing wiring on the main loop.

Covers the PipelineRuntime stroke lifecycle (append / clear / release-freeze),
the draw-vs-pinch precedence rule (suppress_draw_pinch, standalone + through
interaction.step), and the c/k/[/] drawing keys. Headless -- fake webcam, no
sinks/worker/camera/display, synthetic gesture states only.
"""

import numpy as np
import pytest

from src.drawing import (
    DEFAULT_STROKE_WIDTH,
    MAX_STROKE_WIDTH,
    MIN_STROKE_WIDTH,
    STROKE_PALETTE,
    STROKE_WIDTH_STEP,
    Stroke,
)
from src.gestures import GestureState
from src.interaction import InteractionState, step
from src.main import (
    CLEAR_STROKE_KEY,
    CYCLE_COLOR_KEY,
    STROKE_WIDTH_DOWN_KEY,
    STROKE_WIDTH_UP_KEY,
    PipelineRuntime,
    parse_args,
    suppress_draw_pinch,
)
from src.scene import Layer, Scene, Transform
from src.sources import ArraySource
from src.sources.base import Size, Source

FRAME_SIZE = (100, 100)


class _FakePacketWebcam:
    """Minimal ThreadedSource stand-in (unused by the drawing paths)."""

    def latest_packet(self):
        return None


class _FakeSource(Source):
    """A source with a fixed base_size for building selectable layers."""

    def __init__(self, base_size: Size = (10, 10)) -> None:
        self._base_size = base_size

    def read(self):
        return None

    @property
    def base_size(self) -> Size:
        return self._base_size


def _image_runtime(scene=None, interaction=None) -> PipelineRuntime:
    args = parse_args(["--mode", "image", "--image", "assets/test.png"])
    return PipelineRuntime(
        args,
        _FakePacketWebcam(),
        None,
        scene if scene is not None else Scene(),
        interaction if interaction is not None else InteractionState.initial(),
        None,
        None,
    )


def _draw(point) -> GestureState:
    """A drawing gesture (is_draw latched) reporting its fingertip point."""
    return GestureState(
        is_pinching=False, pinch_point=(0.0, 0.0), is_grab=False,
        is_draw=True, draw_point=point,
    )


def _idle() -> GestureState:
    return GestureState(is_pinching=False, pinch_point=(0.0, 0.0), is_grab=False)


def _feed_draw(runtime, points) -> None:
    for point in points:
        runtime._advance_drawing([_draw(point), _idle()], FRAME_SIZE)


# --------------------------------------------------------------------------- #
# Stroke lifecycle: append while drawing
# --------------------------------------------------------------------------- #

def test_advance_drawing_starts_and_extends_a_stroke():
    rt = _image_runtime()
    assert rt._stroke is None
    _feed_draw(rt, [(0.3, 0.4)])
    assert rt._stroke is not None
    assert rt._stroke.points == ((0.3, 0.4),)
    _feed_draw(rt, [(0.35, 0.45)])
    assert rt._stroke.points == ((0.3, 0.4), (0.35, 0.45))


def test_advance_drawing_ignores_hand_when_not_drawing():
    rt = _image_runtime()
    rt._advance_drawing([_idle(), _idle()], FRAME_SIZE)
    assert rt._stroke is None


# --------------------------------------------------------------------------- #
# Release: freeze into a scene layer (>= 2 points), discard otherwise
# --------------------------------------------------------------------------- #

def test_release_freezes_stroke_into_selected_array_layer():
    rt = _image_runtime(scene=Scene())
    _feed_draw(rt, [(0.2, 0.6), (0.25, 0.65), (0.3, 0.7), (0.35, 0.75), (0.4, 0.8)])
    assert len(rt.scene.layers) == 0  # nothing frozen while still drawing
    # Release: a non-draw gesture on the next tick.
    rt._advance_drawing([_idle(), _idle()], FRAME_SIZE)
    assert rt._stroke is None
    assert len(rt.scene.layers) == 1
    frozen = rt.scene.selected_layer
    assert isinstance(frozen.source, ArraySource)
    assert frozen.source.base_size is not None  # a real rasterized image
    assert frozen.source.read().shape[2] == 4  # BGRA


def test_release_places_layer_where_it_was_drawn():
    rt = _image_runtime(scene=Scene())
    # Stroke centered around (0.3, 0.7) -- clearly off the default (0.5, 0.5).
    _feed_draw(rt, [(0.2, 0.6), (0.25, 0.65), (0.3, 0.7), (0.35, 0.75), (0.4, 0.8)])
    rt._advance_drawing([_idle(), _idle()], FRAME_SIZE)
    t = rt.scene.selected_layer.transform
    assert 0.25 < t.center_x < 0.35
    assert 0.65 < t.center_y < 0.75


def test_release_with_one_point_discards_without_a_layer():
    rt = _image_runtime(scene=Scene())
    _feed_draw(rt, [(0.5, 0.5)])  # single point -> not a stroke
    rt._advance_drawing([_idle(), _idle()], FRAME_SIZE)
    assert rt._stroke is None
    assert len(rt.scene.layers) == 0


def test_hand_loss_freezes_active_stroke():
    # A tracking dropout publishes NO gesture states at all; the in-progress
    # stroke must freeze into a layer rather than dangle (or crash on the
    # missing primary hand).
    rt = _image_runtime(scene=Scene())
    _feed_draw(rt, [(0.2, 0.2), (0.3, 0.3), (0.4, 0.4)])
    rt._advance_drawing([], FRAME_SIZE)
    assert rt._stroke is None
    assert len(rt.scene.layers) == 1


def test_release_resets_interaction_so_new_layer_is_not_auto_grabbed():
    from src.interaction import GrabState
    grabbed = InteractionState(grab=GrabState(is_grabbed=True, was_pinching=True))
    rt = _image_runtime(scene=Scene(), interaction=grabbed)
    _feed_draw(rt, [(0.4, 0.4), (0.5, 0.5), (0.6, 0.6)])
    rt._advance_drawing([_idle(), _idle()], FRAME_SIZE)
    assert rt.interaction.grab.is_grabbed is False


# --------------------------------------------------------------------------- #
# clear_stroke: discard the in-progress stroke without freezing
# --------------------------------------------------------------------------- #

def test_clear_stroke_discards_without_freezing():
    rt = _image_runtime(scene=Scene())
    _feed_draw(rt, [(0.3, 0.3), (0.4, 0.4)])
    assert rt._stroke is not None
    rt.clear_stroke()
    assert rt._stroke is None
    assert len(rt.scene.layers) == 0  # nothing frozen


def test_drawing_keys_inert_in_skeleton_mode():
    args = parse_args(["--mode", "skeleton"])
    rt = PipelineRuntime(
        args, _FakePacketWebcam(), None, Scene(), InteractionState.initial(), None, None
    )
    assert rt._handle_keys(CYCLE_COLOR_KEY) is True
    assert rt._stroke_color_index == 0  # 'k' gated to image mode
    rt._stroke = Stroke().append((0.1, 0.1))
    assert rt._handle_keys(CLEAR_STROKE_KEY) is True
    assert rt._stroke is not None  # 'c' gated to image mode


# --------------------------------------------------------------------------- #
# Gesture precedence: suppress_draw_pinch (standalone + through step)
# --------------------------------------------------------------------------- #

def test_suppress_draw_pinch_masks_pinch_only_for_drawing_hand():
    drawing_hand = GestureState(
        is_pinching=True, pinch_point=(0.5, 0.5), is_grab=False,
        is_draw=True, draw_point=(0.5, 0.5),
    )
    normal_hand = GestureState(is_pinching=True, pinch_point=(0.2, 0.2), is_grab=False)
    out = suppress_draw_pinch([drawing_hand, normal_hand])
    # Drawing hand: pinch masked off, but point/draw flag preserved.
    assert out[0].is_pinching is False
    assert out[0].pinch_point == (0.5, 0.5)
    assert out[0].is_draw is True
    # Non-drawing hand: passed through untouched.
    assert out[1] is normal_hand
    assert out[1].is_pinching is True


def test_suppress_draw_pinch_does_not_mutate_inputs():
    drawing_hand = GestureState(
        is_pinching=True, pinch_point=(0.5, 0.5), is_grab=False, is_draw=True
    )
    suppress_draw_pinch([drawing_hand])
    assert drawing_hand.is_pinching is True  # original frozen state unchanged


def test_drawing_hand_pinch_does_not_grab_but_a_real_pinch_does():
    scene = Scene.single(Layer(source=_FakeSource((20, 20)), transform=Transform()))
    idle = _idle()
    # Drawing + (spuriously) pinching over the layer: suppression blocks the grab.
    draw_pinch = GestureState(
        is_pinching=True, pinch_point=(0.5, 0.5), is_grab=False,
        is_draw=True, draw_point=(0.5, 0.5),
    )
    i, _s = step(
        InteractionState.initial(), scene, suppress_draw_pinch([draw_pinch, idle]), FRAME_SIZE
    )
    assert i.grab.is_grabbed is False
    # Control: the identical pinch WITHOUT drawing does grab the layer.
    real_pinch = GestureState(is_pinching=True, pinch_point=(0.5, 0.5), is_grab=False)
    i2, _s2 = step(
        InteractionState.initial(), scene, suppress_draw_pinch([real_pinch, idle]), FRAME_SIZE
    )
    assert i2.grab.is_grabbed is True


# --------------------------------------------------------------------------- #
# Drawing keys: clear (c), cycle color (k), width down/up ([ / ])
# --------------------------------------------------------------------------- #

def test_drawing_key_clear_discards_stroke():
    rt = _image_runtime()
    rt._stroke = Stroke().append((0.1, 0.1)).append((0.2, 0.2))
    rt._handle_drawing_key(CLEAR_STROKE_KEY)
    assert rt._stroke is None


def test_drawing_key_cycle_color_advances_palette():
    rt = _image_runtime()
    assert rt._stroke_color_index == 0
    rt._handle_drawing_key(CYCLE_COLOR_KEY)
    assert rt._stroke_color_index == 1
    assert rt._stroke_color() == STROKE_PALETTE[1]


def test_drawing_key_width_up_and_down_are_clamped():
    rt = _image_runtime()
    assert rt._stroke_width == DEFAULT_STROKE_WIDTH
    rt._handle_drawing_key(STROKE_WIDTH_UP_KEY)
    assert rt._stroke_width == DEFAULT_STROKE_WIDTH + STROKE_WIDTH_STEP
    # Hammer it down past the floor -> clamps at MIN.
    for _ in range(50):
        rt._handle_drawing_key(STROKE_WIDTH_DOWN_KEY)
    assert rt._stroke_width == MIN_STROKE_WIDTH
    # Hammer it up past the ceiling -> clamps at MAX.
    for _ in range(50):
        rt._handle_drawing_key(STROKE_WIDTH_UP_KEY)
    assert rt._stroke_width == MAX_STROKE_WIDTH


def test_drawing_key_other_key_is_a_noop():
    rt = _image_runtime()
    rt._stroke = Stroke().append((0.1, 0.1))
    rt._handle_drawing_key(ord("z"))
    assert rt._stroke is not None
    assert rt._stroke_color_index == 0
    assert rt._stroke_width == DEFAULT_STROKE_WIDTH


def test_handle_keys_routes_clear_in_image_mode():
    rt = _image_runtime()
    rt._stroke = Stroke().append((0.1, 0.1)).append((0.2, 0.2))
    assert rt._handle_keys(CLEAR_STROKE_KEY) is True  # not a quit
    assert rt._stroke is None


def test_draw_active_stroke_overlay_marks_the_frame():
    rt = _image_runtime()
    rt._stroke = Stroke().append((0.2, 0.5)).append((0.8, 0.5))
    frame = np.zeros((60, 60, 3), dtype=np.uint8)
    rt._draw_active_stroke(frame)
    assert frame.sum() > 0  # the live polyline + cursor were drawn

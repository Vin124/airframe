"""Unit tests for air-drawing (pinch-to-ink draw mode) + fist-to-remove on the
main loop. Headless -- fake webcam, no sinks/worker/camera/display, synthetic
gesture states only.

Draw model: 'd' toggles draw mode; in it a PINCH is the pen (down while pinching,
up on release), and toggling draw mode off freezes the strokes into one layer.
Out of draw mode a pinch grabs/moves normally (the regression that the old
always-on draw pose caused) and a held fist over a layer removes it.
"""

import numpy as np

from src.drawing import (
    DEFAULT_STROKE_WIDTH,
    MAX_STROKE_WIDTH,
    MIN_STROKE_WIDTH,
    STROKE_PALETTE,
    STROKE_WIDTH_STEP,
    DrawState,
)
from src.gestures import GestureState
from src.interaction import GrabState, InteractionState
from src.layers import fist_removal_target
from src.main import (
    CLEAR_STROKE_KEY,
    CYCLE_COLOR_KEY,
    DRAW_MODE_KEY,
    FIST_REMOVE_FRAMES,
    STROKE_WIDTH_DOWN_KEY,
    STROKE_WIDTH_UP_KEY,
    PipelineRuntime,
    parse_args,
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

    def __init__(self, base_size: Size = (40, 40)) -> None:
        self._base_size = base_size

    def read(self):
        return None

    @property
    def base_size(self) -> Size:
        return self._base_size


class _Tracking:
    """Minimal TrackingResult stand-in: gesture states + (optional) hands."""

    def __init__(self, gesture_states, hands=()):
        self.gesture_states = tuple(gesture_states)
        self.hands = tuple(hands)


def _image_runtime(scene=None, interaction=None) -> PipelineRuntime:
    args = parse_args(["--mode", "image", "--image", "assets/test.png"])
    rt = PipelineRuntime(
        args,
        _FakePacketWebcam(),
        None,
        scene if scene is not None else Scene(),
        interaction if interaction is not None else InteractionState.initial(),
        None,
        None,
    )
    rt._frame_size = FRAME_SIZE  # as if a frame has flowed (for freeze placement)
    return rt


def _pinch(point) -> GestureState:
    return GestureState(is_pinching=True, pinch_point=point, is_grab=False)


def _idle() -> GestureState:
    return GestureState(is_pinching=False, pinch_point=(0.0, 0.0), is_grab=False)


def _fist(point) -> GestureState:
    return GestureState(is_pinching=False, pinch_point=(0.0, 0.0), is_grab=True, grab_point=point)


def _advance(rt, gesture_states) -> None:
    # The real runtime always tracks two hands (slot 0 = primary); pad so
    # interaction.step's two-hand resize read never runs off the end.
    states = list(gesture_states)
    while len(states) < 2:
        states.append(_idle())
    rt._advance_scene(_Tracking(states), FRAME_SIZE)


def _layer_scene() -> Scene:
    """A scene with one 40x40 layer centered at (0.5, 0.5) -> covers ~0.3..0.7."""
    return Scene().add_layer(Layer(source=_FakeSource((40, 40)), transform=Transform(0.5, 0.5)))


# --------------------------------------------------------------------------- #
# Draw mode toggle
# --------------------------------------------------------------------------- #

def test_draw_mode_off_by_default():
    rt = _image_runtime()
    assert rt._draw.mode is False


def test_d_key_toggles_draw_mode():
    rt = _image_runtime()
    rt._handle_drawing_key(DRAW_MODE_KEY)
    assert rt._draw.mode is True
    rt._handle_drawing_key(DRAW_MODE_KEY)
    assert rt._draw.mode is False


# --------------------------------------------------------------------------- #
# Pinch-to-ink: in draw mode a pinch is the pen
# --------------------------------------------------------------------------- #

def test_pinch_inks_while_in_draw_mode():
    rt = _image_runtime()
    rt.toggle_draw_mode()
    _advance(rt, [_pinch((0.3, 0.4))])
    _advance(rt, [_pinch((0.35, 0.45))])
    assert rt._draw.current == ((0.3, 0.4), (0.35, 0.45))


def test_pinch_release_banks_the_stroke():
    rt = _image_runtime()
    rt.toggle_draw_mode()
    _advance(rt, [_pinch((0.3, 0.4))])
    _advance(rt, [_pinch((0.35, 0.45))])
    _advance(rt, [_idle()])  # pen up
    assert rt._draw.current == ()
    assert rt._draw.segments == (((0.3, 0.4), (0.35, 0.45)),)


def test_no_ink_outside_draw_mode():
    # THE regression fix: with draw mode off, a pinch never draws.
    rt = _image_runtime()
    _advance(rt, [_pinch((0.3, 0.4))])
    assert rt._draw.is_empty is True


# --------------------------------------------------------------------------- #
# Freeze on toggling draw mode off
# --------------------------------------------------------------------------- #

def test_toggle_off_freezes_strokes_into_one_layer():
    rt = _image_runtime(scene=Scene())
    rt.toggle_draw_mode()
    for p in [(0.2, 0.6), (0.25, 0.65), (0.3, 0.7)]:
        _advance(rt, [_pinch(p)])
    _advance(rt, [_idle()])  # bank stroke 1
    for p in [(0.6, 0.2), (0.65, 0.25), (0.7, 0.3)]:
        _advance(rt, [_pinch(p)])
    rt.toggle_draw_mode()  # freeze both strokes
    assert rt._draw.mode is False
    assert len(rt.scene.layers) == 1
    assert isinstance(rt.scene.selected_layer.source, ArraySource)
    assert rt._draw.is_empty is True  # buffers cleared after freeze


def test_toggle_off_with_nothing_drawn_adds_no_layer():
    rt = _image_runtime(scene=Scene())
    rt.toggle_draw_mode()
    rt.toggle_draw_mode()  # entered and left without drawing
    assert len(rt.scene.layers) == 0


def test_freeze_carries_pen_color_and_width():
    rt = _image_runtime(scene=Scene())
    rt.toggle_draw_mode()
    rt._handle_drawing_key(CYCLE_COLOR_KEY)  # color index -> 1
    rt._handle_drawing_key(STROKE_WIDTH_UP_KEY)
    for p in [(0.4, 0.4), (0.5, 0.5)]:
        _advance(rt, [_pinch(p)])
    rt.toggle_draw_mode()  # freeze
    assert rt._draw.color_index == 1  # pen settings carried over
    assert rt._draw.width == DEFAULT_STROKE_WIDTH + STROKE_WIDTH_STEP


# --------------------------------------------------------------------------- #
# Draw keys (only act while in draw mode)
# --------------------------------------------------------------------------- #

def test_clear_discards_current_strokes_in_draw_mode():
    rt = _image_runtime()
    rt.toggle_draw_mode()
    _advance(rt, [_pinch((0.3, 0.3))])
    _advance(rt, [_pinch((0.4, 0.4))])
    rt._handle_drawing_key(CLEAR_STROKE_KEY)
    assert rt._draw.is_empty is True


def test_color_and_width_keys_inert_outside_draw_mode():
    rt = _image_runtime()  # NOT in draw mode
    rt._handle_drawing_key(CYCLE_COLOR_KEY)
    rt._handle_drawing_key(STROKE_WIDTH_UP_KEY)
    assert rt._draw.color_index == 0
    assert rt._draw.width == DEFAULT_STROKE_WIDTH


def test_width_keys_clamp_in_draw_mode():
    rt = _image_runtime()
    rt.toggle_draw_mode()
    for _ in range(50):
        rt._handle_drawing_key(STROKE_WIDTH_DOWN_KEY)
    assert rt._draw.width == MIN_STROKE_WIDTH
    for _ in range(50):
        rt._handle_drawing_key(STROKE_WIDTH_UP_KEY)
    assert rt._draw.width == MAX_STROKE_WIDTH


# --------------------------------------------------------------------------- #
# Draw mode suppresses grabbing (pinch owns the pen, not the scene)
# --------------------------------------------------------------------------- #

def test_pinch_in_draw_mode_does_not_grab_a_layer():
    rt = _image_runtime(scene=_layer_scene())
    rt.toggle_draw_mode()
    _advance(rt, [_pinch((0.5, 0.5))])  # over the layer, but drawing
    assert rt.interaction.grab.is_grabbed is False  # pinch inked, didn't grab
    assert rt._draw.current == ((0.5, 0.5),)


def test_pinch_grabs_layer_when_not_drawing():
    # Control: the SAME pinch grabs the layer with draw mode off (regression fixed).
    rt = _image_runtime(scene=_layer_scene())
    _advance(rt, [_pinch((0.5, 0.5))])
    assert rt.interaction.grab.is_grabbed is True


# --------------------------------------------------------------------------- #
# Fist-to-remove gesture
# --------------------------------------------------------------------------- #

def test_fist_removal_target_hits_layer_under_fist():
    scene = _layer_scene()
    assert fist_removal_target(scene, [_fist((0.5, 0.5))], FRAME_SIZE) == 0
    assert fist_removal_target(scene, [_fist((0.05, 0.05))], FRAME_SIZE) is None  # off the layer


def test_fist_removal_target_ignores_a_pinching_hand():
    # A pinch-grab-move (fingers may curl) must not read as a remove fist.
    scene = _layer_scene()
    pinch_and_grab = GestureState(
        is_pinching=True, pinch_point=(0.5, 0.5), is_grab=True, grab_point=(0.5, 0.5)
    )
    assert fist_removal_target(scene, [pinch_and_grab], FRAME_SIZE) is None


def test_held_fist_removes_the_layer_after_threshold():
    rt = _image_runtime(scene=_layer_scene())
    for _ in range(FIST_REMOVE_FRAMES - 1):
        _advance(rt, [_fist((0.5, 0.5))])
    assert len(rt.scene.layers) == 1  # not yet -- still counting the hold
    _advance(rt, [_fist((0.5, 0.5))])  # crosses the threshold
    assert len(rt.scene.layers) == 0  # removed


def test_momentary_fist_does_not_remove():
    rt = _image_runtime(scene=_layer_scene())
    _advance(rt, [_fist((0.5, 0.5))])   # one frame
    _advance(rt, [_idle()])             # fist lifted -> counter resets
    for _ in range(FIST_REMOVE_FRAMES - 1):
        _advance(rt, [_fist((0.5, 0.5))])
    assert len(rt.scene.layers) == 1  # the reset means the threshold isn't reached


def test_fist_over_empty_space_removes_nothing():
    rt = _image_runtime(scene=_layer_scene())
    for _ in range(FIST_REMOVE_FRAMES + 2):
        _advance(rt, [_fist((0.02, 0.02))])  # nowhere near the layer
    assert len(rt.scene.layers) == 1

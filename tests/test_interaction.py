"""Unit tests for src.interaction: the grab/move/resize state machines operating
on the selected scene layer. Pure/headless -- no webcam, synthetic gestures only.
"""

from types import SimpleNamespace

import pytest

from src.interaction import (
    GrabState,
    InteractionState,
    ResizeState,
    step,
    update_grab,
    update_resize,
)
from src.scene import Layer, Transform
from src.sources.base import Size, Source

FRAME_SIZE = (100, 100)  # (width, height)


def _gesture(is_pinching, pinch_point=(0.0, 0.0)):
    """A minimal stand-in for GestureState (only the fields step/grab read)."""
    return SimpleNamespace(is_pinching=is_pinching, pinch_point=pinch_point)


class _FakeSource(Source):
    """A source whose only relevant attribute for interaction is base_size."""

    def __init__(self, base_size: Size) -> None:
        self._base_size = base_size

    def read(self):
        return None

    @property
    def base_size(self) -> Size:
        return self._base_size


def _layer(size=(10, 10), transform=None) -> Layer:
    """A selected layer with a fake ``size`` (w,h) source, centered by default."""
    return Layer(
        source=_FakeSource(size),
        transform=transform if transform is not None else Transform(),
        selected=True,
    )


def _idle():
    return InteractionState.initial()


# --------------------------------------------------------------------------- #
# Grab state machine (via step, single hand; secondary never pinches)
# --------------------------------------------------------------------------- #

def test_pinch_start_on_image_grabs():
    interaction, _layer_out = step(
        _idle(), _layer(), [_gesture(True, (0.5, 0.5)), _gesture(False)], FRAME_SIZE
    )
    assert interaction.grab.is_grabbed is True


def test_pinch_start_off_image_does_not_grab_even_when_dragged_over():
    layer = _layer()
    # Pinch starts off the image.
    i1, layer = step(_idle(), layer, [_gesture(True, (0.9, 0.9)), _gesture(False)], FRAME_SIZE)
    assert i1.grab.is_grabbed is False
    # Still pinching, now dragged over the image: no rising edge -> no grab.
    i2, layer = step(i1, layer, [_gesture(True, (0.5, 0.5)), _gesture(False)], FRAME_SIZE)
    assert i2.grab.is_grabbed is False


def test_move_follows_pinch_keeping_grab_offset():
    layer = _layer()
    # Grab at an off-center point inside the image (rect normalized [0.45,0.55)).
    i1, layer = step(_idle(), layer, [_gesture(True, (0.47, 0.52)), _gesture(False)], FRAME_SIZE)
    assert i1.grab.is_grabbed is True
    # Offset = center(0.5,0.5) - grab point(0.47,0.52) = (0.03, -0.02).
    i2, layer = step(i1, layer, [_gesture(True, (0.60, 0.60)), _gesture(False)], FRAME_SIZE)
    assert layer.transform.center_x == pytest.approx(0.63)
    assert layer.transform.center_y == pytest.approx(0.58)


def test_unpinch_releases():
    layer = _layer()
    i1, layer = step(_idle(), layer, [_gesture(True, (0.5, 0.5)), _gesture(False)], FRAME_SIZE)
    i2, layer = step(i1, layer, [_gesture(False, (0.5, 0.5)), _gesture(False)], FRAME_SIZE)
    assert i2.grab.is_grabbed is False


def test_hand_loss_releases_without_moving_to_origin():
    layer = _layer()
    i1, layer = step(_idle(), layer, [_gesture(True, (0.5, 0.5)), _gesture(False)], FRAME_SIZE)
    # Hand loss is reported by the detector as not-pinching with a (0,0) point.
    i2, layer = step(i1, layer, [_gesture(False, (0.0, 0.0)), _gesture(False)], FRAME_SIZE)
    assert i2.grab.is_grabbed is False
    # Must NOT have snapped the image center to the origin.
    assert (layer.transform.center_x, layer.transform.center_y) == (0.5, 0.5)


def test_update_grab_pinch_start_off_image_consumes_edge():
    grab, transform = update_grab(
        GrabState(), Transform(), (10, 10), True, (0.9, 0.9), FRAME_SIZE
    )
    assert grab.is_grabbed is False
    assert grab.was_pinching is True  # edge consumed -> no later grab-on-drag-over


# --------------------------------------------------------------------------- #
# Two-hand resize
# --------------------------------------------------------------------------- #

def test_resize_scales_by_distance_ratio():
    layer = _layer()
    # First two-hand frame captures the baseline (distance 0.2, scale 1.0).
    i1, layer = step(_idle(), layer, [_gesture(True, (0.4, 0.5)), _gesture(True, (0.6, 0.5))], FRAME_SIZE)
    assert i1.resize.is_resizing is True
    assert layer.transform.scale == pytest.approx(1.0)
    # Hands spread to distance 0.4 -> factor 2.0.
    i2, layer = step(i1, layer, [_gesture(True, (0.3, 0.5)), _gesture(True, (0.7, 0.5))], FRAME_SIZE)
    assert layer.transform.scale == pytest.approx(2.0)
    # Hands close to distance 0.1 -> factor 0.5 (relative to the baseline 0.2).
    i3, layer = step(i2, layer, [_gesture(True, (0.45, 0.5)), _gesture(True, (0.55, 0.5))], FRAME_SIZE)
    assert layer.transform.scale == pytest.approx(0.5)


def test_resize_releases_and_recaptures_fresh_baseline():
    layer = _layer()
    i1, layer = step(_idle(), layer, [_gesture(True, (0.4, 0.5)), _gesture(True, (0.6, 0.5))], FRAME_SIZE)
    i2, layer = step(i1, layer, [_gesture(True, (0.3, 0.5)), _gesture(True, (0.7, 0.5))], FRAME_SIZE)
    assert layer.transform.scale == pytest.approx(2.0)
    # One hand releases -> resize ends, scale holds.
    i3, layer = step(i2, layer, [_gesture(True, (0.3, 0.5)), _gesture(False)], FRAME_SIZE)
    assert i3.resize.is_resizing is False
    assert layer.transform.scale == pytest.approx(2.0)
    # Re-pinch captures a NEW baseline (distance 0.2, scale 2.0)...
    i4, layer = step(i3, layer, [_gesture(True, (0.4, 0.5)), _gesture(True, (0.6, 0.5))], FRAME_SIZE)
    assert layer.transform.scale == pytest.approx(2.0)
    # ...then doubling the distance to 0.4 doubles from that new baseline -> 4.0.
    i5, layer = step(i4, layer, [_gesture(True, (0.3, 0.5)), _gesture(True, (0.7, 0.5))], FRAME_SIZE)
    assert layer.transform.scale == pytest.approx(4.0)


def test_resize_guards_against_degenerate_zero_distance():
    layer = _layer()
    # Both pinch at the same point: baseline distance ~0.
    i1, layer = step(_idle(), layer, [_gesture(True, (0.5, 0.5)), _gesture(True, (0.5, 0.5))], FRAME_SIZE)
    assert i1.resize.is_resizing is True
    # Next frame with a real distance must not divide by zero / must not crash.
    i2, layer = step(i1, layer, [_gesture(True, (0.4, 0.5)), _gesture(True, (0.6, 0.5))], FRAME_SIZE)
    assert layer.transform.scale == pytest.approx(1.0)  # unchanged, guard held


def test_resize_takes_priority_over_grab():
    layer = _layer()
    # Grab first with the primary hand.
    i1, layer = step(_idle(), layer, [_gesture(True, (0.5, 0.5)), _gesture(False)], FRAME_SIZE)
    assert i1.grab.is_grabbed is True
    # Second hand starts pinching too -> resize mode, grab idled.
    i2, layer = step(i1, layer, [_gesture(True, (0.4, 0.5)), _gesture(True, (0.6, 0.5))], FRAME_SIZE)
    assert i2.resize.is_resizing is True
    assert i2.grab.is_grabbed is False


def test_update_resize_not_both_pinching_resets():
    resize, transform = update_resize(
        ResizeState(is_resizing=True, start_distance=0.2, start_scale=1.0),
        Transform(scale=2.0),
        False,
        (0.0, 0.0),
        (0.0, 0.0),
    )
    assert resize.is_resizing is False
    assert transform.scale == 2.0  # unchanged when not resizing


# --------------------------------------------------------------------------- #
# InteractionState shape
# --------------------------------------------------------------------------- #

def test_initial_interaction_is_idle():
    state = InteractionState.initial()
    assert state.grab.is_grabbed is False
    assert state.resize.is_resizing is False


def test_step_returns_updated_layer_not_mutating_input():
    layer = _layer()
    _interaction, out = step(
        _idle(), layer, [_gesture(True, (0.47, 0.52)), _gesture(False)], FRAME_SIZE
    )
    # step returns a distinct layer; the input layer's transform is unchanged.
    assert out is not layer
    assert (layer.transform.center_x, layer.transform.center_y) == (0.5, 0.5)


def test_update_grab_with_no_base_size_never_grabs():
    # A source with no frame yet (base_size None) has no rect to hit; the
    # pinch edge is still consumed so a later drag-over can't grab.
    grab, transform = update_grab(
        GrabState(), Transform(), None, True, (0.5, 0.5), FRAME_SIZE
    )
    assert grab.is_grabbed is False
    assert grab.was_pinching is True

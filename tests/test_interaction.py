"""Unit tests for src.interaction: the grab/move/resize state machines operating
on the selected scene layer. Pure/headless -- no webcam, synthetic gestures only.
"""

from types import SimpleNamespace

import pytest

from src.interaction import (
    GRAB_RELEASE_GRACE_FRAMES,
    GrabState,
    InteractionState,
    ResizeState,
    resolve_selection,
    step,
    update_grab,
    update_resize,
)
from src.scene import Layer, Scene, Transform
from src.sources.base import Size, Source

FRAME_SIZE = (100, 100)  # (width, height)


def _gesture(is_pinching, pinch_point=(0.0, 0.0)):
    """A minimal stand-in for GestureState (only the fields step/grab read)."""
    return SimpleNamespace(is_pinching=is_pinching, pinch_point=pinch_point)


def _release(i, scene, hand=None):
    """Step enough not-pinching frames to outlast the grab release grace.

    A grab is deliberately sticky (rides out brief tracking flickers), so a
    single un-pinch frame no longer releases -- tests that want a clean release
    step past ``GRAB_RELEASE_GRACE_FRAMES`` via this helper. ``hand`` overrides
    the (both-idle) gesture pair, e.g. to simulate hand loss with a (0,0) point.
    """
    gestures = hand if hand is not None else [_gesture(False), _gesture(False)]
    for _ in range(GRAB_RELEASE_GRACE_FRAMES + 1):
        i, scene = step(i, scene, gestures, FRAME_SIZE)
    return i, scene


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


def _scene(layer=None) -> Scene:
    """A single-layer selected scene (step now operates on a Scene)."""
    return Scene.single(layer if layer is not None else _layer())


def _idle():
    return InteractionState.initial()


# --------------------------------------------------------------------------- #
# Grab state machine (via step, single hand; secondary never pinches)
# --------------------------------------------------------------------------- #

def test_pinch_start_on_image_grabs():
    interaction, _scene_out = step(
        _idle(), _scene(), [_gesture(True, (0.5, 0.5)), _gesture(False)], FRAME_SIZE
    )
    assert interaction.grab.is_grabbed is True


def test_secondary_hand_can_grab():
    # Either hand grabs: with ONLY slot 1 pinching over the layer, it grabs -- the
    # fix for "only the primary (yellow-dot) hand moves things".
    interaction, _scene_out = step(
        _idle(), _scene(), [_gesture(False), _gesture(True, (0.5, 0.5))], FRAME_SIZE
    )
    assert interaction.grab.is_grabbed is True


def test_secondary_hand_grab_moves_the_layer():
    scene = _scene(_layer(transform=Transform(0.5, 0.5)))
    # Grab with slot 1, then move it; the layer should follow slot 1's pinch.
    interaction, scene = step(_idle(), scene, [_gesture(False), _gesture(True, (0.5, 0.5))], FRAME_SIZE)
    interaction, scene = step(interaction, scene, [_gesture(False), _gesture(True, (0.7, 0.6))], FRAME_SIZE)
    layer = scene.selected_layer
    assert layer.transform.center_x > 0.5  # moved toward slot 1's new position


def test_pinch_start_off_image_does_not_grab_even_when_dragged_over():
    scene = _scene()
    # Pinch starts off the image.
    i1, scene = step(_idle(), scene, [_gesture(True, (0.9, 0.9)), _gesture(False)], FRAME_SIZE)
    assert i1.grab.is_grabbed is False
    # Still pinching, now dragged over the image: no rising edge -> no grab.
    i2, scene = step(i1, scene, [_gesture(True, (0.5, 0.5)), _gesture(False)], FRAME_SIZE)
    assert i2.grab.is_grabbed is False


def test_move_follows_pinch_keeping_grab_offset():
    scene = _scene()
    # Grab at an off-center point inside the image (rect normalized [0.45,0.55)).
    i1, scene = step(_idle(), scene, [_gesture(True, (0.47, 0.52)), _gesture(False)], FRAME_SIZE)
    assert i1.grab.is_grabbed is True
    # Offset = center(0.5,0.5) - grab point(0.47,0.52) = (0.03, -0.02).
    i2, scene = step(i1, scene, [_gesture(True, (0.60, 0.60)), _gesture(False)], FRAME_SIZE)
    moved = scene.selected_layer.transform
    assert moved.center_x == pytest.approx(0.63)
    assert moved.center_y == pytest.approx(0.58)


def test_unpinch_within_grace_holds_then_releases():
    scene = _scene()
    i1, scene = step(_idle(), scene, [_gesture(True, (0.5, 0.5)), _gesture(False)], FRAME_SIZE)
    # A single un-pinch frame is inside the grace window -> still grabbed (rides
    # out a momentary pinch/tracking flicker instead of dropping the layer).
    i2, scene = step(i1, scene, [_gesture(False, (0.5, 0.5)), _gesture(False)], FRAME_SIZE)
    assert i2.grab.is_grabbed is True
    # Sustained release past the grace window finally lets go.
    i3, scene = _release(i2, scene)
    assert i3.grab.is_grabbed is False


def test_hand_loss_holds_in_grace_then_releases_without_moving_to_origin():
    scene = _scene()
    i1, scene = step(_idle(), scene, [_gesture(True, (0.5, 0.5)), _gesture(False)], FRAME_SIZE)
    # Hand loss is reported by the detector as not-pinching with a (0,0) point.
    # During grace the layer FREEZES -- it must never snap toward the origin.
    i2, scene = step(i1, scene, [_gesture(False, (0.0, 0.0)), _gesture(False)], FRAME_SIZE)
    assert i2.grab.is_grabbed is True
    t = scene.selected_layer.transform
    assert (t.center_x, t.center_y) == (0.5, 0.5)
    # And once grace lapses it releases, still without having moved to origin.
    i3, scene = _release(i2, scene, hand=[_gesture(False, (0.0, 0.0)), _gesture(False)])
    assert i3.grab.is_grabbed is False
    t = scene.selected_layer.transform
    assert (t.center_x, t.center_y) == (0.5, 0.5)


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
    scene = _scene()
    # First two-hand frame captures the baseline (distance 0.2, scale 1.0).
    i1, scene = step(_idle(), scene, [_gesture(True, (0.4, 0.5)), _gesture(True, (0.6, 0.5))], FRAME_SIZE)
    assert i1.resize.is_resizing is True
    assert scene.selected_layer.transform.scale == pytest.approx(1.0)
    # Hands spread to distance 0.4 -> factor 2.0.
    i2, scene = step(i1, scene, [_gesture(True, (0.3, 0.5)), _gesture(True, (0.7, 0.5))], FRAME_SIZE)
    assert scene.selected_layer.transform.scale == pytest.approx(2.0)
    # Hands close to distance 0.1 -> factor 0.5 (relative to the baseline 0.2).
    i3, scene = step(i2, scene, [_gesture(True, (0.45, 0.5)), _gesture(True, (0.55, 0.5))], FRAME_SIZE)
    assert scene.selected_layer.transform.scale == pytest.approx(0.5)


def test_resize_releases_and_recaptures_fresh_baseline():
    scene = _scene()
    i1, scene = step(_idle(), scene, [_gesture(True, (0.4, 0.5)), _gesture(True, (0.6, 0.5))], FRAME_SIZE)
    i2, scene = step(i1, scene, [_gesture(True, (0.3, 0.5)), _gesture(True, (0.7, 0.5))], FRAME_SIZE)
    assert scene.selected_layer.transform.scale == pytest.approx(2.0)
    # One hand releases -> resize ends, scale holds.
    i3, scene = step(i2, scene, [_gesture(True, (0.3, 0.5)), _gesture(False)], FRAME_SIZE)
    assert i3.resize.is_resizing is False
    assert scene.selected_layer.transform.scale == pytest.approx(2.0)
    # Re-pinch captures a NEW baseline (distance 0.2, scale 2.0)...
    i4, scene = step(i3, scene, [_gesture(True, (0.4, 0.5)), _gesture(True, (0.6, 0.5))], FRAME_SIZE)
    assert scene.selected_layer.transform.scale == pytest.approx(2.0)
    # ...then doubling the distance to 0.4 doubles from that new baseline -> 4.0.
    i5, scene = step(i4, scene, [_gesture(True, (0.3, 0.5)), _gesture(True, (0.7, 0.5))], FRAME_SIZE)
    assert scene.selected_layer.transform.scale == pytest.approx(4.0)


def test_resize_guards_against_degenerate_zero_distance():
    scene = _scene()
    # Both pinch at the same point: baseline distance ~0.
    i1, scene = step(_idle(), scene, [_gesture(True, (0.5, 0.5)), _gesture(True, (0.5, 0.5))], FRAME_SIZE)
    assert i1.resize.is_resizing is True
    # Next frame with a real distance must not divide by zero / must not crash.
    i2, scene = step(i1, scene, [_gesture(True, (0.4, 0.5)), _gesture(True, (0.6, 0.5))], FRAME_SIZE)
    assert scene.selected_layer.transform.scale == pytest.approx(1.0)  # unchanged, guard held


def test_resize_takes_priority_over_grab():
    scene = _scene()
    # Grab first with the primary hand.
    i1, scene = step(_idle(), scene, [_gesture(True, (0.5, 0.5)), _gesture(False)], FRAME_SIZE)
    assert i1.grab.is_grabbed is True
    # Second hand starts pinching too -> resize mode, grab idled.
    i2, scene = step(i1, scene, [_gesture(True, (0.4, 0.5)), _gesture(True, (0.6, 0.5))], FRAME_SIZE)
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


def test_step_returns_updated_scene_not_mutating_input():
    scene = _scene()
    _interaction, out = step(
        _idle(), scene, [_gesture(True, (0.47, 0.52)), _gesture(False)], FRAME_SIZE
    )
    # step returns a distinct scene; the input scene's layer is unchanged.
    assert out is not scene
    t = scene.selected_layer.transform
    assert (t.center_x, t.center_y) == (0.5, 0.5)


def test_update_grab_with_no_base_size_never_grabs():
    # A source with no frame yet (base_size None) has no rect to hit; the
    # pinch edge is still consumed so a later drag-over can't grab.
    grab, transform = update_grab(
        GrabState(), Transform(), None, True, (0.5, 0.5), FRAME_SIZE
    )
    assert grab.is_grabbed is False
    assert grab.was_pinching is True


# --------------------------------------------------------------------------- #
# Multi-layer selection (Phase 8): pinch grabs the top-most layer under it,
# selection resolves only on a rising edge and can't be stolen mid-gesture.
# --------------------------------------------------------------------------- #

def _apart_scene() -> Scene:
    """Two non-overlapping layers on the 100x100 frame: A at x=0.35 (id 0,
    selected, back), B at x=0.65 (id 1, front). Each is 10x10 px."""
    a = Layer(
        source=_FakeSource((10, 10)), transform=Transform(0.35, 0.5),
        z_index=0, layer_id=0, selected=True,
    )
    b = Layer(
        source=_FakeSource((10, 10)), transform=Transform(0.65, 0.5),
        z_index=1, layer_id=1, selected=False,
    )
    return Scene(layers=(a, b), next_id=2)


def test_pinch_selects_top_most_overlapping_layer_and_grabs_it():
    # Two layers stacked at the center; the higher-z one must be the one grabbed.
    back = Layer(
        source=_FakeSource((20, 20)), transform=Transform(0.5, 0.5),
        z_index=0, layer_id=0, selected=True,
    )
    front = Layer(
        source=_FakeSource((20, 20)), transform=Transform(0.5, 0.5),
        z_index=1, layer_id=1, selected=False,
    )
    scene = Scene(layers=(back, front), next_id=2)
    i1, scene = step(_idle(), scene, [_gesture(True, (0.5, 0.5)), _gesture(False)], FRAME_SIZE)
    assert i1.grab.is_grabbed is True
    assert scene.selected_layer.layer_id == 1  # top-most (higher z) selected + grabbed


def test_two_layers_are_independently_grabbable():
    scene = _apart_scene()
    # Grab B (over its rect at x=0.65) and drag it to x=0.60.
    i, scene = step(_idle(), scene, [_gesture(True, (0.65, 0.5)), _gesture(False)], FRAME_SIZE)
    assert scene.selected_layer.layer_id == 1
    i, scene = step(i, scene, [_gesture(True, (0.60, 0.5)), _gesture(False)], FRAME_SIZE)
    assert scene.layer_by_id(1).transform.center_x == pytest.approx(0.60)
    assert scene.layer_by_id(0).transform.center_x == pytest.approx(0.35)  # A untouched
    # Release (past the grace window so B truly lets go), then grab A (over its
    # rect at x=0.35) and drag it to x=0.40.
    i, scene = _release(i, scene)
    i, scene = step(i, scene, [_gesture(True, (0.35, 0.5)), _gesture(False)], FRAME_SIZE)
    assert scene.selected_layer.layer_id == 0
    i, scene = step(i, scene, [_gesture(True, (0.40, 0.5)), _gesture(False)], FRAME_SIZE)
    assert scene.layer_by_id(0).transform.center_x == pytest.approx(0.40)
    assert scene.layer_by_id(1).transform.center_x == pytest.approx(0.60)  # B stayed put


def test_selection_does_not_change_mid_drag():
    scene = _apart_scene()
    # Grab A at its center; offset is zero.
    i, scene = step(_idle(), scene, [_gesture(True, (0.35, 0.5)), _gesture(False)], FRAME_SIZE)
    assert scene.selected_layer.layer_id == 0
    # Still pinching, drag the hand over B's rect (x=0.62): no rising edge, so
    # selection stays A -- A follows the hand, B is untouched.
    i, scene = step(i, scene, [_gesture(True, (0.62, 0.5)), _gesture(False)], FRAME_SIZE)
    assert scene.selected_layer.layer_id == 0
    assert scene.layer_by_id(0).transform.center_x == pytest.approx(0.62)
    assert scene.layer_by_id(1).transform.center_x == pytest.approx(0.65)  # B unchanged


def test_selection_not_stolen_during_resize():
    scene = _apart_scene()
    # Two-hand pinch: primary over A, secondary over B. Selection resolves from
    # the primary's rising edge (A), and can't change while the resize is live.
    i, scene = step(_idle(), scene, [_gesture(True, (0.35, 0.5)), _gesture(True, (0.65, 0.5))], FRAME_SIZE)
    assert i.resize.is_resizing is True
    assert scene.selected_layer.layer_id == 0
    i, scene = step(i, scene, [_gesture(True, (0.30, 0.5)), _gesture(True, (0.70, 0.5))], FRAME_SIZE)
    assert scene.selected_layer.layer_id == 0


def test_grab_uses_offset_of_newly_selected_layer():
    scene = _apart_scene()  # A selected, but the pinch lands on B
    # Pinch at x=0.63, inside B's rect [0.60,0.70): B becomes selected + grabbed.
    i, scene = step(_idle(), scene, [_gesture(True, (0.63, 0.5)), _gesture(False)], FRAME_SIZE)
    assert scene.selected_layer.layer_id == 1
    assert i.grab.is_grabbed is True
    # offset = B.center(0.65) - pinch(0.63) = 0.02; drag to 0.50 -> center 0.52.
    i, scene = step(i, scene, [_gesture(True, (0.50, 0.5)), _gesture(False)], FRAME_SIZE)
    assert scene.layer_by_id(1).transform.center_x == pytest.approx(0.52)
    assert scene.layer_by_id(1).transform.center_y == pytest.approx(0.50)


def test_step_tolerates_empty_scene():
    # Removing the last layer at runtime leaves an empty, selection-less scene.
    i, out = step(_idle(), Scene(), [_gesture(True, (0.5, 0.5)), _gesture(False)], FRAME_SIZE)
    assert out.selected_layer is None
    assert i.grab.is_grabbed is False
    # Edge memory still advanced so a later pinch (over a re-added layer) is clean.
    assert i.grab.was_pinching is True


def test_resolve_selection_off_background_keeps_selection():
    scene = _apart_scene()
    # Rising-edge pinch over bare background (0.05,0.05): selection unchanged.
    out = resolve_selection(_idle(), scene, _gesture(True, (0.05, 0.05)), FRAME_SIZE)
    assert out.selected_layer.layer_id == 0


def test_resolve_selection_only_on_rising_edge():
    scene = _apart_scene()
    # was_pinching already True -> a continuing pinch, not an edge -> no change,
    # even though the pinch point is squarely over B.
    state = InteractionState(grab=GrabState(was_pinching=True))
    out = resolve_selection(state, scene, _gesture(True, (0.65, 0.5)), FRAME_SIZE)
    assert out.selected_layer.layer_id == 0


def test_resize_end_does_not_reselect_or_grab():
    scene = _apart_scene()
    # Two-hand resize starts (primary over A, id 0).
    i, scene = step(
        _idle(), scene, [_gesture(True, (0.35, 0.5)), _gesture(True, (0.65, 0.5))], FRAME_SIZE
    )
    assert i.resize.is_resizing is True
    # Secondary releases; primary still pinching, now hovering over B's rect.
    # No rising edge -> the resize ends without reselecting or grabbing.
    i, scene = step(i, scene, [_gesture(True, (0.63, 0.5)), _gesture(False)], FRAME_SIZE)
    assert i.resize.is_resizing is False
    assert scene.selected_layer.layer_id == 0
    assert i.grab.is_grabbed is False
    # Still held the following frame: same guarantees (no mid-hold selection).
    i, scene = step(i, scene, [_gesture(True, (0.63, 0.5)), _gesture(False)], FRAME_SIZE)
    assert scene.selected_layer.layer_id == 0
    assert i.grab.is_grabbed is False


def test_background_pinch_held_onto_layer_does_not_select_or_grab():
    scene = _apart_scene()  # A (id 0) selected
    # Pinch starts over bare background: selection kept, edge consumed.
    i, scene = step(_idle(), scene, [_gesture(True, (0.9, 0.9)), _gesture(False)], FRAME_SIZE)
    assert scene.selected_layer.layer_id == 0
    assert i.grab.is_grabbed is False
    # Held and dragged over B's rect: no rising edge -> still no selection
    # change, no grab, and B doesn't move.
    i, scene = step(i, scene, [_gesture(True, (0.63, 0.5)), _gesture(False)], FRAME_SIZE)
    assert scene.selected_layer.layer_id == 0
    assert i.grab.is_grabbed is False
    assert scene.layer_by_id(1).transform.center_x == pytest.approx(0.65)

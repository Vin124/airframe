"""Unit tests for src.main: argument parsing/validation and pure key-handling.

Does not exercise run() (the camera loop), so no webcam or display is needed.
Importing src.main does import cv2 at module level, which is fine (no
webcam/display access happens until run() is called).
"""

import pytest

from src.interaction import GrabState, InteractionState, ResizeState
from src.main import (
    MODE_IMAGE,
    MODE_SKELETON,
    RESET_TRANSFORM_KEY,
    SCALE_DOWN_KEYS,
    SCALE_UP_KEYS,
    TOGGLE_SKELETON_KEY,
    _handle_image_key,
    _toggle_skeleton,
    parse_args,
)
from src.scene import Layer, Scene, Transform
from src.sources.base import Size, Source


class _FakeSource(Source):
    """Minimal Source for building a selected image layer in key-handling tests."""

    def __init__(self, base_size: Size = (10, 10)) -> None:
        self._base_size = base_size

    def read(self):
        return None

    @property
    def base_size(self) -> Size:
        return self._base_size


def _scene(transform=None) -> Scene:
    """A single-layer selected scene, centered by default."""
    return Scene.single(
        Layer(source=_FakeSource(), transform=transform if transform is not None else Transform())
    )


def test_skeleton_mode_does_not_require_image():
    args = parse_args(["--mode", "skeleton"])
    assert args.mode == MODE_SKELETON
    assert args.image is None


def test_image_mode_requires_image_argument():
    with pytest.raises(SystemExit):
        parse_args(["--mode", "image"])


def test_image_mode_with_image_argument():
    args = parse_args(["--mode", "image", "--image", "assets/test.png"])
    assert args.mode == MODE_IMAGE
    assert args.image == "assets/test.png"


def test_missing_mode_is_rejected():
    with pytest.raises(SystemExit):
        parse_args([])


def test_invalid_mode_is_rejected():
    with pytest.raises(SystemExit):
        parse_args(["--mode", "bogus"])


# --------------------------------------------------------------------------- #
# Runtime key handling (pure state transitions, no camera/display needed)
# --------------------------------------------------------------------------- #

def test_toggle_skeleton_key_flips_true_to_false():
    assert _toggle_skeleton(TOGGLE_SKELETON_KEY, True) is False


def test_toggle_skeleton_key_flips_false_to_true():
    assert _toggle_skeleton(TOGGLE_SKELETON_KEY, False) is True


def test_toggle_skeleton_other_key_leaves_state_unchanged():
    assert _toggle_skeleton(ord("x"), True) is True
    assert _toggle_skeleton(ord("x"), False) is False


def test_handle_image_key_scale_up():
    scene = _scene()
    before = scene.selected_layer.transform.scale
    result_scene, _interaction = _handle_image_key(SCALE_UP_KEYS[0], scene, InteractionState.initial())
    assert result_scene.selected_layer.transform.scale > before


def test_handle_image_key_scale_down():
    scene = _scene()
    before = scene.selected_layer.transform.scale
    result_scene, _interaction = _handle_image_key(SCALE_DOWN_KEYS[0], scene, InteractionState.initial())
    assert result_scene.selected_layer.transform.scale < before


def test_handle_image_key_other_key_is_a_noop():
    scene = _scene()
    interaction = InteractionState.initial()
    result_scene, result_interaction = _handle_image_key(ord("x"), scene, interaction)
    assert result_scene is scene
    assert result_interaction is interaction


def test_handle_image_key_reset_recenters_transform():
    scene = _scene(Transform(center_x=0.1, center_y=0.9, scale=3.0))
    result_scene, _interaction = _handle_image_key(RESET_TRANSFORM_KEY, scene, InteractionState.initial())
    t = result_scene.selected_layer.transform
    assert (t.center_x, t.center_y, t.scale) == (0.5, 0.5, 1.0)


def test_handle_image_key_reset_also_idles_grab_and_resize():
    scene = _scene(Transform(center_x=0.2, center_y=0.2, scale=2.0))
    grabbed = InteractionState(
        grab=GrabState(is_grabbed=True, offset=(0.1, 0.1), was_pinching=True),
        resize=ResizeState(is_resizing=True, start_distance=0.3, start_scale=2.0),
    )
    result_scene, result_interaction = _handle_image_key(RESET_TRANSFORM_KEY, scene, grabbed)
    assert result_interaction.grab.is_grabbed is False
    assert result_interaction.resize.is_resizing is False
    # Transform is recentered on the selected layer too.
    t = result_scene.selected_layer.transform
    assert (t.center_x, t.center_y, t.scale) == (0.5, 0.5, 1.0)

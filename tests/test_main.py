"""Unit tests for src.main: argument parsing/validation and pure key-handling.

Does not exercise run() (the camera loop), so no webcam or display is needed.
Importing src.main does import cv2 at module level, which is fine (no
webcam/display access happens until run() is called).
"""

import pytest

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
from src.overlay import GrabState, InteractionState, ResizeState, Transform


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
    state = InteractionState.initial()
    result = _handle_image_key(SCALE_UP_KEYS[0], state)
    assert result.transform.scale > state.transform.scale


def test_handle_image_key_scale_down():
    state = InteractionState.initial()
    result = _handle_image_key(SCALE_DOWN_KEYS[0], state)
    assert result.transform.scale < state.transform.scale


def test_handle_image_key_other_key_is_a_noop():
    state = InteractionState.initial()
    result = _handle_image_key(ord("x"), state)
    assert result == state


def test_handle_image_key_reset_recenters_transform():
    moved = InteractionState.initial().with_transform(Transform(center_x=0.1, center_y=0.9, scale=3.0))
    result = _handle_image_key(RESET_TRANSFORM_KEY, moved)
    assert (result.transform.center_x, result.transform.center_y, result.transform.scale) == (0.5, 0.5, 1.0)


def test_handle_image_key_reset_also_idles_grab_and_resize():
    grabbed = InteractionState(
        transform=Transform(center_x=0.2, center_y=0.2, scale=2.0),
        grab=GrabState(is_grabbed=True, offset=(0.1, 0.1), was_pinching=True),
        resize=ResizeState(is_resizing=True, start_distance=0.3, start_scale=2.0),
    )
    result = _handle_image_key(RESET_TRANSFORM_KEY, grabbed)
    assert result.grab.is_grabbed is False
    assert result.resize.is_resizing is False

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
    TOGGLE_VIRTUALCAM_KEY,
    _handle_image_key,
    _pump_virtualcam,
    _toggle_skeleton,
    _toggle_virtualcam,
    parse_args,
)
from src.scene import Layer, Scene, Transform
from src.sinks import SinkError
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


def test_virtualcam_flag_defaults_false():
    args = parse_args(["--mode", "skeleton"])
    assert args.virtualcam is False


def test_virtualcam_flag_can_be_enabled():
    args = parse_args(["--mode", "skeleton", "--virtualcam"])
    assert args.virtualcam is True


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


def test_toggle_virtualcam_key_flips_true_to_false():
    assert _toggle_virtualcam(TOGGLE_VIRTUALCAM_KEY, True) is False


def test_toggle_virtualcam_key_flips_false_to_true():
    assert _toggle_virtualcam(TOGGLE_VIRTUALCAM_KEY, False) is True


def test_toggle_virtualcam_other_key_leaves_state_unchanged():
    assert _toggle_virtualcam(ord("x"), True) is True
    assert _toggle_virtualcam(ord("x"), False) is False


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


# --------------------------------------------------------------------------- #
# _pump_virtualcam: fatal vs graceful SinkError handling
# --------------------------------------------------------------------------- #

class _FakeVirtualCamSink:
    """Records sends; raises SinkError when told to fail."""

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.sent = 0

    def send(self, frame) -> None:
        if self.fail:
            raise SinkError("backend gone")
        self.sent += 1


def test_pump_virtualcam_inactive_sends_nothing():
    sink = _FakeVirtualCamSink()
    active, ever_sent = _pump_virtualcam(sink, "frame", False, False, False)
    assert (active, ever_sent) == (False, False)
    assert sink.sent == 0


def test_pump_virtualcam_success_marks_ever_sent():
    sink = _FakeVirtualCamSink()
    active, ever_sent = _pump_virtualcam(sink, "frame", True, False, True)
    assert (active, ever_sent) == (True, True)
    assert sink.sent == 1


def test_pump_virtualcam_launch_failure_is_fatal():
    # --virtualcam requested, never worked -> fail fast like _init_mode does.
    sink = _FakeVirtualCamSink(fail=True)
    with pytest.raises(SystemExit) as excinfo:
        _pump_virtualcam(sink, "frame", True, False, True)
    assert excinfo.value.code == 1


def test_pump_virtualcam_failure_after_success_degrades_gracefully():
    # Worked once (OBS closed mid-session): warn + toggle off, session lives.
    sink = _FakeVirtualCamSink(fail=True)
    active, ever_sent = _pump_virtualcam(sink, "frame", True, True, True)
    assert (active, ever_sent) == (False, True)


def test_pump_virtualcam_runtime_toggle_failure_is_not_fatal():
    # 'v' toggled at runtime without --virtualcam: never fatal.
    sink = _FakeVirtualCamSink(fail=True)
    active, ever_sent = _pump_virtualcam(sink, "frame", True, False, False)
    assert (active, ever_sent) == (False, False)

"""Unit tests for src.main: argument parsing/validation and pure key-handling.

Does not exercise run() (the camera loop), so no webcam or display is needed.
Importing src.main does import cv2 at module level, which is fine (no
webcam/display access happens until run() is called).
"""

import numpy as np
import pytest

import src.main
from src.interaction import GrabState, InteractionState, ResizeState
from src.main import (
    ADD_IMAGE_KEY,
    ADD_SCREEN_KEY,
    ADD_WINDOW_KEY,
    BRING_TO_FRONT_KEY,
    INITIAL_FIT_FRACTION,
    MODE_IMAGE,
    MODE_SKELETON,
    PipelineRuntime,
    REMOVE_LAYER_KEY,
    RESET_TRANSFORM_KEY,
    SCALE_DOWN_KEYS,
    SCALE_UP_KEYS,
    SOURCE_IMAGE,
    SOURCE_SCREEN,
    SOURCE_WINDOW,
    TOGGLE_SKELETON_KEY,
    TOGGLE_VIRTUALCAM_KEY,
    _apply_pending_fit,
    _build_layer_source,
    _fit_layer_to_frame,
    _handle_image_key,
    _handle_layer_keys,
    _init_mode,
    _initial_fit_ids,
    _pump_virtualcam,
    _toggle_skeleton,
    _toggle_virtualcam,
    parse_args,
)
from src.scene import Layer, Scene, Transform
from src.sinks import SinkError
from src.sources import ImageSource, SourceError
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
# --gpu flag (Phase 9): parsing + plumbing into the tracker
# --------------------------------------------------------------------------- #

def test_gpu_flag_defaults_false():
    assert parse_args(["--mode", "skeleton"]).gpu is False


def test_gpu_flag_can_be_enabled():
    assert parse_args(["--mode", "skeleton", "--gpu"]).gpu is True


def test_init_mode_passes_gpu_flag_to_tracker(monkeypatch):
    created = {}

    class _FakeTracker:
        def __init__(self, num_hands=1, gpu=False):
            created["num_hands"] = num_hands
            created["gpu"] = gpu

    monkeypatch.setattr(src.main, "HandTracker", _FakeTracker)
    _init_mode(parse_args(["--mode", "skeleton", "--gpu"]))
    assert created["gpu"] is True


def test_init_mode_defaults_gpu_false(monkeypatch):
    created = {}

    class _FakeTracker:
        def __init__(self, num_hands=1, gpu=False):
            created["gpu"] = gpu

    monkeypatch.setattr(src.main, "HandTracker", _FakeTracker)
    _init_mode(parse_args(["--mode", "skeleton"]))
    assert created["gpu"] is False


# --------------------------------------------------------------------------- #
# --source / --window-title / --monitor-index parsing + validation
# --------------------------------------------------------------------------- #

def test_source_defaults_to_image():
    args = parse_args(["--mode", "image", "--image", "assets/test.png"])
    assert args.source == SOURCE_IMAGE
    assert args.monitor_index == 1  # 1-based default (windows-capture rejects 0)


def test_source_window_parses_with_title():
    args = parse_args(["--mode", "image", "--source", "window", "--window-title", "Notepad"])
    assert args.source == SOURCE_WINDOW
    assert args.window_title == "Notepad"


def test_source_window_without_title_is_rejected():
    with pytest.raises(SystemExit):
        parse_args(["--mode", "image", "--source", "window"])


def test_source_window_does_not_require_image():
    # --image is only required for --source image; a window source stands in for it.
    args = parse_args(["--mode", "image", "--source", "window", "--window-title", "X"])
    assert args.image is None


def test_source_screen_parses_monitor_index():
    args = parse_args(["--mode", "image", "--source", "screen", "--monitor-index", "2"])
    assert args.source == SOURCE_SCREEN
    assert args.monitor_index == 2


def test_source_screen_does_not_require_image():
    args = parse_args(["--mode", "image", "--source", "screen"])
    assert args.image is None
    assert args.monitor_index == 1


def test_live_source_rejected_outside_image_mode():
    with pytest.raises(SystemExit):
        parse_args(["--mode", "skeleton", "--source", "window", "--window-title", "X"])


def test_image_source_still_requires_image():
    with pytest.raises(SystemExit):
        parse_args(["--mode", "image", "--source", "image"])


# --------------------------------------------------------------------------- #
# Initial fit of a live (large) layer to ~half the frame width
# --------------------------------------------------------------------------- #

def test_fit_layer_scales_to_half_frame_width():
    # A 1600px-wide source in a 640px frame -> width ~= 320px -> scale 0.2.
    layer = Layer(source=_FakeSource(base_size=(1600, 900)))
    fitted = _fit_layer_to_frame(layer, (640, 480))
    assert fitted is not None
    expected = (640 * INITIAL_FIT_FRACTION) / 1600
    assert fitted.transform.scale == pytest.approx(expected)


def test_fit_layer_returns_none_when_size_unknown():
    layer = Layer(source=_FakeSource(base_size=None))
    assert _fit_layer_to_frame(layer, (640, 480)) is None


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


# --------------------------------------------------------------------------- #
# _apply_pending_fit: exactly-once fit semantics (loop-extracted helper)
# --------------------------------------------------------------------------- #

def test_apply_pending_fit_not_pending_is_noop():
    scene = _scene()
    result, pending = _apply_pending_fit(scene, set(), (640, 480))
    assert result is scene
    assert pending == set()


def test_apply_pending_fit_stays_pending_until_size_known():
    scene = Scene.single(Layer(source=_FakeSource(base_size=None)))  # id 0
    result, pending = _apply_pending_fit(scene, {0}, (640, 480))
    assert result is scene
    assert pending == {0}


def test_apply_pending_fit_applies_once_then_clears_flag():
    scene = Scene.single(Layer(source=_FakeSource(base_size=(1600, 900))))  # id 0
    fitted_scene, pending = _apply_pending_fit(scene, {0}, (640, 480))
    assert pending == set()
    expected = (640 * INITIAL_FIT_FRACTION) / 1600
    assert fitted_scene.layer_by_id(0).transform.scale == pytest.approx(expected)
    # A second call with the cleared set must not re-fit (e.g. after resize).
    resized = fitted_scene.update_layer(
        0, fitted_scene.layer_by_id(0).with_transform(Transform(scale=3.0))
    )
    again, pending = _apply_pending_fit(resized, set(), (640, 480))
    assert again is resized
    assert again.layer_by_id(0).transform.scale == pytest.approx(3.0)


def test_apply_pending_fit_only_fits_pending_ids():
    # Two live layers, but only id 1 is pending -> id 0 keeps its scale.
    scene = (
        Scene()
        .add_layer(Layer(source=_FakeSource(base_size=(1000, 500))))  # id 0
        .add_layer(Layer(source=_FakeSource(base_size=(1600, 900))))  # id 1
    )
    result, pending = _apply_pending_fit(scene, {1}, (640, 480))
    assert pending == set()
    assert result.layer_by_id(0).transform.scale == pytest.approx(1.0)  # untouched
    expected = (640 * INITIAL_FIT_FRACTION) / 1600
    assert result.layer_by_id(1).transform.scale == pytest.approx(expected)


def test_apply_pending_fit_drops_id_of_removed_layer():
    # A pending id whose layer is no longer in the scene (removed before its
    # first frame) is dropped, not carried forever.
    scene = Scene.single(Layer(source=_FakeSource(base_size=(1600, 900))))  # id 0
    result, pending = _apply_pending_fit(scene, {99}, (640, 480))
    assert result is scene
    assert pending == set()


# --------------------------------------------------------------------------- #
# _build_layer_source dispatch (fakes patched in, no real capture started)
# --------------------------------------------------------------------------- #

def test_build_layer_source_image_builds_image_source():
    args = parse_args(["--mode", "image", "--image", "assets/test.png"])
    source = _build_layer_source(args)
    assert isinstance(source, ImageSource)


def test_build_layer_source_window_dispatch(monkeypatch):
    created = {}

    class _FakeWindowSource:
        def __init__(self, title):
            created["title"] = title

        def open(self):
            created["opened"] = True
            return self

    monkeypatch.setattr(src.main, "WindowSource", _FakeWindowSource)
    args = parse_args(["--mode", "image", "--source", "window", "--window-title", "Pad"])
    source = _build_layer_source(args)
    assert isinstance(source, _FakeWindowSource)
    assert created == {"title": "Pad", "opened": True}


def test_build_layer_source_screen_dispatch(monkeypatch):
    created = {}

    class _FakeMonitorSource:
        def __init__(self, index):
            created["index"] = index

        def open(self):
            created["opened"] = True
            return self

    monkeypatch.setattr(src.main, "MonitorSource", _FakeMonitorSource)
    args = parse_args(["--mode", "image", "--source", "screen", "--monitor-index", "2"])
    source = _build_layer_source(args)
    assert isinstance(source, _FakeMonitorSource)
    assert created == {"index": 2, "opened": True}


# --------------------------------------------------------------------------- #
# _initial_fit_ids: which launch layers need the one-time fit
# --------------------------------------------------------------------------- #

def test_initial_fit_ids_empty_for_static_image():
    args = parse_args(["--mode", "image", "--image", "assets/test.png"])
    assert _initial_fit_ids(args, _scene()) == set()


def test_initial_fit_ids_empty_in_skeleton_mode():
    args = parse_args(["--mode", "skeleton"])
    assert _initial_fit_ids(args, Scene()) == set()


def test_initial_fit_ids_has_launch_layer_for_live_source():
    args = parse_args(["--mode", "image", "--source", "screen", "--monitor-index", "1"])
    scene = Scene().add_layer(Layer(source=_FakeSource(base_size=None)))  # id 0
    assert _initial_fit_ids(args, scene) == {0}


# --------------------------------------------------------------------------- #
# _handle_layer_keys: add (a/w/m), remove (x), bring-to-front (f)
# --------------------------------------------------------------------------- #

class _ClosableSource(Source):
    """A layer source that records whether close() was called (remove tests)."""

    def __init__(self, base_size: Size = (10, 10)) -> None:
        self._base_size = base_size
        self.closed = False

    def read(self):
        return None

    @property
    def base_size(self):
        return self._base_size

    def close(self) -> None:
        self.closed = True


class _FakeLiveSource:
    """A duck-typed live layer source (window/screen) for add-layer tests.

    The builders construct it positionally (``WindowSource(title)`` /
    ``MonitorSource(index)``), so it records the positional args it was built
    with.
    """

    def __init__(self, *args) -> None:
        self.args = args

    def open(self) -> "_FakeLiveSource":
        return self

    def read(self):
        return None

    @property
    def base_size(self):
        return None  # live sources report no size until their first frame

    def close(self) -> None:
        pass


def test_handle_layer_keys_add_image_adds_selected_top_layer():
    args = parse_args(["--mode", "image", "--image", "assets/test.png"])
    scene = _scene()  # one layer, id 0, selected
    new_scene, interaction, fit_ids = _handle_layer_keys(
        ADD_IMAGE_KEY, args, scene, InteractionState.initial(), set()
    )
    assert len(new_scene.layers) == 2
    added = new_scene.selected_layer
    assert added.layer_id == 1  # new layer selected
    assert added.z_index == 1  # lands on top
    assert isinstance(added.source, ImageSource)
    assert fit_ids == set()  # image source is not live -> no pending fit


def test_handle_layer_keys_add_image_without_path_is_noop(capsys):
    args = parse_args(["--mode", "image", "--source", "screen"])  # no --image
    scene = _scene()
    new_scene, _interaction, fit_ids = _handle_layer_keys(
        ADD_IMAGE_KEY, args, scene, InteractionState.initial(), set()
    )
    assert new_scene is scene  # unchanged
    assert fit_ids == set()
    assert "image" in capsys.readouterr().err.lower()


def test_handle_layer_keys_add_window_marks_live_fit(monkeypatch):
    monkeypatch.setattr(src.main, "WindowSource", _FakeLiveSource)
    args = parse_args(["--mode", "image", "--source", "window", "--window-title", "Pad"])
    scene = _scene()
    new_scene, _interaction, fit_ids = _handle_layer_keys(
        ADD_WINDOW_KEY, args, scene, InteractionState.initial(), set()
    )
    assert len(new_scene.layers) == 2
    new_id = new_scene.selected_layer.layer_id
    assert fit_ids == {new_id}  # live source -> pending fit tracked by id
    assert isinstance(new_scene.selected_layer.source, _FakeLiveSource)


def test_handle_layer_keys_add_window_without_title_is_noop(capsys):
    # --source image at launch means no --window-title was supplied.
    args = parse_args(["--mode", "image", "--image", "assets/test.png"])
    scene = _scene()
    new_scene, _interaction, fit_ids = _handle_layer_keys(
        ADD_WINDOW_KEY, args, scene, InteractionState.initial(), set()
    )
    assert new_scene is scene
    assert "window" in capsys.readouterr().err.lower()


def test_handle_layer_keys_add_window_failure_keeps_running(monkeypatch, capsys):
    class _FailingWindowSource:
        def __init__(self, title):
            pass

        def open(self):
            raise SourceError("Could not capture a window matching title 'X'.")

    monkeypatch.setattr(src.main, "WindowSource", _FailingWindowSource)
    args = parse_args(["--mode", "image", "--source", "window", "--window-title", "X"])
    scene = _scene()
    new_scene, _interaction, fit_ids = _handle_layer_keys(
        ADD_WINDOW_KEY, args, scene, InteractionState.initial(), set()
    )
    assert new_scene is scene  # add failed -> scene untouched, no exit
    assert "window" in capsys.readouterr().err.lower()


def test_handle_layer_keys_add_screen_dispatch(monkeypatch):
    monkeypatch.setattr(src.main, "MonitorSource", _FakeLiveSource)
    args = parse_args(["--mode", "image", "--source", "screen", "--monitor-index", "2"])
    scene = _scene()
    new_scene, _interaction, fit_ids = _handle_layer_keys(
        ADD_SCREEN_KEY, args, scene, InteractionState.initial(), set()
    )
    added = new_scene.selected_layer
    assert isinstance(added.source, _FakeLiveSource)
    assert added.source.args == (2,)  # built with --monitor-index
    assert fit_ids == {added.layer_id}


def test_handle_layer_keys_add_resets_interaction():
    # Adding a layer while a grab is in progress must not leak the grab onto the
    # freshly-selected new layer.
    args = parse_args(["--mode", "image", "--image", "assets/test.png"])
    grabbed = InteractionState(
        grab=GrabState(is_grabbed=True, offset=(0.1, 0.1), was_pinching=True)
    )
    _scene_out, interaction, _fit = _handle_layer_keys(
        ADD_IMAGE_KEY, args, _scene(), grabbed, set()
    )
    assert interaction.grab.is_grabbed is False


def _layer_with(source, selected=False):
    return Layer(source=source, selected=selected)


def test_handle_layer_keys_remove_closes_source_and_selects_new_top():
    src0, src1 = _ClosableSource(), _ClosableSource()
    scene = Scene().add_layer(_layer_with(src0)).add_layer(_layer_with(src1))
    grabbed = InteractionState(
        grab=GrabState(is_grabbed=True, offset=(0.1, 0.1), was_pinching=True)
    )
    new_scene, interaction, fit_ids = _handle_layer_keys(
        REMOVE_LAYER_KEY, None, scene, grabbed, {1}
    )
    assert src1.closed is True  # removed (selected) layer's source is released
    assert src0.closed is False  # kept layer untouched
    assert new_scene.selected_layer.layer_id == 0  # new top-most selected
    assert interaction.grab.is_grabbed is False  # mid-grab reset
    assert fit_ids == set()  # freed id dropped from pending


def test_handle_layer_keys_remove_last_layer_empties_scene():
    src0 = _ClosableSource()
    scene = Scene().add_layer(_layer_with(src0))
    new_scene, _interaction, _fit = _handle_layer_keys(
        REMOVE_LAYER_KEY, None, scene, InteractionState.initial(), set()
    )
    assert new_scene.layers == ()
    assert new_scene.selected_layer is None
    assert src0.closed is True


def test_handle_layer_keys_remove_noop_when_nothing_selected():
    src0 = _ClosableSource()
    scene = Scene(layers=(_layer_with(src0, selected=False),))
    new_scene, _interaction, _fit = _handle_layer_keys(
        REMOVE_LAYER_KEY, None, scene, InteractionState.initial(), set()
    )
    assert new_scene is scene
    assert src0.closed is False


def test_handle_layer_keys_bring_to_front_raises_z():
    scene = Scene().add_layer(_layer_with(_FakeSource())).add_layer(_layer_with(_FakeSource()))
    scene = scene.select_layer(0)  # select the back layer (id 0, z 0)
    new_scene, _interaction, _fit = _handle_layer_keys(
        BRING_TO_FRONT_KEY, None, scene, InteractionState.initial(), set()
    )
    assert new_scene.layer_by_id(0).z_index == 2  # max(0, 1) + 1
    assert new_scene.layers_by_z()[-1].layer_id == 0  # now front-most


def test_handle_layer_keys_other_key_is_noop():
    scene = _scene()
    interaction = InteractionState.initial()
    new_scene, new_interaction, fit_ids = _handle_layer_keys(
        ord("z"), None, scene, interaction, {5}
    )
    assert new_scene is scene
    assert new_interaction is interaction
    assert fit_ids == {5}


def test_fit_layer_preserves_manual_prescale():
    # Two '+' presses (scale ~1.21) made before the first captured frame must
    # survive the fit: it multiplies rather than overwrites the manual scale.
    layer = Layer(source=_FakeSource(base_size=(1600, 900)), transform=Transform(scale=1.21))
    fitted = _fit_layer_to_frame(layer, (640, 480))
    expected = ((640 * INITIAL_FIT_FRACTION) / 1600) * 1.21
    assert fitted.transform.scale == pytest.approx(expected)


# --------------------------------------------------------------------------- #
# PipelineRuntime._next_frame: copy-every-tick invariant (thread-safety seam)
# --------------------------------------------------------------------------- #

class _FakePacketWebcam:
    """Minimal ThreadedSource stand-in: hand-set (seq, frame) packets."""

    def __init__(self):
        self.packet = None

    def latest_packet(self):
        return self.packet


def _runtime(webcam) -> PipelineRuntime:
    # Worker/sinks are None: _next_frame touches only the webcam + clean-frame
    # state, and __init__ merely stores the collaborators.
    args = parse_args(["--mode", "skeleton"])
    return PipelineRuntime(args, webcam, None, Scene(), None, None, None)


def test_next_frame_none_before_first_packet():
    assert _runtime(_FakePacketWebcam())._next_frame() is None


def test_next_frame_returns_fresh_copy_every_tick():
    webcam = _FakePacketWebcam()
    base = np.zeros((4, 4, 3), dtype=np.uint8)
    webcam.packet = (1, base)
    runtime = _runtime(webcam)
    first = runtime._next_frame()
    second = runtime._next_frame()  # same seq -> same clean frame, NEW copy
    assert first is not base and second is not base
    assert first is not second
    # Mutating a returned copy (the compositor draws in place) must affect
    # neither the shared published frame nor the next tick's copy.
    first[:] = 255
    assert base.sum() == 0
    assert runtime._next_frame().sum() == 0


def test_next_frame_picks_up_new_sequence():
    webcam = _FakePacketWebcam()
    webcam.packet = (1, np.zeros((2, 2, 3), dtype=np.uint8))
    runtime = _runtime(webcam)
    assert runtime._next_frame().sum() == 0
    webcam.packet = (2, np.full((2, 2, 3), 7, dtype=np.uint8))
    assert int(runtime._next_frame().sum()) == 2 * 2 * 3 * 7

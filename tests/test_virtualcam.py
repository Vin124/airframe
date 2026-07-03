"""Unit tests for src.sinks.virtualcam.VirtualCamSink.

Covers lazy init on the first send() (recording width/height/fps/fmt via a
fake camera_factory), BGR->RGB conversion, missing-backend -> SinkError with
an actionable message, frame-size-change handling, and close() safety
(never-initialized, called twice). Pure/headless -- no real OBS/virtual
camera device is touched; a fake camera factory stands in for
pyvirtualcam.Camera, the same DI style as WebcamSource's fake Camera in
tests/test_sources.py.
"""

import numpy as np
import pytest

import pyvirtualcam
from src.sinks.base import Sink, SinkError
from src.sinks.virtualcam import DEFAULT_FPS, VirtualCamSink


class _FakeVirtualCamera:
    """Stand-in for pyvirtualcam.Camera: records init args and sent frames."""

    def __init__(self, width, height, fps, fmt) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.fmt = fmt
        self.sent_frames = []
        self.closed = False

    def send(self, frame: np.ndarray) -> None:
        self.sent_frames.append(frame)

    def close(self) -> None:
        self.closed = True


def _fake_factory(cameras: list):
    """A camera_factory that builds _FakeVirtualCamera instances into ``cameras``."""

    def factory(*, width, height, fps, fmt):
        cam = _FakeVirtualCamera(width, height, fps, fmt)
        cameras.append(cam)
        return cam

    return factory


def _raising_factory(exc: Exception):
    def factory(*, width, height, fps, fmt):
        raise exc

    return factory


def _bgr_frame() -> np.ndarray:
    """A 2x2 BGR frame with a distinct color per pixel to detect channel order."""
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    frame[0, 0] = (10, 20, 30)
    frame[0, 1] = (40, 50, 60)
    frame[1, 0] = (70, 80, 90)
    frame[1, 1] = (100, 110, 120)
    return frame


# --------------------------------------------------------------------------- #
# Lazy init on first send()
# --------------------------------------------------------------------------- #

def test_lazy_init_on_first_send_records_width_height_fps_fmt():
    cameras = []
    sink = VirtualCamSink(fps=25, camera_factory=_fake_factory(cameras))
    sink.send(_bgr_frame())
    assert len(cameras) == 1
    cam = cameras[0]
    assert (cam.width, cam.height) == (2, 2)
    assert cam.fps == 25
    assert cam.fmt == pyvirtualcam.PixelFormat.RGB


def test_default_fps_is_used_when_not_specified():
    cameras = []
    sink = VirtualCamSink(camera_factory=_fake_factory(cameras))
    sink.send(_bgr_frame())
    assert cameras[0].fps == DEFAULT_FPS


def test_second_send_does_not_reinitialize_camera():
    cameras = []
    sink = VirtualCamSink(camera_factory=_fake_factory(cameras))
    sink.send(_bgr_frame())
    sink.send(_bgr_frame())
    assert len(cameras) == 1


# --------------------------------------------------------------------------- #
# BGR -> RGB conversion
# --------------------------------------------------------------------------- #

def test_send_converts_bgr_to_rgb():
    cameras = []
    sink = VirtualCamSink(camera_factory=_fake_factory(cameras))
    sink.send(_bgr_frame())
    sent = cameras[0].sent_frames[0]
    assert tuple(int(v) for v in sent[0, 0]) == (30, 20, 10)
    assert tuple(int(v) for v in sent[0, 1]) == (60, 50, 40)
    assert tuple(int(v) for v in sent[1, 1]) == (120, 110, 100)


def test_send_does_not_mutate_input_frame():
    cameras = []
    sink = VirtualCamSink(camera_factory=_fake_factory(cameras))
    frame = _bgr_frame()
    original = frame.copy()
    sink.send(frame)
    assert np.array_equal(frame, original)


# --------------------------------------------------------------------------- #
# Missing backend
# --------------------------------------------------------------------------- #

def test_missing_backend_raises_sink_error_with_actionable_message():
    sink = VirtualCamSink(camera_factory=_raising_factory(RuntimeError("no backend found")))
    with pytest.raises(SinkError) as excinfo:
        sink.send(_bgr_frame())
    message = str(excinfo.value)
    assert "OBS" in message
    assert "obsproject.com" in message
    assert "no backend found" in message


# --------------------------------------------------------------------------- #
# Frame size change after init
# --------------------------------------------------------------------------- #

def test_frame_size_change_after_init_raises_sink_error():
    cameras = []
    sink = VirtualCamSink(camera_factory=_fake_factory(cameras))
    sink.send(_bgr_frame())
    bigger = np.zeros((4, 4, 3), dtype=np.uint8)
    with pytest.raises(SinkError):
        sink.send(bigger)
    # The camera was not recreated for the mismatched size.
    assert len(cameras) == 1


# --------------------------------------------------------------------------- #
# close()
# --------------------------------------------------------------------------- #

def test_close_closes_underlying_camera():
    cameras = []
    sink = VirtualCamSink(camera_factory=_fake_factory(cameras))
    sink.send(_bgr_frame())
    sink.close()
    assert cameras[0].closed is True


def test_close_is_safe_when_never_initialized():
    sink = VirtualCamSink(camera_factory=_fake_factory([]))
    sink.close()  # should not raise


def test_close_is_safe_when_called_twice():
    cameras = []
    sink = VirtualCamSink(camera_factory=_fake_factory(cameras))
    sink.send(_bgr_frame())
    sink.close()
    sink.close()  # should not raise
    assert cameras[0].closed is True


# --------------------------------------------------------------------------- #
# Sink contract
# --------------------------------------------------------------------------- #

def test_virtualcam_sink_is_a_sink():
    assert isinstance(VirtualCamSink(camera_factory=_fake_factory([])), Sink)

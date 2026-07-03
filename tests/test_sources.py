"""Unit tests for src.sources and src.sinks contracts.

Covers the base Source ``base_size`` default, ImageSource loading/alpha/read,
WebcamSource's mirror + camera lifecycle (with a fake camera, no real webcam),
and the PreviewSink surface. Pure/headless -- no webcam, no window shown.
"""

from typing import Optional

import numpy as np
import pytest

from src.sinks import PreviewSink, Sink
from src.sources import ImageSource, Source, SourceError, WebcamSource
from src.sources.image import ensure_bgra
from src.sources.webcam import WebcamSource as WebcamSourceDirect


# --------------------------------------------------------------------------- #
# Base Source contract
# --------------------------------------------------------------------------- #

class _FrameSource(Source):
    """A Source that returns a fixed frame, to exercise the base_size default."""

    def __init__(self, frame: Optional[np.ndarray]) -> None:
        self._frame = frame

    def read(self):
        return self._frame


def test_base_size_default_derives_from_frame_shape():
    frame = np.zeros((3, 7, 3), dtype=np.uint8)  # h=3, w=7
    assert _FrameSource(frame).base_size == (7, 3)


def test_base_size_default_is_none_when_no_frame():
    assert _FrameSource(None).base_size is None


def test_source_close_default_is_noop():
    # Should not raise.
    _FrameSource(None).close()


# --------------------------------------------------------------------------- #
# ImageSource
# --------------------------------------------------------------------------- #

def _solid_bgra(size=(10, 10), alpha=255) -> np.ndarray:
    w, h = size
    img = np.zeros((h, w, 4), dtype=np.uint8)
    img[:, :, 3] = alpha
    return img


def test_ensure_bgra_adds_opaque_alpha_to_bgr():
    result = ensure_bgra(np.zeros((5, 7, 3), dtype=np.uint8))
    assert result.shape == (5, 7, 4)
    assert np.all(result[:, :, 3] == 255)


def test_ensure_bgra_promotes_grayscale():
    result = ensure_bgra(np.zeros((5, 7), dtype=np.uint8))
    assert result.shape == (5, 7, 4)


def test_image_source_constructor_adds_alpha_when_missing():
    source = ImageSource(np.zeros((5, 7, 3), dtype=np.uint8))
    assert source.read().shape == (5, 7, 4)
    assert np.all(source.read()[:, :, 3] == 255)


def test_image_source_base_size_is_width_height():
    source = ImageSource(_solid_bgra(size=(7, 3)))
    assert source.base_size == (7, 3)


def test_image_source_read_returns_same_array():
    source = ImageSource(_solid_bgra())
    assert source.read() is source.read()  # stable identity, no re-decode


def test_load_missing_file_raises_source_error():
    with pytest.raises(SourceError):
        ImageSource.load("this_file_does_not_exist_12345.png")


def test_load_real_asset_has_alpha():
    source = ImageSource.load("assets/test.png")
    assert source.read().shape[2] == 4
    assert source.base_size == (300, 300)


# --------------------------------------------------------------------------- #
# WebcamSource (fake camera -- no real webcam)
# --------------------------------------------------------------------------- #

class _FakeCamera:
    """Stand-in for camera.Camera: records lifecycle, returns queued frames."""

    def __init__(self, frames) -> None:
        self._frames = list(frames)
        self.opened = False
        self.released = False

    def open(self):
        self.opened = True
        return self

    def read(self):
        return self._frames.pop(0) if self._frames else None

    def release(self):
        self.released = True


def _asymmetric_frame() -> np.ndarray:
    """A 2x2 BGR frame with distinct columns so a horizontal flip is detectable."""
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    frame[:, 0] = (1, 1, 1)  # left column
    frame[:, 1] = (9, 9, 9)  # right column
    return frame


def test_webcam_source_mirrors_horizontally_by_default():
    frame = _asymmetric_frame()
    source = WebcamSource(camera=_FakeCamera([frame]))
    out = source.read()
    # After a horizontal flip, the original left column (1) is now on the right.
    assert tuple(int(v) for v in out[0, 0]) == (9, 9, 9)
    assert tuple(int(v) for v in out[0, 1]) == (1, 1, 1)


def test_webcam_source_mirror_disabled_returns_frame_unflipped():
    frame = _asymmetric_frame()
    source = WebcamSource(camera=_FakeCamera([frame]), mirror=False)
    out = source.read()
    assert tuple(int(v) for v in out[0, 0]) == (1, 1, 1)
    assert tuple(int(v) for v in out[0, 1]) == (9, 9, 9)


def test_webcam_source_read_returns_none_when_camera_dry():
    source = WebcamSource(camera=_FakeCamera([]))
    assert source.read() is None


def test_webcam_source_context_manager_opens_and_releases():
    camera = _FakeCamera([_asymmetric_frame()])
    with WebcamSource(camera=camera) as source:
        assert camera.opened is True
        source.read()
    assert camera.released is True


def test_webcam_source_is_exported_from_package():
    assert WebcamSource is WebcamSourceDirect


# --------------------------------------------------------------------------- #
# PreviewSink surface (no window actually shown)
# --------------------------------------------------------------------------- #

def test_preview_sink_is_a_sink_with_window_name():
    sink = PreviewSink("my-window")
    assert isinstance(sink, Sink)
    assert sink.window_name == "my-window"


def test_preview_sink_default_window_name():
    assert PreviewSink().window_name == "handy"

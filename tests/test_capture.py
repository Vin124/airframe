"""Unit tests for src.sources.capture (WindowSource / MonitorSource).

All headless: a fake capture backend stands in for windows_capture.WindowsCapture
(the same DI style as VirtualCamSink's fake camera factory), and frames are fed
through the source's own ingest path with fake Frame objects -- no real Windows
Graphics Capture thread is ever started. Covers the pure BGRA/crop helpers,
latest-frame-wins read semantics, base_size caching, graceful death on close,
close() idempotence, and fail-fast SourceError paths.
"""

import numpy as np
import pytest

from src.sources import MonitorSource, Source, SourceError, WindowSource
from src.sources.capture import (
    OPAQUE_ALPHA,
    _CaptureSource,
    crop_region,
    frame_to_bgra,
)


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #

class _FakeFrame:
    """Stand-in for windows_capture.Frame: only exposes .frame_buffer."""

    def __init__(self, frame_buffer: np.ndarray) -> None:
        self.frame_buffer = frame_buffer


class _FakeControl:
    """Stand-in for CaptureControl: records stop() calls."""

    def __init__(self) -> None:
        self.stops = 0

    def stop(self) -> None:
        self.stops += 1


class _FakeCapture:
    """Stand-in for WindowsCapture: records handlers + init kwargs, never threads."""

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.handlers = {}
        self.control = _FakeControl()
        self.started = False

    def event(self, handler):
        self.handlers[handler.__name__] = handler
        return handler

    def start_free_threaded(self):
        self.started = True
        return self.control


def _fake_factory(created: list, *, start_error: Exception = None):
    """A capture_factory building _FakeCapture(s), optionally raising on start."""

    def factory(**kwargs):
        cap = _FakeCapture(**kwargs)
        if start_error is not None:
            def boom():
                raise start_error
            cap.start_free_threaded = boom  # type: ignore[assignment]
        created.append(cap)
        return cap

    return factory


def _bgra(width: int, height: int, fill: int, alpha: int = 0) -> np.ndarray:
    """A solid HxWx4 BGRA frame with a known BGR fill and (deliberately) alpha."""
    frame = np.empty((height, width, 4), dtype=np.uint8)
    frame[:, :, :3] = fill
    frame[:, :, 3] = alpha
    return frame


# --------------------------------------------------------------------------- #
# frame_to_bgra
# --------------------------------------------------------------------------- #

def test_frame_to_bgra_copies_off_native_buffer():
    src = _bgra(4, 3, fill=10, alpha=200)
    out = frame_to_bgra(src)
    assert out.flags["OWNDATA"] is True  # independent copy, not a view
    src[:] = 99  # mutating the "native buffer" must not touch the copy
    assert out[0, 0, 0] == 10


def test_frame_to_bgra_forces_opaque_alpha():
    out = frame_to_bgra(_bgra(2, 2, fill=5, alpha=0))
    assert np.all(out[:, :, 3] == OPAQUE_ALPHA)


def test_frame_to_bgra_promotes_bgr_to_bgra():
    bgr = np.full((3, 5, 3), 7, dtype=np.uint8)
    out = frame_to_bgra(bgr)
    assert out.shape == (3, 5, 4)
    assert np.all(out[:, :, 3] == OPAQUE_ALPHA)


def test_frame_to_bgra_promotes_grayscale_to_bgra():
    gray = np.full((3, 5), 7, dtype=np.uint8)
    out = frame_to_bgra(gray)
    assert out.shape == (3, 5, 4)


# --------------------------------------------------------------------------- #
# crop_region
# --------------------------------------------------------------------------- #

def test_crop_region_basic():
    frame = _bgra(10, 8, fill=0, alpha=255)
    out = crop_region(frame, (2, 1, 4, 3))
    assert out.shape[:2] == (3, 4)  # (h, w)


def test_crop_region_clamps_to_bounds():
    frame = _bgra(10, 8, fill=0, alpha=255)
    out = crop_region(frame, (8, 6, 100, 100))  # runs off the right/bottom
    assert out.shape[1] <= 2 and out.shape[0] <= 2
    assert out.shape[0] >= 1 and out.shape[1] >= 1


def test_crop_region_never_empty():
    frame = _bgra(10, 8, fill=0, alpha=255)
    out = crop_region(frame, (100, 100, 5, 5))  # origin past the frame
    assert out.shape[0] >= 1 and out.shape[1] >= 1


# --------------------------------------------------------------------------- #
# Latest-frame-wins read + base_size caching (via _ingest_frame)
# --------------------------------------------------------------------------- #

def _opened_monitor(created: list, region=None) -> MonitorSource:
    source = MonitorSource(1, region=region, capture_factory=_fake_factory(created))
    source.open()
    return source


def test_read_none_before_first_frame():
    source = _opened_monitor([])
    assert source.read() is None
    assert source.base_size is None


def test_latest_frame_wins():
    source = _opened_monitor([])
    source._ingest_frame(_FakeFrame(_bgra(4, 4, fill=1, alpha=255)))
    source._ingest_frame(_FakeFrame(_bgra(4, 4, fill=2, alpha=255)))
    assert source.read()[0, 0, 0] == 2  # newest frame is served


def test_base_size_cached_after_first_frame():
    source = _opened_monitor([])
    source._ingest_frame(_FakeFrame(_bgra(7, 3, fill=1, alpha=255)))
    assert source.base_size == (7, 3)  # (width, height), non-consuming


def test_read_returns_bgra_with_opaque_alpha():
    source = _opened_monitor([])
    source._ingest_frame(_FakeFrame(_bgra(4, 4, fill=1, alpha=0)))
    frame = source.read()
    assert frame.shape[2] == 4
    assert np.all(frame[:, :, 3] == OPAQUE_ALPHA)


def test_region_crop_applied_on_ingest():
    created = []
    source = _opened_monitor(created, region=(1, 1, 2, 2))
    source._ingest_frame(_FakeFrame(_bgra(10, 8, fill=1, alpha=255)))
    assert source.base_size == (2, 2)


# --------------------------------------------------------------------------- #
# Graceful death (on_closed) + close()
# --------------------------------------------------------------------------- #

def test_dead_after_closed_read_returns_none(capsys):
    source = _opened_monitor([])
    source._ingest_frame(_FakeFrame(_bgra(4, 4, fill=1, alpha=255)))
    assert source.read() is not None
    source._mark_dead()
    assert source.read() is None
    assert source.base_size is None
    err = capsys.readouterr().err
    assert "monitor 1" in err and "keeps running" in err


def test_mark_dead_notice_printed_once(capsys):
    source = _opened_monitor([])
    source._mark_dead()
    source._mark_dead()
    assert capsys.readouterr().err.count("keeps running") == 1


def test_close_stops_capture_thread_and_is_idempotent():
    created = []
    source = _opened_monitor(created)
    source.close()
    source.close()  # must not raise
    assert created[0].control.stops == 1  # only stopped once
    assert source.read() is None


def test_close_before_open_is_safe():
    source = MonitorSource(1, capture_factory=_fake_factory([]))
    source.close()  # never opened -> no control -> no raise
    assert source.read() is None


# --------------------------------------------------------------------------- #
# open(): handler wiring + fail-fast SourceError
# --------------------------------------------------------------------------- #

def test_open_registers_named_handlers_and_starts():
    created = []
    _opened_monitor(created)
    cap = created[0]
    assert cap.started is True
    assert set(cap.handlers) == {"on_frame_arrived", "on_closed"}


def test_registered_frame_handler_updates_read():
    created = []
    source = _opened_monitor(created)
    on_frame = created[0].handlers["on_frame_arrived"]
    on_frame(_FakeFrame(_bgra(4, 4, fill=3, alpha=255)), object())
    assert source.read()[0, 0, 0] == 3


def test_registered_closed_handler_marks_dead():
    created = []
    source = _opened_monitor(created)
    created[0].handlers["on_closed"]()
    assert source.read() is None


def test_window_not_found_raises_source_error_with_title():
    factory = _fake_factory([], start_error=Exception("Failed to find window"))
    source = WindowSource("Notepad", capture_factory=factory)
    with pytest.raises(SourceError) as excinfo:
        source.open()
    message = str(excinfo.value)
    assert "Notepad" in message
    assert "Failed to find window" in message


def test_bad_monitor_index_at_start_raises_source_error():
    factory = _fake_factory([], start_error=Exception("Failed to find the specified monitor"))
    source = MonitorSource(9, capture_factory=factory)
    with pytest.raises(SourceError) as excinfo:
        source.open()
    assert "monitor 9" in str(excinfo.value)


def test_missing_windows_capture_package_raises_source_error():
    def import_error_factory(**kwargs):
        raise ImportError("No module named 'windows_capture'")

    source = MonitorSource(1, capture_factory=import_error_factory)
    with pytest.raises(SourceError) as excinfo:
        source.open()
    assert "windows-capture" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# Construction-time validation
# --------------------------------------------------------------------------- #

def test_window_source_requires_nonempty_title():
    with pytest.raises(SourceError):
        WindowSource("   ")


def test_monitor_source_rejects_index_below_one():
    with pytest.raises(SourceError):
        MonitorSource(0)


def test_monitor_source_rejects_bad_region():
    with pytest.raises(SourceError):
        MonitorSource(1, region=(0, 0, 0, 5))  # zero width


def test_capture_kwargs_are_passed_to_factory():
    created = []
    WindowSource("Chrome", capture_factory=_fake_factory(created)).open()
    assert created[0].kwargs["window_name"] == "Chrome"
    monitor_created = []
    MonitorSource(2, capture_factory=_fake_factory(monitor_created)).open()
    assert monitor_created[0].kwargs["monitor_index"] == 2


# --------------------------------------------------------------------------- #
# Type/contract
# --------------------------------------------------------------------------- #

def test_sources_are_source_subclasses():
    assert issubclass(WindowSource, Source)
    assert issubclass(MonitorSource, Source)
    assert issubclass(_CaptureSource, Source)

"""Unit tests for src.sources.threaded.ThreadedSource.

Uses a fake blocking inner source (a queue of frames whose read() blocks briefly)
so publish/read/close semantics are exercised with a real capture thread but no
webcam. Timing-sensitive assertions poll a short deadline instead of sleeping a
fixed time, keeping the suite fast and deterministic (no sleep > ~0.2s).
"""

import threading
import time

import numpy as np
import pytest

from src.sources import Source, ThreadedSource


def _wait_until(predicate, timeout=0.5):
    """Poll ``predicate`` until true or ``timeout`` elapses; return the result."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.002)
    return predicate()


class _FakeBlockingSource(Source):
    """A blocking inner Source: read() returns queued frames, then blocks/None.

    Records open/close lifecycle. Once the queue drains, read() returns None
    (like a camera hiccup) so the capture thread idles instead of ending.
    """

    def __init__(self, frames, base_size=(4, 4)) -> None:
        self._frames = list(frames)
        self._lock = threading.Lock()
        self._base_size = base_size
        self.opened = False
        self.closed = False

    def open(self):
        self.opened = True
        return self

    def read(self):
        with self._lock:
            if self._frames:
                return self._frames.pop(0)
        return None  # drained -> capture thread idles

    def push(self, frame):
        with self._lock:
            self._frames.append(frame)

    @property
    def base_size(self):
        return self._base_size

    def close(self):
        self.closed = True


def _frame(fill):
    f = np.empty((4, 4, 3), dtype=np.uint8)
    f[:] = fill
    return f


def test_open_opens_inner_then_publishes_frames():
    inner = _FakeBlockingSource([_frame(7)])
    source = ThreadedSource(inner)
    try:
        source.open()
        assert inner.opened is True
        assert _wait_until(lambda: source.read() is not None)
        assert source.read()[0, 0, 0] == 7
    finally:
        source.close()


def test_read_none_before_first_frame():
    inner = _FakeBlockingSource([])  # never produces a frame
    source = ThreadedSource(inner)
    try:
        source.open()
        assert source.read() is None
        assert source.latest_packet() is None
        assert source.frame_age_ms() is None
    finally:
        source.close()


def test_latest_frame_wins_and_sequence_increments():
    inner = _FakeBlockingSource([_frame(1)])
    source = ThreadedSource(inner)
    try:
        source.open()
        assert _wait_until(lambda: source.latest_packet() is not None)
        first_seq = source.latest_packet()[0]
        inner.push(_frame(2))
        assert _wait_until(lambda: source.read()[0, 0, 0] == 2)
        assert source.latest_packet()[0] > first_seq  # newer frame -> higher seq
    finally:
        source.close()


def test_base_size_delegates_to_inner():
    inner = _FakeBlockingSource([], base_size=(9, 5))
    source = ThreadedSource(inner)
    assert source.base_size == (9, 5)  # non-consuming, no thread needed


def test_frame_age_ms_uses_injected_clock():
    # Inject a clock so the age is deterministic: publish at t=100, read age at t=101.
    ticks = iter([100.0, 101.0, 101.0, 101.0])
    inner = _FakeBlockingSource([_frame(3)])
    source = ThreadedSource(inner, clock=lambda: next(ticks))
    try:
        source.open()
        assert _wait_until(lambda: source.frame_age_ms() is not None)
        # published_at=100.0, next clock read 101.0 -> ~1000ms old.
        assert source.frame_age_ms() == pytest.approx(1000.0)
    finally:
        source.close()


def test_close_stops_thread_and_closes_inner_idempotently():
    inner = _FakeBlockingSource([_frame(1)])
    source = ThreadedSource(inner)
    source.open()
    assert _wait_until(lambda: source.read() is not None)
    source.close()
    source.close()  # idempotent -> must not raise
    assert inner.closed is True
    assert source._thread is None


def test_close_before_open_is_safe():
    inner = _FakeBlockingSource([])
    source = ThreadedSource(inner)
    source.close()  # never opened -> no thread -> no raise
    assert inner.closed is True


def test_returned_frame_is_the_shared_published_array():
    # read() returns the shared frame (read-only contract); the same array the
    # packet carries -- callers copy before mutating (the compositor does).
    inner = _FakeBlockingSource([_frame(4)])
    source = ThreadedSource(inner)
    try:
        source.open()
        assert _wait_until(lambda: source.latest_packet() is not None)
        assert source.read() is source.latest_packet()[1]
    finally:
        source.close()


def test_is_a_source_subclass():
    assert issubclass(ThreadedSource, Source)

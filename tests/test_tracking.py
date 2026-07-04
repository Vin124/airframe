"""Unit tests for src.tracking: publisher state machine + crash policy.

All deterministic and threadless: a fake tracker + fake detectors stand in for
MediaPipe, and ``poll_once`` is driven directly (the thread loop is a thin
wrapper over it), so no real detection, camera, or sleep is involved.
"""

import numpy as np
import pytest

from src.gestures import GestureState
from src.tracking import (
    TrackingResult,
    TrackingWorker,
    detect_and_update,
    initial_result,
)
from src.tuning import GestureTuning


class _FakeTracker:
    """Stand-in for HandTracker: returns queued hand lists, records detect calls."""

    def __init__(self, hands_sequence=None, raises=None) -> None:
        self._sequence = list(hands_sequence or [])
        self._raises = raises
        self.detect_calls = 0

    def detect(self, frame):
        self.detect_calls += 1
        if self._raises is not None:
            raise self._raises
        if self._sequence:
            return self._sequence.pop(0)
        return []


class _FakeDetector:
    """Stand-in for GestureDetector: echoes hand presence, records threshold swaps."""

    def __init__(self) -> None:
        self.updates = []
        self.thresholds = None  # last (enter, exit) applied via set_thresholds

    def set_thresholds(self, enter, exit):
        self.thresholds = (enter, exit)

    def update(self, hand):
        self.updates.append(hand)
        return GestureState(
            is_pinching=hand is not None, pinch_point=(0.1, 0.2), is_grab=False
        )


def _frame():
    return np.zeros((4, 4, 3), dtype=np.uint8)


def _provider(packets):
    """A frame provider that yields the given packets then keeps returning the last."""
    box = {"packets": list(packets)}

    def provider():
        if len(box["packets"]) > 1:
            return box["packets"].pop(0)
        return box["packets"][0] if box["packets"] else None

    return provider


# --------------------------------------------------------------------------- #
# detect_and_update (pure)
# --------------------------------------------------------------------------- #

def test_detect_and_update_feeds_hand_slots_to_detectors():
    hand_a = [(0.0, 0.0, 0.0)]
    tracker = _FakeTracker(hands_sequence=[[hand_a]])
    detectors = [_FakeDetector(), _FakeDetector()]
    hands, gestures = detect_and_update(tracker, detectors, _frame())
    assert hands == (hand_a,)
    assert gestures[0].is_pinching is True  # slot 0 got the hand
    assert gestures[1].is_pinching is False  # slot 1 got None (only one hand)
    assert len(gestures) == 2


def test_detect_and_update_no_hands_gives_idle_states():
    tracker = _FakeTracker(hands_sequence=[[]])
    detectors = [_FakeDetector()]
    hands, gestures = detect_and_update(tracker, detectors, _frame())
    assert hands == ()
    assert gestures[0].is_pinching is False


# --------------------------------------------------------------------------- #
# initial_result
# --------------------------------------------------------------------------- #

def test_initial_result_has_one_idle_state_per_detector():
    result = initial_result(2)
    assert result.hands == ()
    assert result.seq == 0
    assert len(result.gesture_states) == 2
    assert all(not g.is_pinching and not g.is_grab for g in result.gesture_states)


# --------------------------------------------------------------------------- #
# TrackingWorker.poll_once: dedup + publish
# --------------------------------------------------------------------------- #

def test_poll_once_processes_a_new_frame_and_publishes():
    tracker = _FakeTracker(hands_sequence=[[[(0.0, 0.0, 0.0)]]])
    worker = TrackingWorker(tracker, [_FakeDetector()], _provider([(1, _frame())]))
    assert worker.poll_once() is True
    result = worker.latest()
    assert result.seq == 1
    assert len(result.hands) == 1
    assert tracker.detect_calls == 1


def test_poll_once_skips_when_no_frame():
    worker = TrackingWorker(_FakeTracker(), [_FakeDetector()], _provider([]))
    assert worker.poll_once() is False
    assert worker.latest().seq == 0  # still the initial result


def test_poll_once_skips_already_processed_sequence():
    tracker = _FakeTracker(hands_sequence=[[], [], []])
    # Provider keeps returning the same seq=5 packet.
    worker = TrackingWorker(tracker, [_FakeDetector()], _provider([(5, _frame())]))
    assert worker.poll_once() is True
    assert worker.poll_once() is False  # same seq -> no re-detection
    assert tracker.detect_calls == 1  # detected exactly once


def test_poll_once_records_track_ms():
    # First read is the smoothing timestamp (outside the timer); the next two
    # are the timer's start/end (20ms span).
    ticks = iter([9.99, 10.0, 10.02, 99.0, 99.0])
    tracker = _FakeTracker(hands_sequence=[[]])
    worker = TrackingWorker(
        tracker, [_FakeDetector()], _provider([(1, _frame())]), clock=lambda: next(ticks)
    )
    worker.poll_once()
    assert worker.latest().track_ms == pytest.approx(20.0)


def test_poll_once_propagates_tracker_exception():
    tracker = _FakeTracker(raises=RuntimeError("mediapipe boom"))
    worker = TrackingWorker(tracker, [_FakeDetector()], _provider([(1, _frame())]))
    with pytest.raises(RuntimeError):
        worker.poll_once()


# --------------------------------------------------------------------------- #
# Crash policy
# --------------------------------------------------------------------------- #

def test_on_crash_records_error_and_prints_once(capsys):
    worker = TrackingWorker(_FakeTracker(), [_FakeDetector()], _provider([]))
    exc = RuntimeError("detector died")
    worker._on_crash(exc)
    assert worker.error is exc
    err = capsys.readouterr().err
    assert "hand tracking stopped" in err
    assert "keeps running" in err


def test_crash_publishes_idle_result(capsys):
    # A crash must publish an IDLE result (not freeze the crash-time snapshot):
    # a frozen is_pinching=True would pin a grabbed layer at the crash-time
    # pinch point forever, and 'r' couldn't clear it (fresh InteractionState
    # would read the frozen pinch as a new rising edge and re-grab).
    tracker = _FakeTracker(hands_sequence=[[[(0.0, 0.0, 0.0)]]])
    worker = TrackingWorker(tracker, [_FakeDetector()], _provider([(1, _frame())]))
    worker.poll_once()
    assert worker.latest().seq == 1
    worker._on_crash(RuntimeError("later crash"))
    idle = worker.latest()
    assert idle.hands == ()
    assert len(idle.gesture_states) == 1  # one per detector
    assert all(not g.is_pinching for g in idle.gesture_states)


# --------------------------------------------------------------------------- #
# Live tuning: set_tuning atomic publish + per-poll apply to owned detectors
# --------------------------------------------------------------------------- #

def test_set_tuning_applies_to_detectors_on_next_poll():
    detectors = [_FakeDetector(), _FakeDetector()]
    worker = TrackingWorker(_FakeTracker(), detectors, _provider([(1, _frame())]))
    worker.set_tuning(GestureTuning(0.03, 0.06))
    assert all(d.thresholds is None for d in detectors)  # not applied until poll
    worker.poll_once()
    assert all(d.thresholds == (0.03, 0.06) for d in detectors)


def test_set_tuning_applies_even_without_a_new_frame():
    # No frame -> poll_once returns False, but a pending tuning must still land so
    # a slider nudge isn't stuck waiting on the next camera frame.
    detector = _FakeDetector()
    worker = TrackingWorker(_FakeTracker(), [detector], _provider([]))
    worker.set_tuning(GestureTuning(0.04, 0.07))
    assert worker.poll_once() is False
    assert detector.thresholds == (0.04, 0.07)


def test_tuning_applied_once_then_reapplied_on_change():
    detector = _FakeDetector()
    worker = TrackingWorker(
        _FakeTracker(hands_sequence=[[], [], []]),
        [detector],
        _provider([(1, _frame()), (2, _frame()), (3, _frame())]),
    )
    worker.set_tuning(GestureTuning(0.03, 0.06))
    worker.poll_once()
    detector.thresholds = None  # clear to prove the next poll doesn't re-apply
    worker.poll_once()
    assert detector.thresholds is None  # same tuning reference -> not re-applied
    worker.set_tuning(GestureTuning(0.05, 0.09))  # a new reference
    worker.poll_once()
    assert detector.thresholds == (0.05, 0.09)


def test_stop_is_safe_before_start():
    worker = TrackingWorker(_FakeTracker(), [_FakeDetector()], _provider([]))
    # Never started -> no thread -> no raise, and "stopped" is truthfully True
    # (callers may free the native tracker).
    assert worker.stop() is True
    assert worker._thread is None


def test_result_default_is_immutable_snapshot():
    assert isinstance(worker_result := TrackingResult(), TrackingResult)
    with pytest.raises(Exception):
        worker_result.seq = 5  # frozen

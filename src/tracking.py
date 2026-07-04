"""Threaded hand tracking + gesture detection, run at its own cadence.

MediaPipe's ``HandLandmarker`` (VIDEO mode) needs strictly increasing timestamps
and single-threaded access, and a ``detect()`` costs ~40ms -- run inline it
throttles the pipeline. ``TrackingWorker`` moves detection AND the frame-to-frame
gesture hysteresis onto one dedicated thread that consumes the newest webcam
frame and publishes a ``TrackingResult`` (hands + gesture states + timing)
latest-frame-wins. The main loop only READS the published result, so every touch
of the tracker and the gesture detectors stays on this single thread.

State ownership (the app's single-writer rule):
    - THIS thread owns the ``HandTracker`` and the ``GestureDetector`` list (their
      pinch hysteresis latches). Nothing else touches them.
    - The MAIN loop owns the scene/interaction and only reads published results.
    A published ``TrackingResult`` is an immutable snapshot assigned with one
    atomic reference store (GIL-safe, no lock), the same discipline the capture
    sources use for frames.

Live pinch tuning:
    The panel sliders retune pinch detection without touching detector internals
    from the main thread. ``set_tuning`` publishes an immutable ``GestureTuning``
    via a single atomic reference store (main thread -> this thread); each poll
    reads it and, when the reference changed, swaps the thresholds onto the
    detectors THIS thread owns (``GestureDetector.set_thresholds``). So the main
    thread never mutates a detector -- it only hands off an immutable snapshot,
    the same latest-wins discipline as frames and results.

Cadence + dedup:
    The worker processes a frame only when the capture source publishes a NEW one
    (tracked by sequence number); between new frames it sleeps briefly instead of
    re-detecting the same pixels. So tracking runs at its natural rate, decoupled
    from the main loop's display rate.

Crash policy:
    A detect()/gesture exception is caught, recorded on ``error``, printed ONCE to
    stderr, and STOPS the tracking loop. An IDLE no-hands result is published as
    the final one (NOT the crash-time snapshot): a frozen ``is_pinching=True``
    would pin a grabbed layer at the crash-time pinch point forever -- even 'r'
    couldn't clear it, since a fresh InteractionState would read the frozen pinch
    as a new rising edge and re-grab. Idle releases everything; the main loop
    (preview/virtualcam/keys) keeps running. A dead tracker is a degraded but
    live app.
"""

import sys
import threading
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np

from src.gestures import GestureDetector, GestureState
from src.smoothing import LandmarkSmoother
from src.hand_tracker import Hand, HandTracker
from src.timing import StageTimer
from src.tuning import GestureTuning

# Poll interval when no new frame is available yet, so the worker doesn't
# busy-spin between camera frames. Small relative to the ~40ms detect cost.
_IDLE_SLEEP_SECONDS = 0.002

# How long stop() waits for the tracking thread to exit (one detect can be in
# flight when the stop is requested).
_JOIN_TIMEOUT_SECONDS = 2.0

# A frame packet as published by ThreadedSource: (sequence number, frame array).
# Matches src.sources.threaded.Packet exactly.
Packet = Tuple[int, np.ndarray]
FrameProvider = Callable[[], Optional[Packet]]


@dataclass(frozen=True)
class TrackingResult:
    """Immutable snapshot the main loop reads each frame.

    ``hands`` is the detected hands (each 21 landmark tuples); ``gesture_states``
    is one per detector (slot 0 = primary hand); ``track_ms`` is how long the last
    detect + gesture update took; ``seq`` is the webcam frame sequence this result
    came from (0 for the initial, pre-detection result).
    """

    hands: Tuple[Hand, ...] = ()
    gesture_states: Tuple[GestureState, ...] = ()
    track_ms: float = 0.0
    seq: int = 0


def _idle_gesture_state() -> GestureState:
    """A non-pinching, non-grabbing gesture state (no hand)."""
    return GestureState(is_pinching=False, pinch_point=(0.0, 0.0), is_grab=False)


def _wrist_xy(hand: Hand) -> Tuple[float, float]:
    """The (x, y) of a hand's wrist landmark (index 0)."""
    x, y, _z = hand[0]
    return (x, y)


class HandSlotMatcher:
    """Keep each physical hand in a stable detector slot across frames.

    MediaPipe returns detected hands in NO guaranteed order, so with two hands
    in frame a hand can swap detector slots between frames -- taking the pinch
    hysteresis latch and the One-Euro filter bank of the OTHER hand with it
    (a mid-drag pinch teleports to the other hand for a frame). This matcher
    reorders each frame's two-hand detection so the assignment minimizes total
    wrist travel from the previous frame's per-slot wrist positions.

    Only the two-hand case needs a decision: zero/one detected hands pass
    through unchanged (a lone hand stays in slot 0, as before), and the app
    never tracks more than two. A slot's last wrist is kept while the slot is
    empty so a hand that drops out for a few frames re-lands in its old slot.
    Single owner (the tracking thread) -- not thread-safe by design.
    """

    def __init__(self, num_slots: int) -> None:
        self._prev_wrists: List[Optional[Tuple[float, float]]] = [None] * num_slots

    @staticmethod
    def _travel(prev: Optional[Tuple[float, float]], hand: Hand) -> float:
        """Wrist travel from a slot's previous position; 0 for a fresh slot.

        A ``None`` (never-seen) slot contributes nothing, so the choice is
        driven entirely by the slots that DO have history.
        """
        if prev is None:
            return 0.0
        wx, wy = _wrist_xy(hand)
        return float(np.hypot(wx - prev[0], wy - prev[1]))

    def assign(self, hands: List[Hand]) -> List[Hand]:
        """Return ``hands`` reordered for slot stability; remember the wrists."""
        if len(hands) == 2 and len(self._prev_wrists) >= 2:
            keep = (
                self._travel(self._prev_wrists[0], hands[0])
                + self._travel(self._prev_wrists[1], hands[1])
            )
            swap = (
                self._travel(self._prev_wrists[0], hands[1])
                + self._travel(self._prev_wrists[1], hands[0])
            )
            if swap < keep:
                hands = [hands[1], hands[0]]
        for slot in range(min(len(hands), len(self._prev_wrists))):
            if hands[slot]:
                self._prev_wrists[slot] = _wrist_xy(hands[slot])
        return hands


def initial_result(num_detectors: int) -> TrackingResult:
    """A no-hands result with one idle gesture state per detector.

    Lets the main loop run -- and ``interaction.step`` safely read
    ``gesture_states[0]`` / ``[1]`` -- before the tracking thread has produced its
    first real result.
    """
    return TrackingResult(
        hands=(),
        gesture_states=tuple(_idle_gesture_state() for _ in range(num_detectors)),
        track_ms=0.0,
        seq=0,
    )


def detect_and_update(
    tracker: HandTracker,
    detectors: List[GestureDetector],
    frame,
    smoother: Optional["LandmarkSmoother"] = None,
    t: float = 0.0,
    matcher: Optional[HandSlotMatcher] = None,
) -> Tuple[Tuple[Hand, ...], Tuple[GestureState, ...]]:
    """Detect hands, de-jitter them, and feed hand slot ``i`` to ``detectors[i]``.

    Returns ``(hands, gesture_states)`` with one gesture state per detector (slot
    0 = primary hand). When a ``matcher`` is given, a two-hand detection is
    reordered FIRST so each physical hand keeps its slot across frames (MediaPipe
    guarantees no order) -- the per-slot smoother banks and gesture hysteresis
    latches then stay attached to the same hand. When a ``smoother`` is given the
    slot-matched raw landmarks are One-Euro smoothed at time ``t`` BEFORE gesture
    detection, so pinch/grab read a steadier hand (and the published hands the
    skeleton draws are smoothed too). Kept a free function so it's testable with a
    fake tracker and fake detectors.
    """
    hands = list(tracker.detect(frame))
    if matcher is not None:
        hands = matcher.assign(hands)
    if smoother is not None:
        hands = smoother.smooth(hands, t)
    hands = tuple(hands)
    gesture_states = tuple(
        detector.update(hands[i] if i < len(hands) else None)
        for i, detector in enumerate(detectors)
    )
    return hands, gesture_states


class TrackingWorker:
    """Run hand tracking + gesture detection on a dedicated thread.

    Constructed with the (single-threaded) ``HandTracker``, the gesture detectors,
    and a ``frame_provider`` returning the newest ``(seq, frame)`` packet (the
    threaded webcam's ``latest_packet``). ``latest()`` returns the most recent
    published ``TrackingResult`` for the main loop to read.
    """

    def __init__(
        self,
        tracker: HandTracker,
        detectors: List[GestureDetector],
        frame_provider: FrameProvider,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._tracker = tracker
        self._detectors = detectors
        self._frame_provider = frame_provider
        self._clock = clock
        # One-Euro landmark smoother (de-jitters MediaPipe output before gesture
        # detection); owned solely by this thread, one filter bank per hand slot.
        self._smoother = LandmarkSmoother(len(detectors))
        # Slot matcher: keeps each physical hand in the same detector/smoother
        # slot across frames (MediaPipe's output order is unstable with two
        # hands). Owned solely by this thread, like the smoother.
        self._matcher = HandSlotMatcher(len(detectors))
        self._result = initial_result(len(detectors))
        self._last_seq = 0
        # Live pinch tuning: the main thread publishes a new GestureTuning here
        # (one atomic reference store); this thread applies it between frames and
        # records which reference it applied so a fresh publish is never missed
        # or re-applied. None = "no live change yet" (detectors keep the
        # thresholds they were constructed with -- the resolved startup config).
        self._pending_tuning: Optional[GestureTuning] = None
        self._applied_tuning: Optional[GestureTuning] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.error: Optional[Exception] = None

    def start(self) -> "TrackingWorker":
        """Start the tracking thread (daemon)."""
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="tracking-thread", daemon=True
        )
        self._thread.start()
        return self

    def latest(self) -> TrackingResult:
        """Return the most recently published result (never blocks)."""
        return self._result

    def set_tuning(self, tuning: GestureTuning) -> None:
        """Publish new pinch thresholds for this thread to apply (main thread).

        A single atomic reference store (GIL-safe, no lock). The tracking thread
        picks it up on its next poll and swaps it onto the detectors it owns --
        the main thread never touches detector internals. ``tuning`` must already
        be clamped (see ``src.tuning.clamp_tuning``).
        """
        self._pending_tuning = tuning

    def _apply_pending_tuning(self) -> None:
        """Swap a freshly published tuning onto the owned detectors (this thread).

        Compares by identity against the last-applied reference, so a new publish
        (``clamp_tuning`` always returns a fresh object) is applied exactly once
        and a concurrent re-publish is never lost -- no clear-to-None race.
        """
        tuning = self._pending_tuning  # one atomic read
        if tuning is not None and tuning is not self._applied_tuning:
            for detector in self._detectors:
                detector.set_thresholds(tuning.enter, tuning.exit)
            self._applied_tuning = tuning

    def poll_once(self) -> bool:
        """Process the newest frame if it's new; return True iff a detect ran.

        Applies any freshly published pinch tuning FIRST (so a slider nudge takes
        effect even on a frame the worker would otherwise skip), then skips
        (returns False) when there's no frame yet or the newest one shares the
        sequence already processed -- the whole point of running at its own
        cadence is to never re-detect the same pixels. On a genuinely new frame it
        detects + updates gestures and publishes the result. Exceptions propagate
        (the loop's crash policy handles them). Exposed for deterministic tests.
        """
        self._apply_pending_tuning()
        packet = self._frame_provider()
        if packet is None or packet[0] == self._last_seq:
            return False
        seq, frame = packet
        now = self._clock()  # smoothing timestamp, read outside the timed region
        with StageTimer(self._clock) as timer:
            hands, gesture_states = detect_and_update(
                self._tracker, self._detectors, frame, self._smoother, now,
                self._matcher,
            )
        self._last_seq = seq
        self._result = TrackingResult(  # single atomic publish
            hands=hands, gesture_states=gesture_states, track_ms=timer.ms, seq=seq
        )
        return True

    def _on_crash(self, exc: Exception) -> None:
        """Record + announce a tracker crash once, then publish an IDLE result.

        The idle publish (no hands, not pinching) releases any in-flight
        grab/resize -- see the module docstring's crash policy for why freezing
        the crash-time snapshot instead would wedge a grab unrecoverably.
        """
        self.error = exc
        self._result = initial_result(len(self._detectors))  # atomic publish
        print(
            f"Error: hand tracking stopped after an exception ({exc}); tracking "
            "is now idle and the app keeps running.",
            file=sys.stderr,
        )

    def _run(self) -> None:
        """Tracking loop (own thread): poll for new frames until stopped or crashed."""
        while not self._stop.is_set():
            try:
                processed = self.poll_once()
            except Exception as exc:  # noqa: BLE001 - keep the session alive, frozen
                self._on_crash(exc)
                return
            if not processed:
                time.sleep(_IDLE_SLEEP_SECONDS)

    def stop(self) -> bool:
        """Signal the thread to stop and join it with a timeout. Idempotent.

        Returns True when the thread has actually exited (or never ran); False
        when the join timed out because a native detect() is wedged. On False
        the caller must NOT free native resources the thread may still touch
        (the HandTracker) -- the daemon thread is reclaimed at process exit.
        """
        self._stop.set()
        thread = self._thread
        self._thread = None
        if thread is None:
            return True
        thread.join(timeout=_JOIN_TIMEOUT_SECONDS)
        return not thread.is_alive()

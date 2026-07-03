"""Pure timing helpers: per-stage instrumentation and main-loop pacing.

Everything here is side-effect-free apart from reading a monotonic clock, so the
runtime's frame pacing and the HUD's per-stage numbers are unit-testable with a
fake clock -- no threads, no camera, no real sleeps.

Pieces:
  - ``StageTimer``      : a context manager that measures a block's duration in
                          milliseconds (used to time composite / sink stages).
  - ``StageTimings``    : an immutable bundle of the five HUD stage numbers.
  - ``frame_budget_seconds`` / ``pace_wait_ms`` : compute how long to wait after
                          a frame's work to hold a target FPS.
  - ``ema``             : exponential moving average to smooth the HUD numbers.
"""

import time
from dataclasses import dataclass
from typing import Callable, Optional

# cv2.waitKey needs at least 1ms to pump the window and read a key, so pacing
# never asks it to wait less than this even when a frame blew its budget.
MIN_WAIT_MS = 1

# Default weight on the newest sample when smoothing HUD numbers. Small enough to
# keep the readout steady, large enough to still track real changes within ~1s.
DEFAULT_EMA_ALPHA = 0.2


class StageTimer:
    """Context manager measuring the wall-clock milliseconds spent in a block.

    Usage::

        with StageTimer() as t:
            do_work()
        print(t.ms)   # elapsed milliseconds (float)

    ``clock`` is injectable (defaults to ``time.perf_counter``) so tests can drive
    it with a fake clock and assert exact durations without sleeping.
    """

    def __init__(self, clock: Callable[[], float] = time.perf_counter) -> None:
        self._clock = clock
        self._start = 0.0
        self.ms = 0.0

    def __enter__(self) -> "StageTimer":
        self._start = self._clock()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.ms = (self._clock() - self._start) * 1000.0


@dataclass(frozen=True)
class StageTimings:
    """Smoothed per-stage millisecond timings for one loop iteration's HUD.

    ``capture_ms`` is the age of the webcam frame being shown (how stale the
    threaded capture is), not a blocking read cost; the other four are the time
    the tracking thread's last detect took, the composite stage, the sink stage,
    and the whole main-loop iteration (work only, excluding the pacing wait).
    """

    capture_ms: float = 0.0
    track_ms: float = 0.0
    composite_ms: float = 0.0
    sink_ms: float = 0.0
    loop_ms: float = 0.0


def frame_budget_seconds(target_fps: float) -> float:
    """Seconds available per frame at ``target_fps``. Raises on non-positive fps."""
    if target_fps <= 0:
        raise ValueError("target_fps must be positive")
    return 1.0 / target_fps


def pace_wait_ms(
    target_fps: float, work_seconds: float, min_ms: int = MIN_WAIT_MS
) -> int:
    """Milliseconds to wait after ``work_seconds`` of work to hold ``target_fps``.

    Returns ``max(min_ms, round((budget - work) * 1000))``: a fast frame waits the
    remainder of its budget, while a frame that already overran still yields
    ``min_ms`` (cv2.waitKey must get >= 1ms to render the window and read a key).
    Used only for the warm-up branch, where cv2.waitKey both waits and polls keys;
    the steady loop paces with ``remaining_sleep_seconds`` + a ``waitKey(1)``
    because cv2.waitKey's Windows timer granularity is too coarse to hit a target
    rate accurately (it rounds a requested wait up by ~10-15ms).
    """
    budget = frame_budget_seconds(target_fps)
    remaining_ms = round((budget - work_seconds) * 1000.0)
    return max(min_ms, remaining_ms)


def remaining_sleep_seconds(target_fps: float, elapsed_seconds: float) -> float:
    """Seconds to sleep after ``elapsed_seconds`` of work to hold ``target_fps``.

    Never negative: a frame that already overran its budget sleeps zero. Paired
    with a precise ``time.sleep`` (high-resolution on Python 3.11+/Windows) this
    holds the target rate far more accurately than handing the wait to cv2.waitKey.
    """
    return max(0.0, frame_budget_seconds(target_fps) - elapsed_seconds)


def ema(
    previous: float, sample: float, alpha: float = DEFAULT_EMA_ALPHA
) -> float:
    """Exponential moving average of ``sample`` into ``previous``.

    ``alpha`` (0 < alpha <= 1) weights the newest sample. Used to smooth the
    per-stage HUD numbers so they read steadily instead of flickering each frame.
    """
    return previous + alpha * (sample - previous)

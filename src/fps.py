"""Pure frame-rate counter, decoupled from any camera/display I/O so it's easy to test."""

import time

DEFAULT_REPORT_INTERVAL_SECONDS = 1.0


class FpsCounter:
    """Counts frames and reports an FPS figure no more often than once per interval.

    Call `tick()` once per processed frame. It returns the computed FPS (float)
    when the reporting interval has elapsed since the last report, or None
    otherwise (i.e. "not time to print yet").
    """

    def __init__(
        self,
        report_interval_seconds: float = DEFAULT_REPORT_INTERVAL_SECONDS,
        clock=time.perf_counter,
    ) -> None:
        if report_interval_seconds <= 0:
            raise ValueError("report_interval_seconds must be positive")
        self._interval = report_interval_seconds
        self._clock = clock
        self._frame_count = 0
        self._window_start = self._clock()

    def tick(self):
        """Record one frame. Returns FPS (float) if the interval elapsed, else None."""
        self._frame_count += 1
        now = self._clock()
        elapsed = now - self._window_start

        if elapsed < self._interval:
            return None

        fps = self._frame_count / elapsed
        self._frame_count = 0
        self._window_start = now
        return fps

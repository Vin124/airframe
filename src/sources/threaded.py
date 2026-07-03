"""Threaded wrapper that runs a blocking ``Source``'s read() off the main loop.

The webcam's ``cv2.VideoCapture.read()`` blocks until the next frame (~40-50ms at
20-25fps). Calling it inline throttles the whole pipeline to the camera rate,
and stacking ~40ms of detection on top halves it again. ``ThreadedSource`` moves
that blocking read onto its own daemon thread that publishes the newest frame
latest-frame-wins, so the main loop's ``read()`` returns immediately with the
current frame (or the previous one, or ``None`` before the first arrives) and
never stalls waiting on the camera.

Threading model (same discipline as ``sources/capture.py``):
    The capture thread owns the inner source and is its ONLY reader. Each new
    frame is published by assigning a single ``(seq, frame)`` tuple to
    ``self._packet`` -- one attribute store, atomic under the GIL, so a reader on
    the main or tracking thread sees either the previous complete packet or the
    next one, never a torn pair, and no lock is needed. ``cv2.VideoCapture.read``
    returns a fresh array per call and ``WebcamSource`` flips into a new array, so
    published frames are never aliased to a reused buffer. Consumers MUST treat a
    returned frame as READ-ONLY (copy before mutating in place); the compositor
    does exactly that (it draws on a private per-frame copy).

Lifecycle:
    ``open()`` opens the inner source SYNCHRONOUSLY first (so a camera-open
    failure still fails fast on the main thread, exactly like the pre-threaded
    loop) and only then starts the capture thread. ``close()`` signals the thread
    to stop, joins it with a timeout, then closes the inner source; it is
    idempotent and safe before ``open()``. A transient ``None`` read (camera
    hiccup) is swallowed by the thread rather than ending the session; if the
    camera dies mid-run the last frame stays frozen and ONE stderr notice is
    printed per death episode (see ``_DEAD_READS_BEFORE_NOTICE``) so a frozen
    preview is never silent.
"""

import sys
import threading
import time
from typing import Callable, Optional, Tuple

import numpy as np

from src.sources.base import Size, Source

# Sleep briefly when the inner read yields None (device not ready / hiccup) so
# the capture thread doesn't busy-spin at 100% CPU on a stalled device.
_IDLE_SLEEP_SECONDS = 0.005

# After this many CONSECUTIVE failed reads (~0.5s at the idle sleep), print one
# stderr notice that the camera has gone quiet -- a silently frozen preview is
# indistinguishable from a hang otherwise. The counter resets on any good frame,
# so a fresh notice marks each new death episode, mirroring the tracking
# thread's announce-once crash policy.
_DEAD_READS_BEFORE_NOTICE = 100

# How long close() waits for the capture thread to exit before giving up (a
# blocking read can take up to one frame interval to return and notice the stop).
_JOIN_TIMEOUT_SECONDS = 2.0

# (sequence number, frame) published as one atomic unit so readers get a
# consistent pair.
Packet = Tuple[int, np.ndarray]


class ThreadedSource(Source):
    """Run any blocking ``Source`` on a capture thread, publishing latest-frame-wins.

    Wraps an inner source (the webcam in this app). Satisfies the same pull-based
    ``Source`` contract: ``read()`` is non-blocking and returns the newest frame,
    the previous one if none is newer, or ``None`` before the first frame.
    """

    def __init__(
        self,
        inner: Source,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._inner = inner
        self._clock = clock
        self._packet: Optional[Packet] = None
        self._seq = 0
        self._published_at: Optional[float] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def open(self) -> "ThreadedSource":
        """Open the inner source (may raise, e.g. CameraError) then start the thread."""
        opener = getattr(self._inner, "open", None)
        if callable(opener):
            opener()
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="capture-thread", daemon=True
        )
        self._thread.start()
        return self

    def _run(self) -> None:
        """Capture loop (own thread): publish each new frame latest-frame-wins."""
        dead_reads = 0
        while not self._stop.is_set():
            frame = self._inner.read()
            if frame is None:
                dead_reads += 1
                if dead_reads == _DEAD_READS_BEFORE_NOTICE:
                    print(
                        "Warning: the webcam stopped delivering frames "
                        "(disconnected?); showing the last frame until it "
                        "recovers or you quit.",
                        file=sys.stderr,
                    )
                time.sleep(_IDLE_SLEEP_SECONDS)
                continue
            dead_reads = 0
            self._seq += 1
            self._published_at = self._clock()
            self._packet = (self._seq, frame)  # single atomic publish

    def read(self) -> Optional[np.ndarray]:
        """Return the newest published frame (READ-ONLY), or None before the first."""
        packet = self._packet
        return None if packet is None else packet[1]

    def latest_packet(self) -> Optional[Packet]:
        """Return the newest ``(seq, frame)`` packet, or None before the first.

        The tracking thread uses the ``seq`` to skip re-detecting a frame it has
        already processed, and the main loop uses it to copy a fresh frame only
        when a genuinely new one has arrived.
        """
        return self._packet

    def frame_age_ms(self) -> Optional[float]:
        """Milliseconds since the newest frame was published, or None if none yet.

        This is the HUD's ``capture`` figure: how stale the frame being shown is,
        which reflects the real camera cadence (the main-loop read itself is free).
        """
        published_at = self._published_at
        if published_at is None:
            return None
        return (self._clock() - published_at) * 1000.0

    @property
    def base_size(self) -> Optional[Size]:
        """Native (width, height), delegated to the inner source (non-consuming)."""
        return self._inner.base_size

    def close(self) -> None:
        """Stop + join the capture thread, then close the inner source. Idempotent.

        If the join times out (a native read wedged past the timeout), the inner
        source is deliberately NOT closed: releasing the camera under a thread
        still blocked inside cv2 risks a native use-after-free. The daemon
        thread and the device are reclaimed at process exit instead.
        """
        self._stop.set()
        thread = self._thread
        self._thread = None
        if thread is not None:
            thread.join(timeout=_JOIN_TIMEOUT_SECONDS)
            if thread.is_alive():
                print(
                    "Warning: capture thread didn't stop in time; leaving the "
                    "camera to be reclaimed at process exit.",
                    file=sys.stderr,
                )
                return
        self._inner.close()

    def __enter__(self) -> "ThreadedSource":
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

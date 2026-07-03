"""Live Windows capture sources: overlay a running window or a monitor.

``WindowSource`` captures a specific top-level window (matched by a title
substring) and ``MonitorSource`` captures a whole monitor (1-based index, with
an optional region crop), both via the Windows Graphics Capture API (the
``windows-capture`` package). Each satisfies the same pull-based ``Source``
contract as the webcam and image sources, so the rest of the pipeline
(``scene -> compositor -> sinks``) treats a live window exactly like a static
overlay image.

Threading model (latest-frame-wins):
    ``windows-capture`` runs capture on its own dedicated thread and invokes an
    ``on_frame_arrived`` callback per frame. The callback COPIES the frame -- the
    library hands back a numpy *view* over a reused native buffer (verified:
    ``frame_buffer.flags.OWNDATA is False``), so keeping the view would tear or
    crash -- into a fresh BGRA array and publishes it with a single attribute
    assignment. ``read()`` on the main loop returns the latest published frame.
    A single reference assignment is atomic under the GIL, so no lock is needed:
    a reader sees either the previous complete frame or the next complete frame,
    never a half-written one.

Lifecycle / graceful death:
    ``import windows_capture`` is deferred to ``open()`` (optional dependency;
    the base pipeline must run without it, same rationale as
    ``sinks/virtualcam.py``). A window that can't be found -- or a bad monitor
    index -- makes ``start_free_threaded()`` raise synchronously (verified
    empirically), which ``open()`` converts into a ``SourceError`` so ``main``
    fails fast before the camera opens, exactly like a missing image path. If
    the target closes mid-run the library fires ``on_closed``; that marks the
    source dead (``read()`` returns ``None`` thereafter, so the compositor skips
    the layer and the app keeps running) and prints one stderr notice.
    ``close()`` stops the capture thread and is idempotent.
"""

import sys
from typing import Callable, Optional, Protocol, Tuple

import cv2
import numpy as np

from src.sources.base import Size, Source, SourceError

# A captured layer is composited as a solid overlay: the alpha channel is forced
# fully opaque rather than trusting whatever DWM left there.
OPAQUE_ALPHA = 255

# Region crop expressed as (x, y, width, height) in captured pixels.
Region = Tuple[int, int, int, int]

# Factory that builds a windows-capture-compatible object. Typed loosely on
# purpose (the real type is ``windows_capture.WindowsCapture``, imported lazily);
# tests inject a fake so no real capture thread ever starts.
CaptureFactory = Callable[..., "_Capture"]


class _CaptureControl(Protocol):
    """The subset of ``windows_capture.CaptureControl`` this module uses."""

    def stop(self) -> None: ...


class _Capture(Protocol):
    """The subset of ``windows_capture.WindowsCapture`` this module uses."""

    def event(self, handler: Callable) -> Callable: ...

    def start_free_threaded(self) -> "_CaptureControl": ...


def frame_to_bgra(buffer: np.ndarray) -> np.ndarray:
    """Copy a capture frame buffer into a fresh, fully-opaque BGRA array.

    ``buffer`` is a numpy *view* over the library's reused native memory, so
    this always makes an independent copy (never aliases it). Non-4-channel
    input is promoted to BGRA, and the alpha channel is forced opaque (255) so a
    captured window/monitor composites as a solid overlay.
    """
    array = np.array(buffer, copy=True)  # detach from the reused native buffer
    if array.ndim == 2:
        array = cv2.cvtColor(array, cv2.COLOR_GRAY2BGRA)
    elif array.shape[2] == 3:
        array = cv2.cvtColor(array, cv2.COLOR_BGR2BGRA)
    array[:, :, 3] = OPAQUE_ALPHA
    return array


def crop_region(bgra: np.ndarray, region: Region) -> np.ndarray:
    """Return the ``(x, y, w, h)`` sub-rectangle of ``bgra``, clamped to bounds.

    Coordinates are clamped to the frame so an over-large or partly-off-frame
    region still yields a valid (at least 1x1) crop instead of raising or
    producing an empty array. The crop is materialized as an owned contiguous
    array (not a view) so everything published as a latest frame is uniformly
    independent of its parent buffer.
    """
    height, width = bgra.shape[:2]
    x, y, w, h = region
    x0 = max(0, min(x, width - 1))
    y0 = max(0, min(y, height - 1))
    x1 = max(x0 + 1, min(x + w, width))
    y1 = max(y0 + 1, min(y + h, height))
    return np.ascontiguousarray(bgra[y0:y1, x0:x1])


def _validate_region(region: Region) -> None:
    """Raise ``SourceError`` if ``region`` isn't a positive-size (x, y, w, h)."""
    if len(region) != 4:
        raise SourceError("Region must be (x, y, width, height).")
    x, y, w, h = region
    if w <= 0 or h <= 0:
        raise SourceError(f"Region width and height must be positive; got {w}x{h}.")
    if x < 0 or y < 0:
        raise SourceError(f"Region origin must be non-negative; got ({x}, {y}).")


class _CaptureSource(Source):
    """Shared latest-frame-wins store + lifecycle for the live capture sources.

    Subclasses only supply how to build the underlying ``windows-capture``
    object (``_capture_kwargs``), a human description (``_describe``), and the
    fail-fast error text (``_start_error_message``); all frame handling,
    threading, and teardown live here.
    """

    def __init__(
        self,
        region: Optional[Region] = None,
        *,
        capture_factory: Optional[CaptureFactory] = None,
        cursor_capture: bool = False,
    ) -> None:
        self._region = region
        self._capture_factory = capture_factory  # None -> real WindowsCapture at open()
        self._cursor_capture = cursor_capture
        self._latest: Optional[np.ndarray] = None
        self._size: Optional[Size] = None
        self._dead = False
        self._notified_dead = False
        self._control: Optional[_CaptureControl] = None

    # -- Source contract -------------------------------------------------- #
    def read(self) -> Optional[np.ndarray]:
        """Return the latest captured BGRA frame, or ``None``.

        ``None`` before the first frame arrives, and after the source is dead
        (target closed) or ``close()``d -- the compositor skips a ``None`` layer.
        """
        if self._dead:
            return None
        return self._latest

    @property
    def base_size(self) -> Optional[Size]:
        """Native (width, height) of the latest frame; ``None`` if none/dead.

        Cached from the last ingested frame (non-consuming, unlike the base
        default which would call ``read()``), so hit-testing and the initial fit
        never disturb the live feed.
        """
        if self._dead:
            return None
        return self._size

    def close(self) -> None:
        """Stop the capture thread. Idempotent and safe before ``open()``."""
        self._dead = True
        control = self._control
        self._control = None
        if control is not None:
            try:
                control.stop()
            except Exception:  # noqa: BLE001 - teardown must never raise
                pass

    # -- Lifecycle -------------------------------------------------------- #
    def open(self) -> "_CaptureSource":
        """Start the capture thread; raise ``SourceError`` on a bad target.

        The window-not-found / bad-monitor-index failure surfaces synchronously
        from ``start_free_threaded`` (verified empirically), so ``main`` can
        fail fast exactly like a missing image path.
        """
        try:
            capture = self._make_capture()
        except ImportError as exc:
            raise SourceError(
                "Could not start window/screen capture: the 'windows-capture' "
                "package is not installed. Run 'pip install windows-capture' "
                "and try again."
            ) from exc

        # These MUST be named on_frame_arrived / on_closed: windows-capture's
        # event() dispatches on the handler's __name__.
        def on_frame_arrived(frame, capture_control) -> None:
            self._ingest_frame(frame)

        def on_closed() -> None:
            self._mark_dead()

        capture.event(on_frame_arrived)
        capture.event(on_closed)

        try:
            self._control = capture.start_free_threaded()
        except Exception as exc:  # noqa: BLE001 - normalize any start failure
            raise SourceError(self._start_error_message(exc)) from exc
        return self

    def _make_capture(self) -> _Capture:
        """Build the capture object (injected factory in tests, real lib else)."""
        factory = self._capture_factory
        if factory is None:
            import windows_capture  # deferred: optional dependency

            factory = windows_capture.WindowsCapture
        return factory(**self._capture_kwargs())

    # -- Frame handling (runs on the capture thread) ---------------------- #
    def _ingest_frame(self, frame) -> None:
        """Publish ``frame`` as the latest BGRA frame (copies off native memory).

        Called on the library's capture thread. ``_size`` is set before
        ``_latest`` so that whenever a reader sees a frame, ``base_size`` already
        reflects it; the final ``_latest`` assignment is the single atomic
        publish (latest-frame-wins, no lock -- see the module docstring).
        """
        bgra = frame_to_bgra(frame.frame_buffer)
        if self._region is not None:
            bgra = crop_region(bgra, self._region)
        self._size = (bgra.shape[1], bgra.shape[0])
        self._latest = bgra

    def _mark_dead(self) -> None:
        """Mark the source dead when the target closes; warn once on stderr."""
        self._dead = True
        if not self._notified_dead:
            self._notified_dead = True
            print(
                f"Notice: {self._describe()} closed; its overlay is gone but the "
                "app keeps running.",
                file=sys.stderr,
            )

    # -- Subclass hooks --------------------------------------------------- #
    def _capture_kwargs(self) -> dict:
        raise NotImplementedError

    def _describe(self) -> str:
        raise NotImplementedError

    def _start_error_message(self, exc: Exception) -> str:
        raise NotImplementedError


class WindowSource(_CaptureSource):
    """Live capture of a top-level window matched by a title substring."""

    def __init__(
        self,
        title_substring: str,
        *,
        capture_factory: Optional[CaptureFactory] = None,
        cursor_capture: bool = False,
    ) -> None:
        if not title_substring or not str(title_substring).strip():
            raise SourceError("A window title substring is required to capture a window.")
        super().__init__(capture_factory=capture_factory, cursor_capture=cursor_capture)
        self._title = title_substring

    def _capture_kwargs(self) -> dict:
        return {"window_name": self._title, "cursor_capture": self._cursor_capture}

    def _describe(self) -> str:
        return f"window matching '{self._title}'"

    def _start_error_message(self, exc: Exception) -> str:
        return (
            f"Could not capture a window matching title '{self._title}'. No visible "
            "window title contained that text -- check the window is open and the "
            f"substring is correct. Underlying error: {exc}"
        )


class MonitorSource(_CaptureSource):
    """Live capture of a whole monitor (1-based index) or a region of it."""

    def __init__(
        self,
        monitor_index: int = 1,
        region: Optional[Region] = None,
        *,
        capture_factory: Optional[CaptureFactory] = None,
        cursor_capture: bool = False,
    ) -> None:
        if monitor_index < 1:
            raise SourceError(
                "Monitor index must be 1 or greater (1 = primary monitor); got "
                f"{monitor_index}."
            )
        if region is not None:
            _validate_region(region)
        super().__init__(
            region=region, capture_factory=capture_factory, cursor_capture=cursor_capture
        )
        self._monitor_index = monitor_index

    def _capture_kwargs(self) -> dict:
        return {"monitor_index": self._monitor_index, "cursor_capture": self._cursor_capture}

    def _describe(self) -> str:
        return f"monitor {self._monitor_index}"

    def _start_error_message(self, exc: Exception) -> str:
        return (
            f"Could not capture monitor {self._monitor_index} (1-based; monitor 1 is "
            f"the primary display). Underlying error: {exc}"
        )

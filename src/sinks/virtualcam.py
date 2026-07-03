"""Virtual camera sink: broadcast composited frames as a selectable webcam device.

Wraps ``pyvirtualcam.Camera`` behind the ``Sink`` contract so a video-call app
(Zoom, Teams, Meet) can pick the composited feed as its camera. ``pyvirtualcam``
doesn't create a virtual device itself -- on Windows it drives an existing OBS
Virtual Camera backend (installed with OBS Studio 26+), so construction fails
with ``RuntimeError`` when that backend isn't present. This sink translates
that into a ``SinkError`` carrying a concrete setup instruction, so ``main.py``
can show the user exactly what to do instead of a raw traceback.
"""

from typing import Callable, Optional, Tuple

import cv2
import numpy as np

from src.sinks.base import Sink, SinkError

# Target frame rate reported to the virtual camera device. This is a fixed
# hint/upper bound for pyvirtualcam (matching the pipeline's ~30fps target),
# not a measured live rate -- pyvirtualcam does not require frames to arrive
# at exactly this cadence.
DEFAULT_FPS = 30

# Factory producing a pyvirtualcam.Camera-compatible object. Typed loosely on
# purpose: ``pyvirtualcam`` is only imported inside ``_open()`` so the base
# pipeline (preview-only modes) runs even when the optional dependency isn't
# installed. Tests inject fakes here.
CameraFactory = Callable[..., object]


class VirtualCamSink(Sink):
    """Broadcasts composited BGR frames to a virtual webcam device.

    The underlying ``pyvirtualcam.Camera`` is lazily created on the FIRST
    ``send()`` call rather than in ``__init__``: its width/height are fixed at
    creation time and this sink doesn't know the pipeline's frame size until
    it sees a real frame. ``fps`` is a fixed constant (``DEFAULT_FPS``)
    matching the pipeline's target loop rate, not a measured rate.

    ``camera_factory`` defaults to ``pyvirtualcam.Camera`` and is
    constructor-injected the same way ``WebcamSource`` accepts an optional
    ``Camera`` -- so tests can pass a fake that never touches a real
    virtual-camera backend.
    """

    def __init__(
        self,
        fps: int = DEFAULT_FPS,
        camera_factory: Optional[CameraFactory] = None,
    ) -> None:
        self._fps = fps
        self._camera_factory = camera_factory  # None -> pyvirtualcam.Camera at open time
        self._cam = None
        self._size: Optional[Tuple[int, int]] = None  # (width, height) that fixed init

    def send(self, frame: np.ndarray) -> None:
        """Convert ``frame`` (BGR) to RGB and send it, opening the camera on first call.

        Raises ``SinkError`` if no virtual-camera backend is available, or if
        ``frame``'s size differs from the size that fixed initialization (the
        device can't be resized after creation; this is treated as an error
        rather than silently reinitializing, since a mid-session resolution
        change almost always means something upstream is misconfigured).

        No throttling (no ``sleep_until_next_frame()``) happens here: the
        preview sink's ``cv2.waitKey`` already paces the loop, and pyvirtualcam
        tolerates frames arriving at an uneven rate, so adding a second clock
        here would only add latency for no benefit.
        """
        height, width = frame.shape[:2]
        if self._cam is None:
            self._open(width, height)
        elif (width, height) != self._size:
            raise SinkError(
                f"VirtualCamSink was initialized at {self._size[0]}x{self._size[1]} "
                f"but received a {width}x{height} frame. The virtual camera "
                "device's resolution can't change after creation -- restart the "
                "app to pick up a new size."
            )

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            self._cam.send(rgb)
        except Exception as exc:  # pyvirtualcam raises assorted runtime errors
            # Wrap write-time failures (OBS stopped mid-session, backend
            # hiccup) too, so the main loop's single SinkError handler can
            # degrade gracefully instead of the raw error killing the session.
            raise SinkError(f"Virtual camera write failed: {exc}") from exc

    def _open(self, width: int, height: int) -> None:
        """Create the underlying camera device at (width, height, self._fps)."""
        try:
            # Deferred import: pyvirtualcam is only needed when the virtual cam
            # is actually opened, so preview-only runs work without it.
            import pyvirtualcam

            factory = self._camera_factory or pyvirtualcam.Camera
            self._cam = factory(
                width=width, height=height, fps=self._fps, fmt=pyvirtualcam.PixelFormat.RGB
            )
        except ImportError as exc:
            raise SinkError(
                "Could not start the virtual camera: the pyvirtualcam package "
                "is not installed. Run 'pip install pyvirtualcam' and try again."
            ) from exc
        except RuntimeError as exc:
            raise SinkError(
                "Could not start the virtual camera: no virtual-camera backend "
                "was found. Install OBS Studio 26+ (includes OBS Virtual Camera) "
                f"from https://obsproject.com and try again. Original error: {exc}"
            ) from exc
        self._size = (width, height)

    def close(self) -> None:
        """Close the underlying camera device.

        Safe to call when the camera was never initialized (e.g. no frame was
        ever sent), and safe to call more than once.
        """
        if self._cam is not None:
            self._cam.close()
            self._cam = None

"""Live webcam source: the pipeline's background feed.

Wraps the low-level ``camera.Camera`` device and adds the selfie-view mirror
flip, so everything downstream (hand tracking, compositing, display) sees one
consistent, already-mirrored frame -- exactly what the pre-refactor loop did by
calling ``cv2.flip`` right after ``camera.read()``.

Why the mirror lives here rather than in the loop: mirroring is a property of
*this* source (a front-facing webcam shown as a selfie), not of the pipeline.
Keeping it in the source means detection and display can never disagree about
orientation, and future non-mirrored sources (window capture) simply don't do
it.
"""

from typing import Optional

import cv2
import numpy as np

from src.camera import Camera
from src.sources.base import Size, Source

# cv2.flip code for a horizontal (selfie-view) flip.
MIRROR_FLIP_CODE = 1


class WebcamSource(Source):
    """A ``Source`` backed by a ``Camera``, returning mirrored BGR frames.

    Context-manager friendly so the camera opens on ``__enter__`` and releases
    on ``__exit__``, mirroring the old ``with Camera() as camera:`` lifecycle
    (and letting ``camera.CameraError`` propagate on open failure).

    Usage:
        with WebcamSource() as webcam:
            frame = webcam.read()   # mirrored BGR frame, or None on read failure
    """

    def __init__(self, camera: Optional[Camera] = None, mirror: bool = True) -> None:
        self._camera = camera if camera is not None else Camera()
        self._mirror = mirror
        self._last_size: Optional[Size] = None

    def open(self) -> "WebcamSource":
        """Open the underlying camera. Raises ``camera.CameraError`` on failure."""
        self._camera.open()
        return self

    def read(self) -> Optional[np.ndarray]:
        """Return the next mirrored BGR frame, or ``None`` if a frame can't be read."""
        frame = self._camera.read()
        if frame is None:
            return None
        if self._mirror:
            frame = cv2.flip(frame, MIRROR_FLIP_CODE)
        self._last_size = (frame.shape[1], frame.shape[0])
        return frame

    @property
    def base_size(self) -> Optional[Size]:
        """Size of the most recently read frame, or ``None`` before the first.

        Overrides the base default (which calls ``read()``) because consuming
        a live frame just to report a size would silently drop that frame for
        the pipeline.
        """
        return self._last_size

    def close(self) -> None:
        """Release the underlying camera device."""
        self._camera.release()

    def __enter__(self) -> "WebcamSource":
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

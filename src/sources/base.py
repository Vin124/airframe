"""Base ``Source`` contract for everything that produces frames.

A source is a pull-based frame producer: call ``read()`` to get the latest
frame as a numpy array, or ``None`` when no frame is currently available. This
single method is the whole contract the rest of the pipeline depends on, so a
static image, a live webcam, and (later) a threaded window-capture all look
identical downstream.

Frame format by role:
  - Background sources (e.g. the webcam) return BGR ``HxWx3`` arrays; the
    compositor paints them as the base layer.
  - Layer sources (e.g. an overlay image) return BGRA ``HxWx4`` arrays so the
    compositor can alpha-blend them over the background.

Why pull-based ``read()`` rather than a push callback: it keeps the MVP loop
dead simple (main loop pulls one frame per source per tick) while letting a
threaded source publish into a latest-frame-wins buffer that ``read()`` just
returns -- exactly what the live capture sources in ``capture.py`` do --
without changing any caller.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np

# (width, height) in pixels -- the native size of a layer source's frame.
Size = Tuple[int, int]


class SourceError(RuntimeError):
    """Raised when a source can't be initialized or read from.

    Kept generic so callers can catch one error type across every source
    (bad image path, camera open failure surfaces as ``camera.CameraError``,
    etc.).
    """


class Source(ABC):
    """A producer of frames. Subclasses implement ``read()``.

    Usage:
        frame = source.read()   # numpy array, or None if nothing is available

    ``base_size`` reports the native (width, height) of the source's frame and
    is used by layer geometry (hit-testing, compositing). The default derives
    it from the latest frame; image-like sources override it with a cheap,
    side-effect-free implementation.
    """

    @abstractmethod
    def read(self) -> Optional[np.ndarray]:
        """Return the latest frame as a numpy array, or ``None`` if unavailable."""

    @property
    def base_size(self) -> Optional[Size]:
        """Native (width, height) of the source's frame, or ``None`` if unknown.

        Default implementation reads the latest frame and reports its shape.
        Concrete layer sources override this to avoid re-reading a live feed.
        """
        frame = self.read()
        if frame is None:
            return None
        height, width = frame.shape[:2]
        return width, height

    def close(self) -> None:
        """Release any underlying resources. Default is a no-op."""

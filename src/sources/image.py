"""Static image source: a loaded BGRA image served as an overlay layer.

Loads an image from disk (adding an opaque alpha channel if the file has none)
and hands the same immutable BGRA array back on every ``read()``. This is the
layer whose pixels the compositor alpha-blends over the webcam background; the
image's position/scale live in a ``scene.Transform`` on the owning ``Layer``,
not here -- the source only owns pixels, never geometry.
"""

from typing import Optional

import cv2
import numpy as np

from src.sources.base import Size, Source, SourceError


def ensure_bgra(image: np.ndarray) -> np.ndarray:
    """Return ``image`` as an HxWx4 BGRA array, adding an opaque alpha if absent.

    Grayscale is promoted to BGR first; a BGR image gets a fully-opaque alpha
    channel. An image that already has 4 channels is returned unchanged.
    """
    if image.ndim == 2:  # grayscale -> BGR
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 3:
        alpha = np.full(image.shape[:2] + (1,), 255, dtype=image.dtype)
        image = np.concatenate([image, alpha], axis=2)
    return image


class ImageSource(Source):
    """A ``Source`` that always returns one loaded BGRA image.

    The image is immutable; callers thread a separate ``Transform`` through the
    compositor/interaction each frame to move and scale it.
    """

    def __init__(self, image_bgra: np.ndarray) -> None:
        self._image = ensure_bgra(image_bgra)

    @classmethod
    def load(cls, path: str) -> "ImageSource":
        """Load an image (with alpha) from ``path``.

        Raises ``SourceError`` with a clear message if the file is missing or
        is not a readable image. A missing alpha channel is filled in opaque.
        """
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise SourceError(
                f"Could not load overlay image at '{path}'. Check that the path "
                "exists and points to a readable image file."
            )
        return cls(image)

    def read(self) -> np.ndarray:
        """Return the loaded BGRA image (same array every call)."""
        return self._image

    @property
    def base_size(self) -> Size:
        """Native (width, height) of the image in pixels."""
        return self._image.shape[1], self._image.shape[0]

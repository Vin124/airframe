"""Preview sink: show composited frames in an OpenCV window.

Wraps ``cv2.imshow`` behind the ``Sink`` contract. Key polling stays in the
main loop (``cv2.waitKey``) rather than here, because the loop needs the key to
drive quit/toggle/scale actions; this sink is only responsible for putting the
frame on screen and tearing the window down on close.
"""

import cv2
import numpy as np

from src.sinks.base import Sink

DEFAULT_WINDOW_NAME = "handy"


class PreviewSink(Sink):
    """Display composited frames in a named OpenCV window."""

    def __init__(self, window_name: str = DEFAULT_WINDOW_NAME) -> None:
        self._window_name = window_name

    @property
    def window_name(self) -> str:
        return self._window_name

    def send(self, frame: np.ndarray) -> None:
        """Show ``frame`` in the window. A subsequent ``cv2.waitKey`` renders it."""
        cv2.imshow(self._window_name, frame)

    def close(self) -> None:
        """Destroy all OpenCV windows."""
        cv2.destroyAllWindows()

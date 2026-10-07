"""Preview sink: show composited frames in an OpenCV window.

Wraps ``cv2.imshow`` behind the ``Sink`` contract. Key polling stays in the
main loop (``cv2.waitKey``) rather than here, because the loop needs the key to
drive quit/toggle/scale actions; this sink is only responsible for putting the
frame on screen and tearing the window down on close.
"""

from typing import Tuple

import cv2
import numpy as np

from src.sinks.base import Sink

DEFAULT_WINDOW_NAME = "Airframe"

# Initial preview WINDOW cap (the frame itself stays full resolution). With
# native-resolution capture a 1080p+ autosized window would overflow a laptop
# screen; the window opens scaled to fit this box (aspect kept) and stays
# user-resizable (cv2.WINDOW_NORMAL scales the content to the window).
MAX_INITIAL_WINDOW_WIDTH = 1280
MAX_INITIAL_WINDOW_HEIGHT = 720


def initial_window_size(
    frame_width: int,
    frame_height: int,
    max_width: int = MAX_INITIAL_WINDOW_WIDTH,
    max_height: int = MAX_INITIAL_WINDOW_HEIGHT,
) -> Tuple[int, int]:
    """Scale (frame_width, frame_height) down to fit the cap, keeping aspect.

    A frame already within the cap keeps its size (never upscaled). Pure
    function so it's unit-testable without a display.
    """
    scale = min(max_width / frame_width, max_height / frame_height, 1.0)
    return max(1, round(frame_width * scale)), max(1, round(frame_height * scale))


class PreviewSink(Sink):
    """Display composited frames in a named OpenCV window.

    The window is created resizable on the first frame and starts scaled to
    fit ``MAX_INITIAL_WINDOW_*`` so a native-resolution camera doesn't open a
    window larger than the screen.
    """

    def __init__(self, window_name: str = DEFAULT_WINDOW_NAME) -> None:
        self._window_name = window_name
        self._window_created = False

    @property
    def window_name(self) -> str:
        return self._window_name

    def _create_window(self, frame: np.ndarray) -> None:
        """Create the resizable window sized to fit the screen (first frame)."""
        cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
        width, height = initial_window_size(frame.shape[1], frame.shape[0])
        cv2.resizeWindow(self._window_name, width, height)
        self._window_created = True

    def send(self, frame: np.ndarray) -> None:
        """Show ``frame`` in the window. A subsequent ``cv2.waitKey`` renders it."""
        if not self._window_created:
            self._create_window(frame)
        cv2.imshow(self._window_name, frame)

    def close(self) -> None:
        """Destroy all OpenCV windows."""
        cv2.destroyAllWindows()

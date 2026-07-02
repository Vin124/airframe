"""Webcam access wrapper: open the default camera, read frames, release cleanly."""

import cv2

DEFAULT_CAMERA_INDEX = 0


class CameraError(RuntimeError):
    """Raised when the webcam cannot be opened or read from."""


class Camera:
    """Context-manager wrapper around a cv2.VideoCapture device.

    Usage:
        with Camera() as cam:
            frame = cam.read()
    """

    def __init__(self, index: int = DEFAULT_CAMERA_INDEX) -> None:
        self._index = index
        self._capture = None

    def open(self) -> "Camera":
        """Open the webcam. Raises CameraError with a clear message on failure."""
        capture = cv2.VideoCapture(self._index)
        if not capture.isOpened():
            capture.release()
            raise CameraError(
                f"Could not open webcam at index {self._index}. "
                "Check that a camera is connected, drivers are installed, "
                "and no other application is using it."
            )
        self._capture = capture
        return self

    def read(self):
        """Read a single frame. Returns None if a frame could not be read."""
        if self._capture is None:
            raise CameraError("Camera is not open. Call open() or use it as a context manager.")
        ok, frame = self._capture.read()
        if not ok:
            return None
        return frame

    def release(self) -> None:
        """Release the underlying capture device, if open."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self) -> "Camera":
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()

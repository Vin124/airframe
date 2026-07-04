"""Webcam access wrapper: open the default camera, read frames, release cleanly."""

import cv2

DEFAULT_CAMERA_INDEX = 0

# Requested for BOTH width and height at open so the driver clamps to the
# camera's largest supported mode (its native resolution) -- far beyond any
# real sensor, so the closest supported mode IS the maximum one. Without this,
# Windows hands out a small default (typically 640x480) even on a 1080p camera.
_NATIVE_RESOLUTION_REQUEST = 10000


class CameraError(RuntimeError):
    """Raised when the webcam cannot be opened or read from."""


class Camera:
    """Context-manager wrapper around a cv2.VideoCapture device.

    Opens at the camera's NATIVE (maximum) resolution by default; pass
    ``native_resolution=False`` for the driver's small default mode.

    Usage:
        with Camera() as cam:
            frame = cam.read()
    """

    def __init__(
        self,
        index: int = DEFAULT_CAMERA_INDEX,
        native_resolution: bool = True,
    ) -> None:
        self._index = index
        self._native_resolution = native_resolution
        self._capture = None

    def open(self) -> "Camera":
        """Open the webcam. Raises CameraError with a clear message on failure.

        The resolution request MUST happen here, before the first ``read()``:
        the MSMF backend crashes on a mid-stream mode change (observed with
        OpenCV 5.0 on Windows 11), so the mode is negotiated once at open and
        never touched again.
        """
        capture = cv2.VideoCapture(self._index)
        if not capture.isOpened():
            capture.release()
            raise CameraError(
                f"Could not open webcam at index {self._index}. "
                "Check that a camera is connected, drivers are installed, "
                "and no other application is using it."
            )
        if self._native_resolution:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, _NATIVE_RESOLUTION_REQUEST)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, _NATIVE_RESOLUTION_REQUEST)
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"[camera] capturing at {width}x{height}")
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

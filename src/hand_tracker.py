"""MediaPipe hand tracking wrapper (Tasks API, VIDEO running mode).

Wraps `mediapipe.tasks.python.vision.HandLandmarker` and exposes a small,
MediaPipe-free surface: feed it a BGR frame, get back a list of hands, each
hand a list of 21 normalized ``(x, y, z)`` landmark tuples (empty if no hand
is detected). Callers never touch MediaPipe result objects.

The `.task` model is auto-downloaded into ``models/`` at the repo root on
first use if it isn't already present.
"""

import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import mediapipe as mp
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)

from src.tuning import TrackerConfidence

# 21 landmarks per hand, each an (x, y, z) triple in normalized [0, 1] space.
LANDMARKS_PER_HAND = 21

# Default to a single hand — enough for the MVP (see CLAUDE.md).
DEFAULT_NUM_HANDS = 1

# Official float16 hand-landmarker bundle.
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_FILENAME = "hand_landmarker.task"

# Repo root is the parent of the src/ package directory.
_REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = _REPO_ROOT / "models"

_DOWNLOAD_TIMEOUT_SECONDS = 60

# One landmark tuple and one hand, as returned to callers.
Landmark = Tuple[float, float, float]
Hand = List[Landmark]

# Builds a HandLandmarker from options; the real one is
# ``HandLandmarker.create_from_options``. Injected in tests so no real model
# loads and the GPU/CPU fallback path is exercised without a GPU.
LandmarkerFactory = Callable[[HandLandmarkerOptions], HandLandmarker]


class HandTrackerError(RuntimeError):
    """Raised when the model can't be obtained or the tracker can't initialize."""


def ensure_model(
    model_path: Path = MODELS_DIR / MODEL_FILENAME,
    url: str = MODEL_URL,
) -> Path:
    """Return a path to the local `.task` model, downloading it if missing.

    Raises HandTrackerError with a clear message on network failure or if the
    downloaded file is empty.
    """
    if model_path.exists() and model_path.stat().st_size > 0:
        return model_path

    model_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = model_path.with_suffix(model_path.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=_DOWNLOAD_TIMEOUT_SECONDS) as response:
            data = response.read()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise HandTrackerError(
            f"Failed to download the hand-tracking model from {url}: {exc}. "
            "Check your internet connection, or place the file manually at "
            f"{model_path}."
        ) from exc

    if not data:
        raise HandTrackerError(
            f"Downloaded model from {url} was empty. Delete any partial file at "
            f"{model_path} and try again."
        )

    tmp_path.write_bytes(data)
    tmp_path.replace(model_path)
    return model_path


def landmarks_to_hands(hand_landmarks_list) -> List[Hand]:
    """Convert a MediaPipe result's ``hand_landmarks`` into plain tuples.

    Pure function: takes the list-of-lists of landmark objects (each with
    ``.x``, ``.y``, ``.z``) and returns a list of hands, each a list of
    ``(x, y, z)`` float tuples. Kept MediaPipe-free so it's trivially testable
    with fake landmark objects.
    """
    hands: List[Hand] = []
    for hand in hand_landmarks_list:
        hands.append([(float(lm.x), float(lm.y), float(lm.z)) for lm in hand])
    return hands


def _build_options(
    model_path: Path,
    num_hands: int,
    delegate: "BaseOptions.Delegate",
    confidence: TrackerConfidence,
) -> HandLandmarkerOptions:
    """Build VIDEO-mode landmarker options for a specific compute delegate.

    ``confidence`` maps onto MediaPipe's three startup-only min-confidence knobs
    (detection / presence / tracking). These live in the options object, so a
    change needs a fresh landmarker -- hence they are CLI/config-resolved, never
    a live slider (see ``src.tuning.TrackerConfidence``).
    """
    return HandLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=str(model_path), delegate=delegate
        ),
        running_mode=RunningMode.VIDEO,
        num_hands=num_hands,
        min_hand_detection_confidence=confidence.detection,
        min_hand_presence_confidence=confidence.presence,
        min_tracking_confidence=confidence.tracking,
    )


def create_landmarker(
    factory: LandmarkerFactory,
    model_path: Path,
    num_hands: int,
    gpu: bool,
    confidence: TrackerConfidence = TrackerConfidence(),
) -> HandLandmarker:
    """Create the landmarker, honoring a GPU-delegate request with CPU fallback.

    When ``gpu`` is True, first try the GPU delegate; on ANY failure (the usual
    case -- MediaPipe's Python wheels ship no GPU delegate on Windows) print one
    clear line and fall back to the CPU delegate. A CPU-delegate failure is fatal
    (there's nothing left to fall back to) and surfaces as ``HandTrackerError``.

    ``confidence`` carries MediaPipe's startup-only min-confidence knobs into the
    options for whichever delegate is used.

    Chunk 3 deferral: this flag only requests MediaPipe's built-in GPU delegate.
    A full ONNX Runtime / TensorRT port of the hand model is intentionally NOT
    attempted here (see spec.md's out-of-scope notes).
    """
    if gpu:
        try:
            return factory(
                _build_options(model_path, num_hands, BaseOptions.Delegate.GPU, confidence)
            )
        except Exception as exc:  # noqa: BLE001 - any GPU init failure -> CPU
            print(
                f"GPU delegate unavailable ({exc}); falling back to CPU.",
                file=sys.stderr,
            )
    try:
        return factory(
            _build_options(model_path, num_hands, BaseOptions.Delegate.CPU, confidence)
        )
    except Exception as exc:  # MediaPipe raises broad RuntimeError subtypes here
        raise HandTrackerError(
            f"Failed to initialize the MediaPipe HandLandmarker: {exc}"
        ) from exc


class HandTracker:
    """Detect hands in webcam frames using the MediaPipe Tasks API (VIDEO mode).

    Usage:
        with HandTracker() as tracker:
            hands = tracker.detect(bgr_frame)  # list of hands, each 21 (x,y,z)

    ``gpu=True`` requests the GPU delegate and cleanly falls back to CPU with a
    one-line notice when it isn't available (see ``create_landmarker``).
    ``confidence`` sets MediaPipe's three startup-only min-confidence knobs
    (detection / presence / tracking); they bake into the landmarker, so they are
    CLI/config-resolved once at construction, not live-tunable.
    """

    def __init__(
        self,
        num_hands: int = DEFAULT_NUM_HANDS,
        model_path: Path = MODELS_DIR / MODEL_FILENAME,
        gpu: bool = False,
        landmarker_factory: Optional[LandmarkerFactory] = None,
        confidence: TrackerConfidence = TrackerConfidence(),
    ) -> None:
        resolved_model = ensure_model(model_path=model_path)
        factory = landmarker_factory or HandLandmarker.create_from_options
        self._landmarker = create_landmarker(
            factory, resolved_model, num_hands, gpu, confidence
        )

        # VIDEO mode requires strictly increasing timestamps (milliseconds).
        self._last_timestamp_ms = -1

    def _next_timestamp_ms(self) -> int:
        """Return a strictly increasing millisecond timestamp for VIDEO mode."""
        now_ms = int(time.monotonic() * 1000)
        if now_ms <= self._last_timestamp_ms:
            now_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = now_ms
        return now_ms

    def detect(self, frame_bgr) -> List[Hand]:
        """Detect hands in a BGR frame.

        Returns a list of hands (each 21 normalized ``(x, y, z)`` tuples), or an
        empty list if no hand is found.
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(mp_image, self._next_timestamp_ms())
        return landmarks_to_hands(result.hand_landmarks)

    def close(self) -> None:
        """Release the underlying MediaPipe landmarker."""
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def __enter__(self) -> "HandTracker":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

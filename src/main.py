"""Entry point: argument parsing, mode dispatch, and the main capture loop.

Skeleton mode runs live MediaPipe hand tracking, draws a 21-point skeleton
over the mirrored feed, and overlays a pinch/grab gesture debug label. Image
mode is still a passthrough (filled in Phase 3). Both modes mirror the feed,
show it in a window, print FPS, and quit on 'q'.
"""

import argparse
import sys

import cv2

from src.camera import Camera, CameraError
from src.fps import FpsCounter
from src.gestures import GestureDetector
from src.hand_tracker import HandTracker, HandTrackerError
from src.skeleton import draw_gesture_debug, draw_skeleton

WINDOW_NAME = "handy"
QUIT_KEY = "q"
WAIT_KEY_DELAY_MS = 1
MIRROR_FLIP_CODE = 1  # cv2.flip code for a horizontal (selfie-view) flip

MODE_SKELETON = "skeleton"
MODE_IMAGE = "image"
MODES = (MODE_SKELETON, MODE_IMAGE)


def parse_args(argv=None) -> argparse.Namespace:
    """Parse and validate CLI arguments. Exits via argparse on invalid input."""
    parser = argparse.ArgumentParser(description="Webcam hand-tracking debug harness.")
    parser.add_argument("--mode", required=True, choices=MODES, help="Which mode to run.")
    parser.add_argument(
        "--image", default=None, help="Path to the overlay image (required for --mode image)."
    )
    args = parser.parse_args(argv)

    if args.mode == MODE_IMAGE and not args.image:
        parser.error("--image is required when --mode is 'image'")

    return args


def process_frame(frame, args: argparse.Namespace, tracker, gesture_detector=None):
    """Per-mode frame-processing hook, run on the already-mirrored frame.

    Skeleton mode detects hands, draws the skeleton, and overlays a pinch/grab
    debug label + pinch-point dot. Image mode is a passthrough for now (image
    overlay lands in Phase 3). ``tracker`` is a HandTracker in skeleton mode,
    else None. ``gesture_detector`` is a GestureDetector in skeleton mode,
    else None; only the first detected hand is fed to it (single-hand MVP).
    """
    if args.mode == MODE_SKELETON and tracker is not None:
        hands = tracker.detect(frame)
        draw_skeleton(frame, hands)
        if gesture_detector is not None:
            first_hand = hands[0] if hands else None
            gesture_state = gesture_detector.update(first_hand)
            if first_hand is not None:
                draw_gesture_debug(frame, gesture_state)
    return frame


def run(args: argparse.Namespace) -> None:
    """Open the webcam and run the capture/display/quit loop until 'q' or failure."""
    tracker = None
    gesture_detector = None
    if args.mode == MODE_SKELETON:
        try:
            tracker = HandTracker()
        except HandTrackerError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
        gesture_detector = GestureDetector()

    try:
        with Camera() as camera:
            fps_counter = FpsCounter()
            while True:
                frame = camera.read()
                if frame is None:
                    print("Warning: failed to read frame from webcam.", file=sys.stderr)
                    break

                # Mirror first, then detect, so drawn landmarks align with the view.
                frame = cv2.flip(frame, MIRROR_FLIP_CODE)
                frame = process_frame(frame, args, tracker, gesture_detector)

                cv2.imshow(WINDOW_NAME, frame)

                fps = fps_counter.tick()
                if fps is not None:
                    print(f"FPS: {fps:.1f}")

                key = cv2.waitKey(WAIT_KEY_DELAY_MS) & 0xFF
                if key == ord(QUIT_KEY):
                    break
    except CameraError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        if tracker is not None:
            tracker.close()
        cv2.destroyAllWindows()


def main(argv=None) -> None:
    args = parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()

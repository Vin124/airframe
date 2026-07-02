"""Entry point: argument parsing, mode dispatch, and the main capture loop.

Phase 0: no hand tracking yet. Just opens the webcam, mirrors the feed,
shows it in a window, prints FPS, and quits on 'q'.
"""

import argparse
import sys

import cv2

from src.camera import Camera, CameraError
from src.fps import FpsCounter

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


def process_frame(frame, args: argparse.Namespace):
    """Per-mode frame-processing hook.

    Phase 0: passthrough only. Hand tracking / skeleton drawing (Phase 1) and
    image overlay (Phase 3) slot in here, branching on args.mode.
    """
    return frame


def run(args: argparse.Namespace) -> None:
    """Open the webcam and run the capture/display/quit loop until 'q' or failure."""
    try:
        with Camera() as camera:
            fps_counter = FpsCounter()
            while True:
                frame = camera.read()
                if frame is None:
                    print("Warning: failed to read frame from webcam.", file=sys.stderr)
                    break

                frame = cv2.flip(frame, MIRROR_FLIP_CODE)
                frame = process_frame(frame, args)

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
        cv2.destroyAllWindows()


def main(argv=None) -> None:
    args = parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()

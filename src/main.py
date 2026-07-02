"""Entry point: argument parsing, mode dispatch, and the main capture loop.

Skeleton mode runs live MediaPipe hand tracking, draws a 21-point skeleton
over the mirrored feed, and overlays a pinch/grab gesture debug label. Image
mode composites an image over the feed and lets you pinch-grab/move it, scale
it with +/- keys, and two-hand-pinch to resize (see ``src.overlay``). Both
modes mirror the feed, show it in a window, print FPS, draw a small HUD
(mode/FPS/gesture/key legend), and quit on 'q'.
"""

import argparse
import sys
from typing import Optional

import cv2

from src import hud
from src.camera import Camera, CameraError
from src.fps import FpsCounter
from src.gestures import GestureDetector
from src.hand_tracker import HandTracker, HandTrackerError
from src.overlay import InteractionState, Overlay, OverlayError, step
from src.skeleton import draw_gesture_debug, draw_skeleton

WINDOW_NAME = "handy"
QUIT_KEY = "q"
WAIT_KEY_DELAY_MS = 1
MIRROR_FLIP_CODE = 1  # cv2.flip code for a horizontal (selfie-view) flip

MODE_SKELETON = "skeleton"
MODE_IMAGE = "image"
MODES = (MODE_SKELETON, MODE_IMAGE)

# Number of hands to track in image mode (single hand grabs/moves; two hands
# pinch-resize).
IMAGE_MODE_NUM_HANDS = 2

# Keyboard scaling in image mode. '+' usually needs shift, so accept its
# unshifted twin '='; likewise '-' and '_'. One key press multiplies the scale
# by this step (clamped inside Transform).
KEYBOARD_SCALE_STEP = 1.1
SCALE_UP_KEYS = (ord("+"), ord("="))
SCALE_DOWN_KEYS = (ord("-"), ord("_"))

# Runtime toggle keys. 's' flips skeleton-overlay drawing on/off in both
# modes (skeleton mode: the skeleton + gesture debug label; image mode: same,
# gated the same way --debug-skeleton was). 'r' is image-mode only: resets
# the transform/grab/resize back to the initial centered state.
TOGGLE_SKELETON_KEY = ord("s")
RESET_TRANSFORM_KEY = ord("r")


def parse_args(argv=None) -> argparse.Namespace:
    """Parse and validate CLI arguments. Exits via argparse on invalid input."""
    parser = argparse.ArgumentParser(description="Webcam hand-tracking debug harness.")
    parser.add_argument("--mode", required=True, choices=MODES, help="Which mode to run.")
    parser.add_argument(
        "--image", default=None, help="Path to the overlay image (required for --mode image)."
    )
    parser.add_argument(
        "--debug-skeleton",
        action="store_true",
        help="In image mode, also draw the hand skeleton + gesture label.",
    )
    args = parser.parse_args(argv)

    if args.mode == MODE_IMAGE and not args.image:
        parser.error("--image is required when --mode is 'image'")

    return args


def process_frame(frame, args: argparse.Namespace, tracker, gesture_detector=None, draw_debug=True):
    """Skeleton-mode per-frame hook, run on the already-mirrored frame.

    Detects hands and always computes gesture state (needed for the HUD even
    when the skeleton overlay is toggled off); drawing the skeleton + debug
    label/pinch dot is gated on ``draw_debug`` so the 's' key can hide the
    overlay without stopping gesture tracking. ``tracker`` is a HandTracker
    and ``gesture_detector`` a GestureDetector; only the first detected hand
    is fed to it (single-hand MVP). Image mode is handled separately by
    ``_process_image_frame`` because it threads persistent interaction state.

    Returns ``(frame, gesture_state, hand_present)``; ``gesture_state`` is
    None when there's no gesture detector.
    """
    gesture_state = None
    hand_present = False
    if args.mode == MODE_SKELETON and tracker is not None:
        hands = tracker.detect(frame)
        hand_present = bool(hands)
        if draw_debug:
            draw_skeleton(frame, hands)
        if gesture_detector is not None:
            first_hand = hands[0] if hands else None
            gesture_state = gesture_detector.update(first_hand)
            if draw_debug and first_hand is not None:
                draw_gesture_debug(frame, gesture_state)
    return frame, gesture_state, hand_present


def _process_image_frame(frame, tracker, detectors, overlay, interaction, draw_debug):
    """Advance image mode one frame; composite the overlay; return updated state.

    Detects up to two hands and feeds hand slot ``i`` to ``detectors[i]``.
    NOTE: MediaPipe may swap hand order between frames, so a hand can jump
    detector slots; that's accepted MVP jitter (see CLAUDE.md). Drawing the
    skeleton + debug label is gated on ``draw_debug`` (the 's' toggle /
    --debug-skeleton flag) but gesture state is always computed for the HUD.

    Returns ``(interaction, gesture_states, hand_present)`` where
    ``hand_present`` reflects whether the primary hand (slot 0) was tracked
    this frame.
    """
    hands = tracker.detect(frame)
    gesture_states = [
        detector.update(hands[i] if i < len(hands) else None)
        for i, detector in enumerate(detectors)
    ]

    frame_size = (frame.shape[1], frame.shape[0])
    interaction = step(interaction, overlay, gesture_states, frame_size)
    overlay.composite(frame, interaction.transform)

    if draw_debug:
        draw_skeleton(frame, hands)
        if hands:
            draw_gesture_debug(frame, gesture_states[0])
    return interaction, gesture_states, bool(hands)


def _toggle_skeleton(key: int, show_skeleton: bool) -> bool:
    """Flip skeleton-overlay visibility on the 's' key; otherwise unchanged."""
    if key == TOGGLE_SKELETON_KEY:
        return not show_skeleton
    return show_skeleton


def _handle_image_key(key: int, interaction: InteractionState) -> InteractionState:
    """Apply the image-mode-only keys: '+/-' scale and 'r' transform reset.

    Reset returns a full ``InteractionState.initial()`` (transform recentered
    AND grab/resize idled) rather than just the transform, so a reset mid-grab
    can't leave a stale grab offset that would otherwise yank the image back
    out of center on the next tracked frame.
    """
    if key in SCALE_UP_KEYS:
        return interaction.with_transform(interaction.transform.scaled_by(KEYBOARD_SCALE_STEP))
    if key in SCALE_DOWN_KEYS:
        return interaction.with_transform(interaction.transform.scaled_by(1.0 / KEYBOARD_SCALE_STEP))
    if key == RESET_TRANSFORM_KEY:
        return InteractionState.initial()
    return interaction


def _init_mode(args: argparse.Namespace):
    """Build per-mode state BEFORE opening the camera.

    Returns ``(tracker, gesture_detector, detectors, overlay, interaction)``
    with the unused slots as None. Loads the overlay image first in image mode
    so a missing/unreadable file fails fast, before any camera or model work.
    Exits nonzero with a clear message on overlay/tracker init failure.
    """
    try:
        if args.mode == MODE_SKELETON:
            tracker = HandTracker()
            return tracker, GestureDetector(), None, None, None

        overlay = Overlay.load(args.image)  # fail fast on a bad image path
        tracker = HandTracker(num_hands=IMAGE_MODE_NUM_HANDS)
        detectors = [GestureDetector() for _ in range(IMAGE_MODE_NUM_HANDS)]
        return tracker, None, detectors, overlay, InteractionState.initial()
    except (HandTrackerError, OverlayError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


def run(args: argparse.Namespace) -> None:
    """Open the webcam and run the capture/display/quit loop until 'q' or failure.

    ``show_skeleton`` starts True in skeleton mode (skeleton is the point of
    that mode) and from --debug-skeleton in image mode (off by default
    there); the 's' key flips it at runtime in both modes. ``last_fps`` holds
    the most recent FpsCounter reading for the HUD, since ``tick()`` only
    returns a fresh value once per reporting interval and None otherwise.
    """
    tracker, gesture_detector, detectors, overlay, interaction = _init_mode(args)

    show_skeleton = True if args.mode == MODE_SKELETON else args.debug_skeleton
    last_fps: Optional[float] = None

    try:
        with Camera() as camera:
            fps_counter = FpsCounter()
            while True:
                frame = camera.read()
                if frame is None:
                    print(
                        "Warning: failed to read frame from webcam (stream ended "
                        "or camera disconnected). Stopping.",
                        file=sys.stderr,
                    )
                    break

                # Mirror first, then detect, so drawn landmarks align with the view.
                frame = cv2.flip(frame, MIRROR_FLIP_CODE)
                if args.mode == MODE_SKELETON:
                    frame, gesture_state, hand_present = process_frame(
                        frame, args, tracker, gesture_detector, show_skeleton
                    )
                    status = None
                else:
                    interaction, gesture_states, hand_present = _process_image_frame(
                        frame, tracker, detectors, overlay, interaction, show_skeleton
                    )
                    gesture_state = gesture_states[0] if gesture_states else None
                    status = hud.resize_or_grab_status(
                        interaction.grab.is_grabbed, interaction.resize.is_resizing
                    )

                fps = fps_counter.tick()
                if fps is not None:
                    print(f"FPS: {fps:.1f}")
                    last_fps = fps

                label = hud.gesture_label(
                    is_grab=bool(gesture_state is not None and gesture_state.is_grab),
                    is_pinching=bool(gesture_state is not None and gesture_state.is_pinching),
                    hand_present=hand_present,
                )
                hud.draw_hud(frame, hud.build_hud_lines(args.mode, last_fps, label, status))

                cv2.imshow(WINDOW_NAME, frame)

                key = cv2.waitKey(WAIT_KEY_DELAY_MS) & 0xFF
                if key == ord(QUIT_KEY):
                    break
                show_skeleton = _toggle_skeleton(key, show_skeleton)
                if args.mode == MODE_IMAGE:
                    interaction = _handle_image_key(key, interaction)
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

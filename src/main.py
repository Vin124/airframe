"""Entry point: argument parsing, mode dispatch, and the main pipeline loop.

The app is a small pipeline: a webcam ``Source`` produces the background frame,
a ``Scene`` of ``Layer`` objects is composited over it, and the result goes to a
preview ``Sink``. Hand tracking + gesture detection drive interaction on the
selected layer.

Skeleton mode runs live MediaPipe hand tracking over an empty scene, drawing a
21-point skeleton and a pinch/grab gesture debug label on the mirrored feed.
Image mode adds one image ``Layer`` you can pinch-grab/move, scale with +/-
keys, and two-hand-pinch to resize (see ``src.interaction``). Both modes mirror
the feed (in ``WebcamSource``), show it via ``PreviewSink``, print FPS, draw a
small HUD (mode/FPS/gesture/key legend), and quit on 'q'. Optionally (``
--virtualcam`` or the 'v' key) the same composited+HUD frame is also broadcast
to a virtual webcam via ``VirtualCamSink`` so it can be selected as a camera in
Zoom/Teams/Meet.
"""

import argparse
import sys
from typing import List, Optional, Tuple

import cv2

from src import compositor, hud
from src.camera import CameraError
from src.fps import FpsCounter
from src.gestures import GestureDetector, GestureState
from src.hand_tracker import HandTracker, HandTrackerError
from src.interaction import InteractionState, step
from src.scene import Layer, Scene, Transform
from src.sinks import PreviewSink, SinkError, VirtualCamSink
from src.skeleton import draw_gesture_debug, draw_skeleton
from src.sources import ImageSource, SourceError, WebcamSource

WINDOW_NAME = "handy"
QUIT_KEY = "q"
WAIT_KEY_DELAY_MS = 1

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
# the transform/grab/resize back to the initial centered state. 'v' flips the
# virtual-camera sink on/off live, independent of mode.
TOGGLE_SKELETON_KEY = ord("s")
RESET_TRANSFORM_KEY = ord("r")
TOGGLE_VIRTUALCAM_KEY = ord("v")


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
    parser.add_argument(
        "--virtualcam",
        action="store_true",
        help=(
            "Also broadcast the composited frame to a virtual webcam (OBS "
            "Virtual Camera backend required) so it can be selected as a "
            "camera in Zoom/Teams/Meet."
        ),
    )
    args = parser.parse_args(argv)

    if args.mode == MODE_IMAGE and not args.image:
        parser.error("--image is required when --mode is 'image'")

    return args


def _detect_gestures(
    tracker: HandTracker, frame, detectors: List[GestureDetector]
) -> Tuple[list, List[GestureState]]:
    """Detect hands and feed hand slot ``i`` to ``detectors[i]``.

    Returns ``(hands, gesture_states)`` where ``gesture_states`` is one per
    detector (slot 0 = primary hand). NOTE: MediaPipe may swap hand order
    between frames, so a hand can jump detector slots; that's accepted MVP
    jitter (see CLAUDE.md).
    """
    hands = tracker.detect(frame)
    gesture_states = [
        detector.update(hands[i] if i < len(hands) else None)
        for i, detector in enumerate(detectors)
    ]
    return hands, gesture_states


def _draw_debug(frame, hands, gesture_state: Optional[GestureState]) -> None:
    """Draw the skeleton and (if a hand is present) the gesture debug label."""
    draw_skeleton(frame, hands)
    if hands and gesture_state is not None:
        draw_gesture_debug(frame, gesture_state)


def _toggle_skeleton(key: int, show_skeleton: bool) -> bool:
    """Flip skeleton-overlay visibility on the 's' key; otherwise unchanged."""
    if key == TOGGLE_SKELETON_KEY:
        return not show_skeleton
    return show_skeleton


def _toggle_virtualcam(key: int, active: bool) -> bool:
    """Flip virtual-cam broadcasting on the 'v' key; otherwise unchanged."""
    if key == TOGGLE_VIRTUALCAM_KEY:
        return not active
    return active


def _pump_virtualcam(
    virtualcam: VirtualCamSink,
    frame,
    active: bool,
    ever_sent: bool,
    requested_at_launch: bool,
) -> Tuple[bool, bool]:
    """Send ``frame`` to the virtual cam when active; degrade gracefully on failure.

    Returns the updated ``(active, ever_sent)`` pair. A ``SinkError`` on the
    very first launch-requested send is fatal (``exit(1)``, matching
    ``_init_mode``'s fail-fast-before-real-work behavior); any later failure --
    or a failure of a runtime 'v' toggle that was never requested at launch --
    prints to stderr and turns the toggle off, keeping the live session alive.
    """
    if not active:
        return active, ever_sent
    try:
        virtualcam.send(frame)
        return True, True
    except SinkError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        if requested_at_launch and not ever_sent:
            sys.exit(1)
        return False, ever_sent


def _handle_image_key(
    key: int, scene: Scene, interaction: InteractionState
) -> Tuple[Scene, InteractionState]:
    """Apply the image-mode-only keys: '+/-' scale and 'r' transform reset.

    Scaling multiplies the selected layer's transform. Reset recenters the
    selected layer AND idles grab/resize (returns a fresh ``InteractionState``)
    rather than just the transform, so a reset mid-grab can't leave a stale grab
    offset that would otherwise yank the image back out of center on the next
    tracked frame. No-op (returns the inputs) when nothing is selected.
    """
    layer = scene.selected_layer
    if layer is None:
        return scene, interaction

    if key in SCALE_UP_KEYS:
        scaled = layer.with_transform(layer.transform.scaled_by(KEYBOARD_SCALE_STEP))
        return scene.replace_selected(scaled), interaction
    if key in SCALE_DOWN_KEYS:
        scaled = layer.with_transform(layer.transform.scaled_by(1.0 / KEYBOARD_SCALE_STEP))
        return scene.replace_selected(scaled), interaction
    if key == RESET_TRANSFORM_KEY:
        reset = layer.with_transform(Transform())
        return scene.replace_selected(reset), InteractionState.initial()
    return scene, interaction


def _init_mode(args: argparse.Namespace):
    """Build per-mode state BEFORE opening the camera.

    Returns ``(tracker, detectors, scene, interaction)``. Skeleton mode gets a
    single detector and an empty scene (no image layer, so the compositor is a
    no-op); ``interaction`` is None there. Image mode loads the overlay image
    first -- so a missing/unreadable file fails fast, before any camera or model
    work -- and builds a single-layer selected scene. Exits nonzero with a clear
    message on image/tracker init failure.
    """
    try:
        if args.mode == MODE_SKELETON:
            tracker = HandTracker()
            return tracker, [GestureDetector()], Scene(), None

        source = ImageSource.load(args.image)  # fail fast on a bad image path
        tracker = HandTracker(num_hands=IMAGE_MODE_NUM_HANDS)
        detectors = [GestureDetector() for _ in range(IMAGE_MODE_NUM_HANDS)]
        scene = Scene.single(Layer(source=source, transform=Transform()))
        return tracker, detectors, scene, InteractionState.initial()
    except (HandTrackerError, SourceError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


def run(args: argparse.Namespace) -> None:
    """Open the webcam and run the capture/composite/display/quit loop.

    ``show_skeleton`` starts True in skeleton mode (skeleton is the point of
    that mode) and from --debug-skeleton in image mode (off by default there);
    the 's' key flips it at runtime in both modes. ``last_fps`` holds the most
    recent FpsCounter reading for the HUD, since ``tick()`` only returns a fresh
    value once per reporting interval and None otherwise.

    ``virtualcam`` is always constructed (cheap; it does nothing until a frame
    is sent) so ``close()`` in ``finally`` is always safe to call.
    ``virtualcam_active`` starts from ``--virtualcam`` and flips on the 'v'
    key. ``virtualcam_ever_sent`` distinguishes a failure on the very first
    launch-time attempt (fatal -- exit(1), matching ``_init_mode``'s
    fail-fast-before-real-work behavior) from a failure after it already
    worked once, or from a runtime ('v' key) toggle that was never requested
    at launch (both non-fatal: print + turn the toggle back off, keep the
    session alive).
    """
    tracker, detectors, scene, interaction = _init_mode(args)

    show_skeleton = True if args.mode == MODE_SKELETON else args.debug_skeleton
    last_fps: Optional[float] = None
    preview = PreviewSink(WINDOW_NAME)

    virtualcam = VirtualCamSink()
    virtualcam_active = args.virtualcam
    virtualcam_ever_sent = False

    try:
        with WebcamSource() as webcam:
            fps_counter = FpsCounter()
            while True:
                frame = webcam.read()  # already mirrored (selfie view)
                if frame is None:
                    print(
                        "Warning: failed to read frame from webcam (stream ended "
                        "or camera disconnected). Stopping.",
                        file=sys.stderr,
                    )
                    break

                hands, gesture_states = _detect_gestures(tracker, frame, detectors)
                gesture_state = gesture_states[0]
                hand_present = bool(hands)

                if args.mode == MODE_IMAGE:
                    frame_size = (frame.shape[1], frame.shape[0])
                    # Image mode always keeps exactly one layer selected; the
                    # guard makes that invariant explicit rather than crashing
                    # inside step() if it's ever broken.
                    layer = scene.selected_layer
                    if layer is not None:
                        interaction, layer = step(interaction, layer, gesture_states, frame_size)
                        scene = scene.replace_selected(layer)
                    status = hud.resize_or_grab_status(
                        interaction.grab.is_grabbed, interaction.resize.is_resizing
                    )
                else:
                    status = None

                # Composite the scene over the webcam background (empty scene in
                # skeleton mode = no-op), then draw debug overlays on top.
                compositor.render(frame, scene)
                if show_skeleton:
                    _draw_debug(frame, hands, gesture_state)

                fps = fps_counter.tick()
                if fps is not None:
                    print(f"FPS: {fps:.1f}")
                    last_fps = fps

                label = hud.gesture_label(
                    is_grab=bool(gesture_state is not None and gesture_state.is_grab),
                    is_pinching=bool(gesture_state is not None and gesture_state.is_pinching),
                    hand_present=hand_present,
                )
                hud.draw_hud(
                    frame,
                    hud.build_hud_lines(
                        args.mode, last_fps, label, status, virtualcam_active=virtualcam_active
                    ),
                )

                # Both sinks are fed the SAME composited+HUD frame. The HUD is
                # intentionally visible in the virtual-cam feed for this debug
                # harness; hiding it there specifically is future polish.
                preview.send(frame)
                virtualcam_active, virtualcam_ever_sent = _pump_virtualcam(
                    virtualcam, frame, virtualcam_active, virtualcam_ever_sent, args.virtualcam
                )

                key = cv2.waitKey(WAIT_KEY_DELAY_MS) & 0xFF
                if key == ord(QUIT_KEY):
                    break
                show_skeleton = _toggle_skeleton(key, show_skeleton)
                virtualcam_active = _toggle_virtualcam(key, virtualcam_active)
                if args.mode == MODE_IMAGE:
                    scene, interaction = _handle_image_key(key, scene, interaction)
    except CameraError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        if tracker is not None:
            tracker.close()
        preview.close()
        virtualcam.close()


def main(argv=None) -> None:
    args = parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()

"""Entry point: argument parsing, mode dispatch, and the main pipeline loop.

The app is a small pipeline: a webcam ``Source`` produces the background frame,
a ``Scene`` of ``Layer`` objects is composited over it, and the result goes to a
preview ``Sink``. Hand tracking + gesture detection drive interaction on the
selected layer.

Skeleton mode runs live MediaPipe hand tracking over an empty scene, drawing a
21-point skeleton and a pinch/grab gesture debug label on the mirrored feed.
Image mode starts with one overlay ``Layer`` you can pinch-grab/move, scale with
+/- keys, and two-hand-pinch to resize (see ``src.interaction``). Its source is
chosen by ``--source``: a static image (default), a live app ``window`` captured
by title, or a monitor ``screen`` (Windows Graphics Capture). At runtime you can
add more layers ('a'/'w'/'m' for image/window/screen), remove the selected layer
('x'), and bring it to the front ('f'); a pinch grabs the top-most layer under
the pinch point, and the selected layer is outlined. Both modes mirror the feed
(in ``WebcamSource``), show it via ``PreviewSink``, print FPS, draw a small HUD
(mode/FPS/gesture/key legend), and quit on 'q'. Optionally (``--virtualcam`` or
the 'v' key) the same composited+HUD frame is also broadcast to a virtual webcam
via ``VirtualCamSink`` so it can be selected as a camera in Zoom/Teams/Meet.
"""

import argparse
import sys
import time
from typing import Callable, Optional, Set, Tuple

import cv2
import numpy as np

from src import compositor, hud
from src.camera import CameraError
from src.fps import FpsCounter
from src.gestures import GestureDetector, GestureState
from src.hand_tracker import HandTracker, HandTrackerError
from src.interaction import InteractionState, step
from src.scene import DEFAULT_SCALE, Layer, Scene, Transform
from src.sinks import PreviewSink, SinkError, VirtualCamSink
from src.skeleton import draw_gesture_debug, draw_skeleton
from src.sources import (
    ImageSource,
    MonitorSource,
    Source,
    SourceError,
    ThreadedSource,
    WebcamSource,
    WindowSource,
)
from src.timing import (
    StageTimer,
    StageTimings,
    ema,
    pace_wait_ms,
    remaining_sleep_seconds,
)
from src.tracking import TrackingResult, TrackingWorker

WINDOW_NAME = "handy"
QUIT_KEY = "q"

# Main-loop display target. Capture and hand tracking run on their own threads at
# their natural rates; the loop composites + displays paced to this via a
# post-work cv2.waitKey (which also renders the window and reads keys). See
# src.timing.pace_wait_ms and PipelineRuntime._run_frame.
TARGET_FPS = 30

MODE_SKELETON = "skeleton"
MODE_IMAGE = "image"
MODES = (MODE_SKELETON, MODE_IMAGE)

# Image-mode overlay source. 'image' is a static file; 'window' captures a live
# app window by title substring; 'screen' captures a monitor (Windows Graphics
# Capture). Only meaningful in image mode -- skeleton mode has no layers.
SOURCE_IMAGE = "image"
SOURCE_WINDOW = "window"
SOURCE_SCREEN = "screen"
SOURCES = (SOURCE_IMAGE, SOURCE_WINDOW, SOURCE_SCREEN)
LIVE_SOURCES = (SOURCE_WINDOW, SOURCE_SCREEN)

# windows-capture indexes monitors 1-based (1 = primary); index 0 is rejected.
DEFAULT_MONITOR_INDEX = 1

# A captured window/monitor is usually far larger than the 640x480 webcam frame;
# at scale 1.0 it would bury the feed. On the first frame its size is known, fit
# its width to this fraction of the frame so the webcam stays visible.
INITIAL_FIT_FRACTION = 0.5

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

# Image-mode multi-layer keys (Phase 8). 'a'/'w'/'m' add an image / window /
# screen layer, reusing the launch args (--image / --window-title /
# --monitor-index) for the new source; each new layer lands on top and becomes
# selected. 'x' removes the selected layer (selecting the new top-most); 'f'
# brings the selected layer to the front. Chosen to avoid the keys above.
ADD_IMAGE_KEY = ord("a")
ADD_WINDOW_KEY = ord("w")
ADD_SCREEN_KEY = ord("m")
REMOVE_LAYER_KEY = ord("x")
BRING_TO_FRONT_KEY = ord("f")


def parse_args(argv=None) -> argparse.Namespace:
    """Parse and validate CLI arguments. Exits via argparse on invalid input."""
    parser = argparse.ArgumentParser(description="Webcam hand-tracking debug harness.")
    parser.add_argument("--mode", required=True, choices=MODES, help="Which mode to run.")
    parser.add_argument(
        "--image",
        default=None,
        help="Path to the overlay image (required for --mode image --source image).",
    )
    parser.add_argument(
        "--source",
        default=SOURCE_IMAGE,
        choices=SOURCES,
        help=(
            "Image-mode overlay source: a static 'image', a live 'window' (by "
            "title), or a monitor 'screen'. Only valid with --mode image."
        ),
    )
    parser.add_argument(
        "--window-title",
        default=None,
        help="Window title substring to capture (required for --source window).",
    )
    parser.add_argument(
        "--monitor-index",
        type=int,
        default=DEFAULT_MONITOR_INDEX,
        help="1-based monitor index for --source screen (1 = primary display).",
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
    parser.add_argument(
        "--gpu",
        action="store_true",
        help=(
            "Request the MediaPipe GPU delegate for hand tracking. Falls back to "
            "CPU with a one-line notice when it isn't available (the usual case "
            "for MediaPipe's Python wheels on Windows)."
        ),
    )
    args = parser.parse_args(argv)

    if args.source in LIVE_SOURCES and args.mode != MODE_IMAGE:
        parser.error("--source window/screen is only valid with --mode image")
    if args.mode == MODE_IMAGE and args.source == SOURCE_IMAGE and not args.image:
        parser.error("--image is required when --mode is 'image' and --source is 'image'")
    if args.source == SOURCE_WINDOW and not args.window_title:
        parser.error("--window-title is required when --source is 'window'")

    return args


def _draw_debug(frame: np.ndarray, hands, gesture_state: Optional[GestureState]) -> None:
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


def _build_layer_source(args: argparse.Namespace) -> Source:
    """Build and open the image-mode overlay source for the selected --source.

    Returns an already-open ``Source``. Fails fast with ``SourceError`` on a bad
    image path, a window whose title isn't found, or a bad monitor index -- the
    live sources start their capture thread here, so this is the last step in
    ``_init_mode`` that can fail (nothing risky follows the thread start).
    """
    if args.source == SOURCE_WINDOW:
        return WindowSource(args.window_title).open()
    if args.source == SOURCE_SCREEN:
        return MonitorSource(args.monitor_index).open()
    return ImageSource.load(args.image)  # fail fast on a bad image path


def _apply_pending_fit(
    scene: Scene, fit_ids: Set[int], frame_size: Tuple[int, int]
) -> Tuple[Scene, Set[int]]:
    """Apply the one-time initial fit of each pending live layer once sized.

    ``fit_ids`` is the set of ``layer_id``s still awaiting their initial fit
    (a live window/screen layer reports no size until its first captured frame
    arrives). Returns the (possibly updated) scene and the remaining pending
    set. A layer whose size is now known is fitted and dropped from the set; one
    whose size is still unknown stays pending and is retried next frame; one no
    longer in the scene (removed before its first frame) is dropped. Tracking by
    id -- not "the selected layer" -- means a layer added then deselected still
    gets fitted. Extracted from the loop so the exactly-once semantics are
    unit-testable without a camera.
    """
    if not fit_ids:
        return scene, fit_ids
    remaining = set(fit_ids)
    for layer_id in fit_ids:
        layer = scene.layer_by_id(layer_id)
        if layer is None:
            remaining.discard(layer_id)
            continue
        fitted = _fit_layer_to_frame(layer, frame_size)
        if fitted is None:
            continue
        scene = scene.update_layer(layer_id, fitted)
        remaining.discard(layer_id)
    return scene, remaining


def _initial_fit_ids(args: argparse.Namespace, scene: Scene) -> Set[int]:
    """Layer ids needing a one-time initial fit at launch (live sources only).

    A launch-time live window/screen layer reports no size until its first frame,
    so it is fitted lazily in the loop; a static image layer already knows its
    size and keeps the historical scale-1.0 behavior (no fit pending).
    """
    if args.mode != MODE_IMAGE or args.source not in LIVE_SOURCES:
        return set()
    return {layer.layer_id for layer in scene.layers}


def _build_image_layer_source(args: argparse.Namespace) -> Source:
    """Build an image overlay source for the 'a' key, reusing ``--image``.

    Raises ``SourceError`` when no ``--image`` path was provided (so the caller
    can print a notice and keep running) or when the path can't be loaded.
    """
    if not args.image:
        raise SourceError(
            "Cannot add an image layer: no --image path was provided at launch."
        )
    return ImageSource.load(args.image)


def _build_window_layer_source(args: argparse.Namespace) -> Source:
    """Build a live window source for the 'w' key, reusing ``--window-title``.

    Raises ``SourceError`` when no ``--window-title`` was provided, or (from
    ``open()``) when no visible window matches it.
    """
    if not args.window_title:
        raise SourceError(
            "Cannot add a window layer: no --window-title was provided at launch."
        )
    return WindowSource(args.window_title).open()


def _build_monitor_layer_source(args: argparse.Namespace) -> Source:
    """Build a live monitor source for the 'm' key, reusing ``--monitor-index``.

    Raises ``SourceError`` (from ``open()``) on a bad monitor index.
    """
    return MonitorSource(args.monitor_index).open()


# Dispatch for the add-layer keys: key -> (source builder, is-live-source). Live
# sources need a lazy initial fit; image sources already know their size.
_ADD_LAYER_BUILDERS = {
    ADD_IMAGE_KEY: (_build_image_layer_source, False),
    ADD_WINDOW_KEY: (_build_window_layer_source, True),
    ADD_SCREEN_KEY: (_build_monitor_layer_source, True),
}


def _add_layer(
    scene: Scene,
    interaction: InteractionState,
    fit_ids: Set[int],
    build_source: Callable[[], Source],
    live: bool,
) -> Tuple[Scene, InteractionState, Set[int]]:
    """Add a new top-most, selected layer from ``build_source``; degrade on failure.

    A ``SourceError`` (missing arg, or a window that can't be found) is caught,
    printed to stderr, and the scene is left unchanged -- adding a layer is a
    runtime action, so a failure must NOT exit (same principle as the virtual-cam
    'v' toggle). On success the new layer is selected and interaction is reset to
    idle so an in-progress grab on the previously-selected layer can't leak onto
    the new one; a live layer's id joins ``fit_ids`` for its one-time fit.
    """
    try:
        source = build_source()
    except SourceError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return scene, interaction, fit_ids
    scene = scene.add_layer(Layer(source=source))
    if live:
        fit_ids = fit_ids | {scene.selected_layer.layer_id}
    return scene, InteractionState.initial(), fit_ids


def _remove_selected_layer(
    scene: Scene, interaction: InteractionState, fit_ids: Set[int]
) -> Tuple[Scene, InteractionState, Set[int]]:
    """Remove the selected layer, close its source, and reset interaction.

    No-op (returns the inputs) when nothing is selected. The removed layer's
    source is per-layer, so it's ``close()``d here (stops a live capture thread;
    a no-op for an image). Interaction is reset to idle so a grab in progress on
    the removed layer doesn't leak onto the newly-selected top-most layer;
    ``remove_selected`` selects that new top (or leaves an empty scene if it was
    the last layer). The freed id drops out of ``fit_ids``.
    """
    layer = scene.selected_layer
    if layer is None:
        return scene, interaction, fit_ids
    layer.source.close()
    fit_ids = fit_ids - {layer.layer_id}
    return scene.remove_selected(), InteractionState.initial(), fit_ids


def _handle_layer_keys(
    key: int,
    args: argparse.Namespace,
    scene: Scene,
    interaction: InteractionState,
    fit_ids: Set[int],
) -> Tuple[Scene, InteractionState, Set[int]]:
    """Apply the image-mode multi-layer keys: add (a/w/m), remove (x), front (f).

    Returns the (possibly updated) ``(scene, interaction, fit_ids)``. Any other
    key is a no-op. Kept out of ``run`` so the multi-layer transitions are
    unit-testable without a camera or a real capture source.
    """
    builder = _ADD_LAYER_BUILDERS.get(key)
    if builder is not None:
        build_source, live = builder
        return _add_layer(scene, interaction, fit_ids, lambda: build_source(args), live)
    if key == REMOVE_LAYER_KEY:
        return _remove_selected_layer(scene, interaction, fit_ids)
    if key == BRING_TO_FRONT_KEY:
        return scene.raise_selected(), interaction, fit_ids
    return scene, interaction, fit_ids


def _fit_layer_to_frame(layer: Layer, frame_size: Tuple[int, int]) -> Optional[Layer]:
    """Return ``layer`` rescaled to ~``INITIAL_FIT_FRACTION`` of the frame width.

    Returns ``None`` when the layer's size isn't known yet (a live capture
    source reports no size until its first frame arrives) so the caller can try
    again next frame. Only the scale changes; the center is left at the default.

    The fit multiplies into the layer's current scale (relative to
    ``DEFAULT_SCALE``) rather than overwriting it, so '+'/'-' presses made
    while the capture was still warming up survive the fit landing instead of
    being silently discarded.
    """
    base = layer.base_size
    if base is None:
        return None
    base_w, _ = base
    if base_w <= 0:
        return None
    frame_w, _ = frame_size
    fit_scale = (frame_w * INITIAL_FIT_FRACTION) / base_w
    manual_ratio = layer.transform.scale / DEFAULT_SCALE
    return layer.with_transform(layer.transform.with_scale(fit_scale * manual_ratio))


def _init_mode(args: argparse.Namespace):
    """Build per-mode state BEFORE opening the camera.

    Returns ``(tracker, detectors, scene, interaction)``. Skeleton mode gets a
    single detector and an empty scene (no image layer, so the compositor is a
    no-op); ``interaction`` is None there. Image mode builds tracker/detectors
    first, then opens the overlay source LAST (a bad image path, unfound window,
    or bad monitor index fails fast here -- and, for live sources, the capture
    thread only starts once nothing risky follows), then a single-layer selected
    scene. Exits nonzero with a clear message on source/tracker init failure.
    """
    try:
        if args.mode == MODE_SKELETON:
            tracker = HandTracker(gpu=args.gpu)
            return tracker, [GestureDetector()], Scene(), None

        tracker = HandTracker(num_hands=IMAGE_MODE_NUM_HANDS, gpu=args.gpu)
        detectors = [GestureDetector() for _ in range(IMAGE_MODE_NUM_HANDS)]
        source = _build_layer_source(args)  # last step that can fail
        scene = Scene.single(Layer(source=source, transform=Transform()))
        return tracker, detectors, scene, InteractionState.initial()
    except (HandTrackerError, SourceError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


class PipelineRuntime:
    """Owns the per-frame main-loop state and drives one composite/display tick.

    Phase 9 decouples the pipeline into three threads:
      - a CAPTURE thread (``ThreadedSource`` around the webcam) publishing frames
        latest-frame-wins,
      - a TRACKING thread (``TrackingWorker``) detecting hands + advancing gesture
        hysteresis at its own cadence and publishing results latest-wins,
      - this MAIN loop, which is the SINGLE WRITER of the scene/interaction and
        composites + displays paced to ``TARGET_FPS``.

    This object lives on the main thread only. It reads the two threads' latest
    published values (never mutating their state) and threads the immutable
    scene/interaction forward each frame. It's split out of ``run`` so the loop
    body is a handful of small named stages and ``run`` stays a thin
    setup/teardown shell; the stages are unit-testable with fakes.

    ``show_skeleton`` starts True in skeleton mode and from --debug-skeleton in
    image mode; 's' flips it live. ``virtualcam_ever_sent`` distinguishes a fatal
    first-attempt virtual-cam failure from a graceful later one (see
    ``_pump_virtualcam``). Stage timings are EMA-smoothed and shown one frame late
    (the snapshot updates after the HUD is drawn).
    """

    def __init__(
        self,
        args: argparse.Namespace,
        webcam: ThreadedSource,
        worker: TrackingWorker,
        scene: Scene,
        interaction: Optional[InteractionState],
        preview: PreviewSink,
        virtualcam: VirtualCamSink,
    ) -> None:
        self._args = args
        self._webcam = webcam
        self._worker = worker
        self._preview = preview
        self._virtualcam = virtualcam
        self.scene = scene
        self.interaction = interaction
        self.show_skeleton = True if args.mode == MODE_SKELETON else args.debug_skeleton
        self.virtualcam_active = args.virtualcam
        self._virtualcam_ever_sent = False
        self._fit_ids = _initial_fit_ids(args, scene)
        self._fps_counter = FpsCounter()
        self._last_fps: Optional[float] = None
        self._clean_frame = None  # newest webcam frame (shared, read-only)
        self._clean_seq = -1
        self._status: Optional[str] = None
        self._timings = StageTimings()

    def run(self) -> None:
        """Loop until 'q' (or the window is closed), pacing each tick to TARGET_FPS."""
        while self._run_frame():
            pass

    def _run_frame(self) -> bool:
        """One main-loop iteration. Returns False to quit.

        Does the frame's work, pumps the window + reads a key with a single
        ``cv2.waitKey(1)``, then paces to ``TARGET_FPS`` by sleeping the remainder
        of the budget (a precise ``time.sleep`` -- cv2.waitKey's Windows timer is
        too coarse to pace accurately). ``loop_ms`` is the work + key-pump time
        (excludes the pacing sleep); timings are recorded for the NEXT frame's HUD.
        """
        frame_start = time.perf_counter()
        frame = self._next_frame()
        if frame is None:
            # No webcam frame yet (threads warming up): pace + poll keys via waitKey.
            return self._handle_keys(cv2.waitKey(pace_wait_ms(TARGET_FPS, 0.0)) & 0xFF)

        tracking = self._worker.latest()
        frame_size = (frame.shape[1], frame.shape[0])
        self._advance_scene(tracking, frame_size)
        composite_ms = self._composite(frame, tracking)
        self._update_fps()
        self._draw_hud(frame, tracking)
        sink_ms = self._emit(frame)

        key = cv2.waitKey(1) & 0xFF  # render the window + read the key
        elapsed = time.perf_counter() - frame_start
        time.sleep(remaining_sleep_seconds(TARGET_FPS, elapsed))
        self._update_timings(tracking, composite_ms, sink_ms, elapsed * 1000.0)
        return self._handle_keys(key)

    def _next_frame(self) -> Optional[np.ndarray]:
        """Return a fresh, mutable compositing copy of the current webcam frame.

        The capture thread publishes each frame as a shared, READ-ONLY array; the
        compositor draws in place, so we hand back a private per-frame COPY (never
        mutating the shared frame the tracking thread also reads). A new copy is
        taken only when the camera has published a new sequence, but a fresh copy
        of the clean frame is returned every tick so drawings never accumulate.
        Reuses the last clean frame when no newer one exists (render can repeat a
        frame so interaction/HUD stay live); returns ``None`` only before the very
        first frame arrives.
        """
        packet = self._webcam.latest_packet()
        if packet is not None and packet[0] != self._clean_seq:
            self._clean_seq = packet[0]
            self._clean_frame = packet[1]  # shared, treated read-only
        if self._clean_frame is None:
            return None
        return self._clean_frame.copy()

    def _advance_scene(self, tracking: TrackingResult, frame_size: Tuple[int, int]) -> None:
        """Apply the one-time live-layer fit + advance interaction (image mode only).

        Single writer of scene/interaction (main thread). Runs the initial fit
        BEFORE ``step`` so grab/move/resize act on the fitted layer from the
        start; ``step`` resolves selection + advances the active layer and
        tolerates an empty scene. Skeleton mode has no layers, so this only clears
        the status.
        """
        if self._args.mode != MODE_IMAGE:
            self._status = None
            return
        self.scene, self._fit_ids = _apply_pending_fit(self.scene, self._fit_ids, frame_size)
        self.interaction, self.scene = step(
            self.interaction, self.scene, list(tracking.gesture_states), frame_size
        )
        self._status = hud.resize_or_grab_status(
            self.interaction.grab.is_grabbed, self.interaction.resize.is_resizing
        )

    def _composite(self, frame: np.ndarray, tracking: TrackingResult) -> float:
        """Composite the scene + selection highlight (+ optional skeleton); return ms.

        Empty scene in skeleton mode = a no-op composite; the skeleton + gesture
        debug label is drawn on top when enabled, using the tracking thread's
        latest hands/gesture (possibly frozen if tracking crashed).
        """
        with StageTimer() as timer:
            compositor.render(frame, self.scene)
            compositor.draw_selection_highlight(frame, self.scene)
            if self.show_skeleton:
                _draw_debug(frame, list(tracking.hands), self._primary_gesture(tracking))
        return timer.ms

    def _update_fps(self) -> None:
        """Tick the FPS counter (loop rate) and print/remember a fresh reading."""
        fps = self._fps_counter.tick()
        if fps is not None:
            print(f"FPS: {fps:.1f}")
            self._last_fps = fps

    def _draw_hud(self, frame: np.ndarray, tracking: TrackingResult) -> None:
        """Draw the HUD strip: mode/FPS/gesture/status/vcam + last frame's timings."""
        gesture_state = self._primary_gesture(tracking)
        label = hud.gesture_label(
            is_grab=bool(gesture_state is not None and gesture_state.is_grab),
            is_pinching=bool(gesture_state is not None and gesture_state.is_pinching),
            hand_present=bool(tracking.hands),
        )
        timing_line = hud.build_timing_line(
            self._timings.capture_ms,
            self._timings.track_ms,
            self._timings.composite_ms,
            self._timings.sink_ms,
            self._timings.loop_ms,
        )
        hud.draw_hud(
            frame,
            hud.build_hud_lines(
                self._args.mode,
                self._last_fps,
                label,
                self._status,
                virtualcam_active=self.virtualcam_active,
                timing_line=timing_line,
            ),
        )

    def _emit(self, frame: np.ndarray) -> float:
        """Feed the composited+HUD frame to the preview + virtual cam; return ms.

        Both sinks get the SAME frame. A fatal first-attempt virtual-cam failure
        exits; a later failure degrades gracefully (see ``_pump_virtualcam``).
        """
        with StageTimer() as timer:
            self._preview.send(frame)
            self.virtualcam_active, self._virtualcam_ever_sent = _pump_virtualcam(
                self._virtualcam,
                frame,
                self.virtualcam_active,
                self._virtualcam_ever_sent,
                self._args.virtualcam,
            )
        return timer.ms

    def _update_timings(
        self, tracking: TrackingResult, composite_ms: float, sink_ms: float, loop_ms: float
    ) -> None:
        """EMA-smooth this frame's per-stage timings for the next frame's HUD."""
        capture_ms = self._webcam.frame_age_ms() or 0.0
        prev = self._timings
        self._timings = StageTimings(
            capture_ms=ema(prev.capture_ms, capture_ms),
            track_ms=ema(prev.track_ms, tracking.track_ms),
            composite_ms=ema(prev.composite_ms, composite_ms),
            sink_ms=ema(prev.sink_ms, sink_ms),
            loop_ms=ema(prev.loop_ms, loop_ms),
        )

    def _handle_keys(self, key: int) -> bool:
        """Apply runtime keys (quit/toggles/image keys); return False to quit."""
        if key == ord(QUIT_KEY):
            return False
        self.show_skeleton = _toggle_skeleton(key, self.show_skeleton)
        self.virtualcam_active = _toggle_virtualcam(key, self.virtualcam_active)
        if self._args.mode == MODE_IMAGE:
            self.scene, self.interaction = _handle_image_key(key, self.scene, self.interaction)
            self.scene, self.interaction, self._fit_ids = _handle_layer_keys(
                key, self._args, self.scene, self.interaction, self._fit_ids
            )
        return True

    @staticmethod
    def _primary_gesture(tracking: TrackingResult) -> Optional[GestureState]:
        """The primary hand's gesture state (slot 0), or None if none published."""
        return tracking.gesture_states[0] if tracking.gesture_states else None


def run(args: argparse.Namespace) -> None:
    """Wire the threaded pipeline, run it, and tear everything down on exit.

    Capture (webcam) and hand tracking each run on their OWN thread publishing
    latest-frame-wins; this main thread only composites + displays + handles keys,
    paced to ``TARGET_FPS`` -- so a slow camera or a ~40ms detect no longer stalls
    render (see ``PipelineRuntime`` / ``src.tracking.TrackingWorker`` /
    ``src.sources.threaded.ThreadedSource`` for the threading model). The webcam
    opens synchronously first (fail-fast on CameraError) before its thread starts.
    """
    tracker, detectors, scene, interaction = _init_mode(args)
    preview = PreviewSink(WINDOW_NAME)
    virtualcam = VirtualCamSink()
    webcam = ThreadedSource(WebcamSource())
    worker: Optional[TrackingWorker] = None
    runtime: Optional[PipelineRuntime] = None

    try:
        webcam.open()  # opens the camera (fail-fast) then starts the capture thread
        worker = TrackingWorker(tracker, detectors, webcam.latest_packet)
        worker.start()
        runtime = PipelineRuntime(args, webcam, worker, scene, interaction, preview, virtualcam)
        runtime.run()
    except CameraError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        # stop() reports whether the tracking thread really exited; if a native
        # detect() wedged past the join timeout, freeing the tracker under the
        # still-running thread risks a native use-after-free, so skip close()
        # and let process exit reclaim it (same policy as ThreadedSource.close).
        worker_stopped = worker.stop() if worker is not None else True
        webcam.close()
        if tracker is not None:
            if worker_stopped:
                tracker.close()
            else:
                print(
                    "Warning: tracking thread didn't stop in time; leaving the "
                    "hand tracker to be reclaimed at process exit.",
                    file=sys.stderr,
                )
        preview.close()
        virtualcam.close()
        # Release any layer sources (stops live window/screen capture threads; a
        # no-op for image layers). Use the runtime's final scene so layers added
        # at runtime are also closed; fall back to the initial scene if the
        # runtime never got built. Empty in skeleton mode.
        final_scene = runtime.scene if runtime is not None else scene
        for layer in final_scene.layers:
            layer.source.close()


def main(argv=None) -> None:
    args = parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()

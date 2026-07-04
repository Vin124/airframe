"""Entry point: argument parsing, mode dispatch, and the main pipeline loop.

The app is a small pipeline: a webcam ``Source`` produces the background frame, a
``Scene`` of ``Layer`` objects is composited over it, and the result goes to a
preview ``Sink``. Hand tracking + gesture detection drive interaction on the
selected layer. Skeleton mode draws a 21-point skeleton; image mode adds overlay
layers you pinch-grab/move/resize/add/remove/raise, air-draw (draw mode), and
fist-remove. Both mirror the feed, draw a HUD, print FPS, quit on 'q', and can
broadcast to a virtual cam. Scene/source-layer transitions live in ``src.layers``
and the launch glue in ``src.launch``; both re-exported below for the tests.
"""

import argparse
import sys
import time
from typing import TYPE_CHECKING, Callable, Optional, Tuple

import cv2
import numpy as np

from src import compositor, config, drawing, hud
from src.drawing import (  # air-drawing keys, re-exported for tests
    CLEAR_STROKE_KEY, CYCLE_COLOR_KEY, DRAW_MODE_KEY,
    STROKE_WIDTH_DOWN_KEY, STROKE_WIDTH_UP_KEY,
)
from src.camera import CameraError
from src.fps import FpsCounter
from src.gestures import (
    PINCH_ENTER_THRESHOLD,
    PINCH_EXIT_THRESHOLD,
    GestureDetector,
    GestureState,
)
from src.hand_tracker import HandTracker, HandTrackerError
from src.interaction import InteractionState, step
from src.scene import Layer, Scene, Transform
from src.sinks import PreviewSink, SinkError, VirtualCamSink
from src.skeleton import draw_gesture_debug, draw_skeleton
from src.speech import speech_session
from src.sources import (
    ArraySource,
    Source,
    SourceError,
    ThreadedSource,
    WebcamSource,
)
from src.timing import (
    StageTimer,
    StageTimings,
    ema,
    pace_wait_ms,
    remaining_sleep_seconds,
)
from src.tracking import TrackingResult, TrackingWorker
from src.tuning import (
    DEFAULT_MEDIAPIPE_CONFIDENCE,
    GestureTuning,
    TrackerConfidence,
    clamp_tuning,
)

# Re-exported for the test import surface (tests import these from src.main); the
# implementations + the mode/source/layer-key vocabulary moved to src.layers /
# src.launch in the Phase 5-10 extraction, and most are also used below. Tests
# monkeypatch the capture classes at src.layers.WindowSource / MonitorSource.
from src.layers import (  # noqa: F401  (re-exported for tests / used below)
    ADD_IMAGE_KEY,
    ADD_SCREEN_KEY,
    ADD_WINDOW_KEY,
    BRING_TO_FRONT_KEY,
    DEFAULT_MONITOR_INDEX,
    INITIAL_FIT_FRACTION,
    LIVE_SOURCES,
    MODE_IMAGE,
    MODE_SKELETON,
    MODES,
    REMOVE_LAYER_KEY,
    SOURCE_IMAGE,
    SOURCE_SCREEN,
    SOURCE_WINDOW,
    SOURCES,
    _add_layer,
    _add_source_layer,
    _apply_pending_fit,
    _build_layer_source,
    _close_layer_sources,
    _fit_layer_to_frame,
    _handle_layer_keys,
    _image_source_from_path,
    _initial_fit_ids,
    _monitor_source_from_index,
    _remove_selected_layer,
    _window_source_from_title,
    fist_removal_target,
)
from src.launch import (  # noqa: F401  (re-exported for tests / used below)
    _build_panel,
    _config_from_args,
    _final_settings,
    build_speech_on_phrase,
    confidence_from_prefill,
    parse_args,
    resolve_panel_prefill,
    tuning_from_prefill,
)
from src.spawn import (  # noqa: F401  (Phase 15 spawn gesture; re-exported for tests)
    PreviewInbox, PreviewState, advance_spawn, dock_preview, render_dock,
    suppress_spawn_pinch,
)

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids Tk import at runtime
    from src.panel import ControlPanel

WINDOW_NAME = "handy"
QUIT_KEY = "q"

# Main-loop display target. Capture and hand tracking run on their own threads at
# their natural rates; the loop composites + displays paced to this via a
# post-work cv2.waitKey (which also renders the window and reads keys). See
# src.timing.pace_wait_ms and PipelineRuntime._run_frame.
TARGET_FPS = 30

# Number of hands to track in image mode (single hand grabs/moves; two hands
# pinch-resize).
IMAGE_MODE_NUM_HANDS = 2

# Consecutive main-loop ticks a fist must be held over a layer before it's
# removed (~0.4 s at TARGET_FPS) -- a deliberate hold, not a momentary fist.
FIST_REMOVE_FRAMES = 12

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

    Returns the updated ``(active, ever_sent)`` pair. A ``SinkError`` on the very
    first launch-requested send is fatal (``exit(1)``); any later failure -- or a
    runtime 'v' toggle never requested at launch -- warns and turns the toggle off.
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

    Scaling multiplies the selected layer's transform. Reset recenters it AND
    idles grab/resize (a fresh ``InteractionState``) so a reset mid-grab can't
    leave a stale offset. No-op when nothing is selected.
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


def _init_mode(
    args: argparse.Namespace,
    tuning: Optional[GestureTuning] = None,
    confidence: Optional[TrackerConfidence] = None,
):
    """Build per-mode state BEFORE opening the camera.

    Returns ``(tracker, detectors, scene, interaction)``. Skeleton mode gets a
    single detector and an empty scene (no image layer, so the compositor is a
    no-op); ``interaction`` is None there. Image mode builds tracker/detectors
    first, then opens the overlay source LAST (a bad image path, unfound window,
    or bad monitor index fails fast here -- and, for live sources, the capture
    thread only starts once nothing risky follows), then a single-layer selected
    scene. Exits nonzero with a clear message on source/tracker init failure.

    ``tuning`` (resolved pinch thresholds) is threaded into every
    ``GestureDetector`` so the persisted/config value applies from frame one, not
    only after a slider nudge; ``confidence`` carries the startup-only MediaPipe
    min-confidence knobs into the tracker. Both default to the built-ins when
    omitted (keeps direct callers/tests simple).
    """
    tuning = tuning if tuning is not None else GestureTuning(
        PINCH_ENTER_THRESHOLD, PINCH_EXIT_THRESHOLD
    )
    confidence = confidence if confidence is not None else TrackerConfidence()
    try:
        if args.mode == MODE_SKELETON:
            tracker = HandTracker(gpu=args.gpu, confidence=confidence)
            detector = GestureDetector(tuning.enter, tuning.exit)
            return tracker, [detector], Scene(), None

        tracker = HandTracker(
            num_hands=IMAGE_MODE_NUM_HANDS, gpu=args.gpu, confidence=confidence
        )
        detectors = [
            GestureDetector(tuning.enter, tuning.exit)
            for _ in range(IMAGE_MODE_NUM_HANDS)
        ]
        source = _build_layer_source(args)  # last step that can fail
        scene = Scene.single(Layer(source=source, transform=Transform()))
        return tracker, detectors, scene, InteractionState.initial()
    except (HandTrackerError, SourceError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


class PipelineRuntime:
    """Owns the per-frame main-loop state and drives one composite/display tick.

    Phase 9 decouples the pipeline into three threads: a CAPTURE thread
    (``ThreadedSource``) publishing frames latest-wins, a TRACKING thread
    (``TrackingWorker``) publishing hand + gesture results latest-wins, and this
    MAIN loop -- the SINGLE WRITER of scene/interaction -- which composites +
    displays paced to ``TARGET_FPS``. It lives on the main thread only, reading the
    two threads' latest published values (never mutating them). Split out of
    ``run`` so the loop body is small named stages, unit-testable with fakes.
    ``show_skeleton`` starts from mode/--debug-skeleton ('s' flips it);
    ``virtualcam_ever_sent`` gates the fatal-first vs graceful-later cam failure.
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
        virtualcam_active: Optional[bool] = None,
        tuning: Optional[GestureTuning] = None,
        inbox: Optional[PreviewInbox] = None,
    ) -> None:
        self._args = args
        self._webcam = webcam
        self._worker = worker
        self._preview = preview
        self._virtualcam = virtualcam
        # Phase 15: speech->main hand-off slot (None = no spawn path) + the docked
        # preview/spawn state, owned solely by this main loop.
        self._inbox = inbox
        self._preview_state = PreviewState()
        self.scene = scene
        self.interaction = interaction
        # Last-applied pinch tuning (runtime-authoritative, driven by the sliders);
        # run() passes the resolved startup value matching the detectors.
        self.tuning = tuning if tuning is not None else GestureTuning(
            PINCH_ENTER_THRESHOLD, PINCH_EXIT_THRESHOLD
        )
        # Skeleton overlay defaults ON in both modes during development (toggle
        # with 's'); --debug-skeleton is kept for parity but no longer needed.
        self.show_skeleton = args.mode == MODE_SKELETON or args.mode == MODE_IMAGE or args.debug_skeleton
        # Initial virtual-cam state: resolved CLI-or-config value. The fatal-on-
        # first-failure check in _pump_virtualcam still keys off args.virtualcam, so
        # a config-driven ON degrades gracefully like a runtime 'v' toggle.
        self.virtualcam_active = args.virtualcam if virtualcam_active is None else virtualcam_active
        self._virtualcam_ever_sent = False
        self._fit_ids = _initial_fit_ids(args, scene)
        self._fps_counter = FpsCounter()
        self._last_fps: Optional[float] = None
        self._clean_frame = None  # newest webcam frame (shared, read-only)
        self._clean_seq = -1
        self._status: Optional[str] = None
        self._panel_status: Optional[str] = None  # last panel-action message
        self._panel = None  # optional Tk ControlPanel, pumped once per tick
        self._timings = StageTimings()
        # Air-drawing (main-loop-owned): draw mode + pen strokes + color/width
        # (drawing.DrawState). In draw mode a pinch inks; toggling off freezes it.
        self._draw = drawing.DrawState()
        self._draw_was_pinching = False  # pinch edge for pen up/down
        # Fist-to-remove hold: the layer index a fist is over + its held-frames.
        self._fist_index: Optional[int] = None
        self._fist_streak = 0
        self._frame_size: Tuple[int, int] = (0, 0)  # last frame size (freeze on 'd')

    def run(self) -> None:
        """Loop until 'q' (or the window is closed), pacing each tick to TARGET_FPS."""
        while self._run_frame():
            pass

    def _run_frame(self) -> bool:
        """One main-loop iteration. Returns False to quit.

        Does the frame's work, pumps the window + reads a key with one
        ``cv2.waitKey(1)``, then paces to ``TARGET_FPS`` with a precise
        ``time.sleep`` of the remaining budget. Timings feed the NEXT frame's HUD.
        """
        frame_start = time.perf_counter()
        frame = self._next_frame()
        if frame is None:
            # No webcam frame yet (threads warming up): pace + poll keys via waitKey.
            keep = self._handle_keys(cv2.waitKey(pace_wait_ms(TARGET_FPS, 0.0)) & 0xFF)
            self._pump_panel()
            return keep

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
        keep = self._handle_keys(key)
        # Pump the Tk panel last: its callbacks run inline on the main thread and
        # call this runtime's methods, so scene edits keep the single-writer rule.
        self._pump_panel()
        return keep

    def _next_frame(self) -> Optional[np.ndarray]:
        """Return a fresh, mutable compositing copy of the current webcam frame.

        The capture thread publishes each frame as a shared, READ-ONLY array; the
        compositor draws in place, so we hand back a private per-frame COPY every
        tick (adopting a new clean frame only on a new sequence). ``None`` only
        before the very first frame arrives.
        """
        packet = self._webcam.latest_packet()
        if packet is not None and packet[0] != self._clean_seq:
            self._clean_seq = packet[0]
            self._clean_frame = packet[1]  # shared, treated read-only
        if self._clean_frame is None:
            return None
        return self._clean_frame.copy()

    def _advance_scene(self, tracking: TrackingResult, frame_size: Tuple[int, int]) -> None:
        """Advance drawing OR interaction (image mode only); single scene writer.

        Draw mode and normal mode are exclusive: in DRAW MODE a pinch is the pen
        (interaction/spawn/fist inert, so a pinch can't also grab); in NORMAL mode
        a pinch grabs/moves, a spawn-preview drag takes pinch precedence, and a
        fist held over a layer removes it. Skeleton mode just clears the status.
        """
        if self._args.mode != MODE_IMAGE:
            self._status = None
            return
        self._frame_size = frame_size
        self.scene, self._fit_ids = _apply_pending_fit(self.scene, self._fit_ids, frame_size)
        gesture_states = list(tracking.gesture_states)
        primary = gesture_states[0] if gesture_states else None
        if self._draw.mode:
            self._advance_pen(primary)
            self._status = "DRAW MODE (pinch=pen, d=finish)"
            return
        grabbing = self._advance_spawn(gesture_states, frame_size)
        gestures = suppress_spawn_pinch(gesture_states, grabbing)
        self.interaction, self.scene = step(
            self.interaction, self.scene, gestures, frame_size
        )
        self._advance_fist_removal(gesture_states, frame_size)
        self._status = hud.resize_or_grab_status(
            self.interaction.grab.is_grabbed, self.interaction.resize.is_resizing
        )

    def _advance_pen(self, primary: Optional[GestureState]) -> None:
        """Pinch-to-ink in draw mode: ink while pinching, bank the stroke on release
        (reads only the published pinch, never raw landmarks)."""
        pinching = bool(primary is not None and primary.is_pinching)
        if pinching:
            self._draw = self._draw.pen_move(primary.pinch_point)
        elif self._draw_was_pinching:
            self._draw = self._draw.pen_up()
        self._draw_was_pinching = pinching

    def _freeze_draw(self) -> None:
        """Freeze the drawn strokes into one scene layer (on toggling draw mode off);
        pen color/width carry over, interaction resets so no stale pinch grabs it."""
        state = self._draw.pen_up()
        self._draw = drawing.DrawState(color_index=state.color_index, width=state.width)
        self._draw_was_pinching = False
        if self._frame_size[0] == 0:  # no frame seen yet -> nothing placeable
            return
        frozen = drawing.freeze_draw_state(state, self._frame_size)
        if frozen is None:
            return
        bgra, transform = frozen
        self.scene = self.scene.add_layer(Layer(source=ArraySource(bgra), transform=transform))
        if self.interaction is not None:
            self.interaction = InteractionState.initial()

    def _advance_fist_removal(self, gesture_states: list, frame_size: Tuple[int, int]) -> None:
        """Remove a layer after a fist is held over it for ``FIST_REMOVE_FRAMES`` ticks.

        A hold counter debounces a momentary fist; at the threshold the target is
        selected + removed via the 'x' key's source-closing path.
        """
        target = fist_removal_target(self.scene, gesture_states, frame_size)
        if target is None:
            self._fist_index, self._fist_streak = None, 0
            return
        self._fist_streak = self._fist_streak + 1 if target == self._fist_index else 1
        self._fist_index = target
        if self._fist_streak >= FIST_REMOVE_FRAMES:
            self.scene = self.scene.select_layer(target)
            self.scene, self.interaction, self._fit_ids = _remove_selected_layer(
                self.scene, self.interaction, self._fit_ids
            )
            self._fist_index, self._fist_streak = None, 0

    def _advance_spawn(self, gesture_states: list, frame_size: Tuple[int, int]) -> bool:
        """Drain the preview inbox, advance the spawn machine, apply any spawn.

        Single writer of scene/interaction (main thread): docks any asset the
        SPEECH thread published, advances the pure ``advance_spawn`` off the PRIMARY
        hand, and on a spawn action adds an ``ArraySource`` layer at the pull-to
        point via the same ``add_layer`` machinery air-drawing uses (selected, on
        top), resetting interaction so the still-held pinch grabs the new layer.
        Returns whether a preview is being dragged (for the pinch suppression).
        """
        now = time.perf_counter()
        primary = gesture_states[0] if gesture_states else None
        if self._inbox is not None:
            asset = self._inbox.take()
            if asset is not None:
                pinching = bool(primary is not None and primary.is_pinching)
                self._preview_state = dock_preview(asset, now, pinching)
        self._preview_state, action = advance_spawn(
            self._preview_state, primary, now, frame_size
        )
        if action is not None:
            self.scene = self.scene.add_layer(
                Layer(ArraySource(action.asset.bgra), action.transform)
            )
            if self.interaction is not None:
                self.interaction = InteractionState.initial()
        return self._preview_state.is_grabbing

    def toggle_draw_mode(self) -> None:
        """Toggle draw mode; freezes the drawn strokes into a layer on turning off."""
        if self._draw.mode:
            self._freeze_draw()  # leaving draw mode banks the drawing
        else:
            self._draw = self._draw.with_mode(True)

    def _handle_drawing_key(self, key: int) -> None:
        """Apply the drawing keys: toggle mode ('d'), clear ('c'), color ('k'), width."""
        if key == DRAW_MODE_KEY:
            self.toggle_draw_mode()
        elif not self._draw.mode:
            return  # c/k/[/] only act while in draw mode
        elif key == CLEAR_STROKE_KEY:
            self._draw = self._draw.cleared()
        elif key == CYCLE_COLOR_KEY:
            self._draw = self._draw.with_next_color()
        elif key == STROKE_WIDTH_DOWN_KEY:
            self._draw = self._draw.with_width_delta(-drawing.STROKE_WIDTH_STEP)
        elif key == STROKE_WIDTH_UP_KEY:
            self._draw = self._draw.with_width_delta(drawing.STROKE_WIDTH_STEP)

    def _composite(self, frame: np.ndarray, tracking: TrackingResult) -> float:
        """Composite the scene + highlight (+ skeleton/live ink/preview); return ms.

        The skeleton + gesture label, live draw-mode ink, and the docked preview
        all draw on top (in that order, before the HUD) so they reach the vcam too.
        """
        with StageTimer() as timer:
            compositor.render(frame, self.scene)
            compositor.draw_selection_highlight(frame, self.scene)
            if self.show_skeleton:
                _draw_debug(frame, list(tracking.hands), self._primary_gesture(tracking))
            if self._draw.mode:
                drawing.draw_live_state(frame, self._draw)  # live ink -> preview + vcam
            render_dock(frame, self._preview_state)  # docked preview -> preview + vcam
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
            self._handle_drawing_key(key)
        return True

    # -- panel integration (Phase 10) -------------------------------------- #
    #
    # These are the control-panel's drive points. They run on the MAIN thread
    # (from a Tk button callback during _pump_panel), so mutating scene/
    # interaction here keeps this runtime the single writer -- the exact same
    # invariant the hotkeys rely on, with no duplicated transition logic (add
    # reuses _add_source_layer; remove/reset reuse _close_layer_sources).

    def set_panel(self, panel: Optional["ControlPanel"]) -> None:
        """Attach the Tk control panel to pump each tick (None = hotkeys only)."""
        self._panel = panel

    def _pump_panel(self) -> None:
        """Pump the attached panel once (no-op when there's none / it's closed)."""
        if self._panel is not None:
            self._panel.pump()

    @property
    def last_fps(self) -> Optional[float]:
        """Last reported main-loop FPS (None before the first one-second window)."""
        return self._last_fps

    @property
    def panel_status(self) -> Optional[str]:
        """Last panel-action message (e.g. 'Added window layer.' or an error)."""
        return self._panel_status

    def add_image_layer(self, image_path: str) -> None:
        """Panel action: add an image layer from ``image_path``."""
        self._panel_add_layer(lambda: _image_source_from_path(image_path), False, "image")

    def add_window_layer(self, window_title: str) -> None:
        """Panel action: add a live window layer from ``window_title``."""
        self._panel_add_layer(lambda: _window_source_from_title(window_title), True, "window")

    def add_screen_layer(self, monitor_index: int) -> None:
        """Panel action: add a live monitor layer from ``monitor_index``."""
        self._panel_add_layer(lambda: _monitor_source_from_index(monitor_index), True, "screen")

    def _panel_add_layer(self, build_source: Callable[[], Source], live: bool, label: str) -> None:
        """Add a layer for a panel action; surface the outcome to the status label.

        Reuses ``_add_source_layer`` (no duplicated transition logic). A
        ``SourceError`` is shown in the panel status AND printed to stderr, and
        the scene is left unchanged -- never a crash or exit.
        """
        self.scene, self.interaction, self._fit_ids, error = _add_source_layer(
            self.scene, self.interaction, self._fit_ids, build_source, live
        )
        if error is not None:
            print(f"Error: {error}", file=sys.stderr)
            self._panel_status = error
        else:
            self._panel_status = f"Added {label} layer."

    def toggle_virtualcam(self) -> None:
        """Panel action: flip the virtual-cam sink on/off (shares state with 'v')."""
        self.virtualcam_active = not self.virtualcam_active

    def set_tuning(self, enter: float, exit: float) -> GestureTuning:
        """Panel action: retune pinch thresholds live (main thread). Returns the clamp.

        Runs the raw slider pair through the shared ``clamp_tuning`` (enforcing
        the hysteresis gap), records it as the runtime-authoritative tuning (so it
        persists on exit), and publishes it to the tracking worker via one atomic
        reference store -- the worker swaps it onto the detectors it owns. The
        main thread NEVER touches detector internals. Returns the clamped tuning
        so the panel can snap its numeric labels to the effective values. A worker
        of ``None`` (skeleton smokes / tests) just updates the stored tuning.
        """
        tuning = clamp_tuning(enter, exit)
        self.tuning = tuning
        if self._worker is not None:
            self._worker.set_tuning(tuning)
        return tuning

    def reset_scene(self) -> None:
        """Panel action: remove ALL layers (closing each source) and idle interaction.

        Implemented as a runtime method (single writer of scene/interaction) so a
        future hotkey could call it too. Reuses ``_close_layer_sources`` to stop
        live capture threads. Interaction is reset only in image mode (it's None
        in skeleton mode). Empties ``_fit_ids`` since every layer is gone.
        """
        _close_layer_sources(self.scene.layers)
        self.scene = Scene()
        if self.interaction is not None:
            self.interaction = InteractionState.initial()
        self._fit_ids = set()
        self._panel_status = "Scene reset."

    @staticmethod
    def _primary_gesture(tracking: TrackingResult) -> Optional[GestureState]:
        """The primary hand's gesture state (slot 0), or None if none published."""
        return tracking.gesture_states[0] if tracking.gesture_states else None


def run(args: argparse.Namespace) -> None:
    """Wire the threaded pipeline, run it, and tear everything down on exit.

    Capture (webcam) and hand tracking each run on their OWN thread publishing
    latest-frame-wins; this main thread only composites + displays + handles keys,
    paced to ``TARGET_FPS`` (see ``PipelineRuntime`` / ``TrackingWorker`` /
    ``ThreadedSource``). The webcam opens synchronously first (fail-fast) before
    its thread starts. Unless ``--no-panel``, a Tk panel is created on this thread
    and pumped per tick. Settings load from ``handy_config.json`` (CLI > config >
    defaults) and persist back on exit.
    """
    cfg = config.load_config(config.CONFIG_FILENAME)
    prefill = resolve_panel_prefill(args, cfg)
    # Resolved startup tuning (tuning_from_prefill clamps, so the hysteresis
    # invariant holds even if cfg was bypassed) + startup-only confidences. The
    # detectors are built WITH the tuning so the persisted pinch value applies
    # from frame one, not only after a slider nudge.
    tuning = tuning_from_prefill(prefill)
    confidence = confidence_from_prefill(prefill)

    tracker, detectors, scene, interaction = _init_mode(args, tuning, confidence)
    preview = PreviewSink(WINDOW_NAME)
    virtualcam = VirtualCamSink()
    webcam = ThreadedSource(WebcamSource())
    inbox = PreviewInbox()  # Phase 15: the speech->main asset hand-off slot
    worker: Optional[TrackingWorker] = None
    runtime: Optional[PipelineRuntime] = None
    panel = None

    try:
        webcam.open()  # opens the camera (fail-fast) then starts the capture thread
        worker = TrackingWorker(tracker, detectors, webcam.latest_packet)
        worker.start()
        runtime = PipelineRuntime(
            args, webcam, worker, scene, interaction, preview, virtualcam,
            virtualcam_active=prefill["virtualcam"],
            tuning=tuning,
            inbox=inbox,
        )
        if not args.no_panel:
            panel = _build_panel(runtime, prefill)
            runtime.set_panel(panel)
        # Speech (Phase 12) runs on its OWN thread. Phase 13/14: when --parse is
        # on, its finalized-phrase callback parses + retrieves ON THE SPEECH
        # THREAD and publishes any asset to the inbox (Phase 15). None = default.
        on_phrase = build_speech_on_phrase(prefill["parse"], inbox, prefill["generate"])
        with speech_session(prefill["speech"], on_phrase=on_phrase):
            runtime.run()
    except CameraError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        # stop() reports whether the tracking thread really exited; if a native
        # detect() wedged past the join timeout, freeing the tracker under it
        # risks a use-after-free, so skip close() and let process exit reclaim it.
        worker_stopped = worker.stop() if worker is not None else True
        webcam.close()
        if tracker is not None:
            if worker_stopped:
                tracker.close()
            else:
                print(
                    "Warning: tracking thread didn't stop in time; leaving the "
                    "hand tracker to be reclaimed at process exit.", file=sys.stderr,
                )
        preview.close()
        virtualcam.close()
        # Persist last-used settings: prefer the panel's/runtime's live values
        # (sliders drove the tuning), fall back to launch args when the panel was
        # off or never built. save_config never raises.
        vcam_state = runtime.virtualcam_active if runtime is not None else args.virtualcam
        tuning_state = runtime.tuning if runtime is not None else tuning
        config.save_config(
            config.CONFIG_FILENAME,
            _final_settings(
                panel, args, vcam_state, tuning_state, confidence,
                prefill["speech"], prefill["parse"], prefill["generate"],
            ),
        )
        if panel is not None:
            panel.close()
        # Release any layer sources (stops live window/screen capture threads;
        # no-op for image layers). Use the runtime's final scene so runtime-added
        # layers close too; fall back to the initial scene if run never started.
        final_scene = runtime.scene if runtime is not None else scene
        for layer in final_scene.layers:
            layer.source.close()


def main(argv=None) -> None:
    args = parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()

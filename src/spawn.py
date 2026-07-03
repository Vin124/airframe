"""Spawn gesture: dock a retrieved image, then pinch-from-top-and-pull it in.

Phase 15 -- the moment speech -> parser -> retriever meets the scene. When the
retriever (running on the SPEECH thread) produces a ``RetrievedAsset``, it is
handed to the MAIN loop through one ``PreviewInbox`` slot; the main loop docks it
as a preview at the top edge of the frame, and a pinch that starts over that dock
and pulls downward past a threshold instantiates the asset as a normal scene
``Layer`` at the pinch point -- after which the existing grab/move/resize
interaction takes over. An ignored preview dismisses after a few seconds.

Everything here is PURE and unit-testable (numpy + cv2 only; ``scene.Transform``
for placement, ``RetrievedAsset`` under ``TYPE_CHECKING`` for typing) EXCEPT the
one tiny ``PreviewInbox`` class, which is the single cross-thread channel. No
camera, MediaPipe, or threads. Coordinates are normalized 0-1 except in the dock
geometry / renderer, which work in frame pixels (like the compositor).

THREADING -- ``PreviewInbox`` is the ONLY speech->main hand-off:
    ``publish(asset)`` is called from the SPEECH thread (in the retriever
    callback); ``take()`` is called by the MAIN loop each tick. It is a single
    atomic reference slot (GIL-safe store/load, no lock), latest-wins: a publish
    overwrites any un-taken asset, and take clears the slot. The speech thread
    NEVER touches the preview/scene/interaction state -- it only drops an asset in
    the slot. The main loop stays the SINGLE WRITER of the scene, exactly like the
    ThreadedSource / TrackingWorker latest-frame-wins pattern.
"""

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, List, Optional, Protocol, Tuple

import cv2
import numpy as np

from src.scene import DEFAULT_SCALE, Transform

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps this module import-light
    from src.retriever import RetrievedAsset

Point2D = Tuple[float, float]
PixelRect = Tuple[int, int, int, int]  # (x, y, w, h) in frame pixels

# --------------------------------------------------------------------------- #
# Tunable constants (every threshold named, per the project convention).
# --------------------------------------------------------------------------- #

# The docked preview is fit into a top band this tall (fraction of frame height)
# and no wider than this fraction of frame width, keeping its aspect ratio.
DOCK_BAND_HEIGHT_FRACTION = 0.22
DOCK_MAX_WIDTH_FRACTION = 0.35
# Small gap from the very top edge so the preview + its border aren't clipped.
DOCK_TOP_MARGIN_FRACTION = 0.03

# How far (normalized, downward) the pinch must travel below its grab point
# before the preview instantiates into the scene.
PULL_DOWN_DISTANCE = 0.18

# An idle (un-grabbed) preview auto-dismisses this many seconds after it docks.
DISMISS_SECONDS = 6.0

# Spawned layer size: its width is fit to this fraction of the frame width (the
# same spirit as the live-source INITIAL_FIT_FRACTION), centered at the pinch.
SPAWN_WIDTH_FRACTION = 0.4

# Dock chrome (BGR). Orange border matches the selected-layer accent; the hint is
# a thin white label so it reads over a webcam feed.
DOCK_BORDER_COLOR = (0, 165, 255)
DOCK_BORDER_THICKNESS = 2
DOCK_HINT_TEXT = "pull down to add"
DOCK_HINT_COLOR = (255, 255, 255)
DOCK_HINT_SCALE = 0.5
DOCK_HINT_THICKNESS = 1
DOCK_HINT_GAP_PX = 16       # gap below the dock before the hint text baseline
DOCK_HINT_MARGIN_PX = 4     # keep the hint at least this far above the frame edge

# Spawn phases: a docked preview waits IDLE until a pinch over it starts a
# GRABBED drag; crossing PULL_DOWN_DISTANCE emits a spawn and clears the preview.
PHASE_IDLE = "idle"
PHASE_GRABBED = "grabbed"


class SpawnGesture(Protocol):
    """Structural type for the one hand the spawn machine reads (``GestureState``)."""

    is_pinching: bool
    pinch_point: Point2D


# --------------------------------------------------------------------------- #
# PreviewInbox: the ONE cross-thread slot (speech thread -> main loop)
# --------------------------------------------------------------------------- #

class PreviewInbox:
    """A single atomic reference slot handing a retrieved asset to the main loop.

    ``publish`` (SPEECH thread) stores the asset with one reference store;
    ``take`` (MAIN loop) reads and clears it. Latest-wins: a second publish before
    a take simply overwrites the first (an old, un-taken preview is dropped in
    favor of the newest). GIL-safe single store/load -- no lock, mirroring the
    ThreadedSource / TrackingWorker hand-off. The clear-on-take has a vanishingly
    small window where a publish landing between the read and the clear could be
    dropped; that is acceptable latest-wins tolerance for this debug harness (a
    phrase arrives seconds apart, take runs every ~33ms).
    """

    def __init__(self) -> None:
        self._asset: Optional["RetrievedAsset"] = None

    def publish(self, asset: "RetrievedAsset") -> None:
        """Drop the newest asset in the slot (SPEECH thread; latest-wins)."""
        self._asset = asset  # single atomic reference store

    def take(self) -> Optional["RetrievedAsset"]:
        """Return + clear the pending asset, or ``None`` if empty (MAIN loop)."""
        asset = self._asset  # one atomic read
        self._asset = None
        return asset


# --------------------------------------------------------------------------- #
# Immutable spawn state + the action the machine emits
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class PreviewState:
    """The docked-preview + spawn-gesture state (frozen; functional updates).

    ``asset`` is the docked ``RetrievedAsset`` (or ``None`` when nothing is
    docked). ``phase`` is ``PHASE_IDLE`` (waiting) or ``PHASE_GRABBED`` (a pinch is
    dragging it). ``dismiss_at`` is an ABSOLUTE monotonic timestamp (the caller
    passes the clock in, so this module never reads time); an idle preview past it
    dismisses. ``grab_point`` is the normalized pinch point captured when the grab
    started, and ``was_pinching`` is the previous frame's primary-hand pinch flag
    for rising-edge detection.
    """

    asset: Optional["RetrievedAsset"] = None
    phase: str = PHASE_IDLE
    dismiss_at: float = 0.0
    grab_point: Point2D = (0.0, 0.0)
    was_pinching: bool = False

    @property
    def is_docked(self) -> bool:
        """True when a preview asset is currently docked."""
        return self.asset is not None

    @property
    def is_grabbing(self) -> bool:
        """True while a pinch is dragging the docked preview (GRABBED phase)."""
        return self.asset is not None and self.phase == PHASE_GRABBED


@dataclass(frozen=True)
class SpawnAction:
    """Instruction the main loop applies: add ``asset`` at ``transform``."""

    asset: "RetrievedAsset"
    transform: Transform


def dock_preview(
    asset: "RetrievedAsset",
    now: float,
    was_pinching: bool,
    dismiss_seconds: float = DISMISS_SECONDS,
) -> PreviewState:
    """Build the IDLE docked state for a freshly retrieved ``asset``.

    ``was_pinching`` seeds the rising-edge memory with the primary hand's current
    pinch, so a hand that happens to already be pinching when the preview appears
    must release and re-pinch to grab it (no accidental instant spawn).
    """
    return PreviewState(
        asset=asset,
        phase=PHASE_IDLE,
        dismiss_at=now + dismiss_seconds,
        was_pinching=was_pinching,
    )


# --------------------------------------------------------------------------- #
# Pure geometry: dock placement, hit-test, pull distance, spawn placement
# --------------------------------------------------------------------------- #

def _asset_size(bgra: np.ndarray) -> Tuple[int, int]:
    """Return an asset's native ``(width, height)`` from its BGRA array."""
    return int(bgra.shape[1]), int(bgra.shape[0])


def dock_rect(frame_size: Tuple[int, int], asset_size: Tuple[int, int]) -> PixelRect:
    """Place the preview scaled-to-fit the top band, horizontally centered.

    Fits ``asset_size`` inside a band ``DOCK_BAND_HEIGHT_FRACTION`` tall and
    ``DOCK_MAX_WIDTH_FRACTION`` wide (whichever bound binds first), preserving
    aspect ratio, and returns the pixel rect ``(x, y, w, h)`` sitting a small top
    margin below the frame's top edge. A degenerate (zero-dim) asset yields a
    1x1 rect at the margin so callers can no-op safely.
    """
    frame_w, frame_h = frame_size
    asset_w, asset_h = asset_size
    top_margin = int(round(frame_h * DOCK_TOP_MARGIN_FRACTION))
    if asset_w <= 0 or asset_h <= 0:
        return (max(0, frame_w // 2), top_margin, 1, 1)
    band_h = frame_h * DOCK_BAND_HEIGHT_FRACTION
    max_w = frame_w * DOCK_MAX_WIDTH_FRACTION
    scale = min(max_w / asset_w, band_h / asset_h)
    w = max(1, int(round(asset_w * scale)))
    h = max(1, int(round(asset_h * scale)))
    x = int(round((frame_w - w) / 2.0))
    return (x, top_margin, w, h)


def pinch_over_dock(
    pinch_point: Point2D, rect: PixelRect, frame_size: Tuple[int, int]
) -> bool:
    """Hit-test: is the normalized ``pinch_point`` inside the dock ``rect``?

    Half-open pixel rect (top/left inclusive, bottom/right exclusive), matching
    the scene compositor's ``contains`` so a pinch on the visible edge counts.
    """
    frame_w, frame_h = frame_size
    x, y, w, h = rect
    px = pinch_point[0] * frame_w
    py = pinch_point[1] * frame_h
    return x <= px < x + w and y <= py < y + h


def pull_down_distance(grab_point: Point2D, pinch_point: Point2D) -> float:
    """Normalized DOWNWARD travel of the pinch from where the grab started.

    Positive means the pinch moved down the frame (larger y); zero/negative means
    level or upward. Only the vertical component matters for the pull gesture.
    """
    return pinch_point[1] - grab_point[1]


def spawn_transform(
    pinch_point: Point2D, asset_size: Tuple[int, int], frame_size: Tuple[int, int]
) -> Transform:
    """Placement for the spawned layer: centered at the pinch, fit to a sane size.

    The layer is centered at ``pinch_point`` and scaled so its width is about
    ``SPAWN_WIDTH_FRACTION`` of the frame width (``Transform`` clamps both center
    into the frame and scale into its valid range). At this scale the compositor's
    ``top_left_px`` places the image's center back at the pinch point -- the same
    round-trip ``drawing.bbox_to_transform`` relies on.
    """
    frame_w, _frame_h = frame_size
    asset_w, _asset_h = asset_size
    if asset_w <= 0:
        return Transform(pinch_point[0], pinch_point[1], DEFAULT_SCALE)
    scale = (frame_w * SPAWN_WIDTH_FRACTION) / asset_w
    return Transform(pinch_point[0], pinch_point[1], scale)


# --------------------------------------------------------------------------- #
# The spawn state machine (deterministic, side-effect-free)
# --------------------------------------------------------------------------- #

def advance_spawn(
    state: PreviewState,
    gesture: Optional[SpawnGesture],
    timestamp: float,
    frame_size: Tuple[int, int],
) -> Tuple[PreviewState, Optional[SpawnAction]]:
    """Advance the docked-preview spawn machine one frame (pure).

    Rules, driven by the PRIMARY hand's ``gesture`` (or ``None`` on hand loss):
      - No preview docked -> no-op.
      - IDLE + a pinch RISING EDGE whose point is over the dock -> GRABBED
        (records the grab point).
      - GRABBED + still pinching + pulled down past ``PULL_DOWN_DISTANCE`` ->
        EMIT a ``SpawnAction`` (asset + ``spawn_transform`` at the current pinch)
        and CLEAR the preview.
      - GRABBED + released before the threshold -> back to IDLE (still docked).
      - IDLE past ``dismiss_at`` -> cleared (dismissed).
    Returns the new state and the spawn action (or ``None``). No side effects --
    the caller applies the action to the scene.
    """
    if state.asset is None:
        return state, None

    is_pinching = bool(gesture is not None and gesture.is_pinching)
    pinch_point: Point2D = gesture.pinch_point if gesture is not None else (0.0, 0.0)

    if state.phase == PHASE_GRABBED:
        if not is_pinching:
            return replace(state, phase=PHASE_IDLE, was_pinching=False), None
        if pull_down_distance(state.grab_point, pinch_point) >= PULL_DOWN_DISTANCE:
            transform = spawn_transform(pinch_point, _asset_size(state.asset.bgra), frame_size)
            return PreviewState(), SpawnAction(state.asset, transform)
        return replace(state, was_pinching=True), None

    # PHASE_IDLE (docked, waiting).
    rect = dock_rect(frame_size, _asset_size(state.asset.bgra))
    pinch_started = is_pinching and not state.was_pinching
    if pinch_started and pinch_over_dock(pinch_point, rect, frame_size):
        return replace(state, phase=PHASE_GRABBED, grab_point=pinch_point, was_pinching=True), None
    if timestamp >= state.dismiss_at:
        return PreviewState(), None  # ignored too long -> dismissed
    return replace(state, was_pinching=is_pinching), None


# --------------------------------------------------------------------------- #
# Gesture precedence: mask the spawning hand's pinch from interaction.step
# --------------------------------------------------------------------------- #

def suppress_spawn_pinch(gesture_states: List, grabbing: bool) -> List:
    """Force the PRIMARY hand's ``is_pinching`` False while it drags the preview.

    Deterministic precedence: while a pinch is dragging the docked preview
    (GRABBED), that same pinch must not also grab an existing scene layer via
    ``interaction.step``. Masking slot 0's pinch (a ``dataclasses.replace`` copy,
    inputs never mutated) keeps the drag exclusive; once the preview spawns
    (``grabbing`` False) the pinch flows through so interaction picks up the new
    layer. A no-op when not grabbing or no hands are present. Pure + testable.
    """
    if not grabbing or not gesture_states:
        return gesture_states
    primary = replace(gesture_states[0], is_pinching=False)
    return [primary, *gesture_states[1:]]


# --------------------------------------------------------------------------- #
# Rendering: draw the docked preview + border + hint onto the composited frame
# --------------------------------------------------------------------------- #

def _blit_bgra(frame: np.ndarray, bgra: np.ndarray, x: int, y: int) -> None:
    """Alpha-blend a BGRA image onto ``frame`` (BGR) at pixel ``(x, y)``, clipped.

    Same edge-clip + per-pixel alpha blend the compositor uses; mutates ``frame``.
    """
    frame_h, frame_w = frame.shape[:2]
    bh, bw = bgra.shape[:2]
    fx0, fy0 = max(0, x), max(0, y)
    fx1, fy1 = min(frame_w, x + bw), min(frame_h, y + bh)
    if fx0 >= fx1 or fy0 >= fy1:
        return
    ox0, oy0 = fx0 - x, fy0 - y
    ox1, oy1 = ox0 + (fx1 - fx0), oy0 + (fy1 - fy0)
    region = bgra[oy0:oy1, ox0:ox1]
    alpha = region[:, :, 3:4].astype(np.float32) / 255.0
    fg = region[:, :, :3].astype(np.float32)
    bg = frame[fy0:fy1, fx0:fx1].astype(np.float32)
    frame[fy0:fy1, fx0:fx1] = (alpha * fg + (1.0 - alpha) * bg).astype(frame.dtype)


def render_dock(frame: np.ndarray, state: PreviewState) -> None:
    """Draw the docked preview (scaled into ``dock_rect``) + border + hint, in place.

    A no-op when nothing is docked. Drawn on the private composited copy AFTER the
    scene + active stroke and BEFORE the HUD, so the preview shows in BOTH the
    preview window and the virtual-cam feed. Cheap: one resize, one blit, a
    rectangle, and a short text label.
    """
    if state.asset is None:
        return
    frame_h, frame_w = frame.shape[:2]
    bgra = state.asset.bgra
    bh, bw = bgra.shape[:2]
    if bw < 1 or bh < 1:
        return  # defensive: a 0-dim asset would crash cv2.resize (unreachable today)
    x, y, w, h = dock_rect((frame_w, frame_h), _asset_size(bgra))
    if w < 1 or h < 1:
        return
    scaled = cv2.resize(bgra, (w, h), interpolation=cv2.INTER_AREA)
    _blit_bgra(frame, scaled, x, y)
    cv2.rectangle(frame, (x, y), (x + w, y + h), DOCK_BORDER_COLOR, DOCK_BORDER_THICKNESS)
    hint_y = min(frame_h - DOCK_HINT_MARGIN_PX, y + h + DOCK_HINT_GAP_PX)
    cv2.putText(
        frame, DOCK_HINT_TEXT, (x, hint_y), cv2.FONT_HERSHEY_SIMPLEX,
        DOCK_HINT_SCALE, DOCK_HINT_COLOR, DOCK_HINT_THICKNESS, cv2.LINE_AA,
    )

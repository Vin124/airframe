"""Unit tests for Phase 15 spawn gesture (src.spawn + its main-loop wiring).

Covers the PreviewInbox thread hand-off, the pure dock/hit-test/pull/placement
geometry, the advance_spawn state machine, the suppress-during-grab precedence,
the dock renderer, and the PipelineRuntime lifecycle (dock -> pull -> spawn).
Headless -- fake webcam/asset, no sinks/worker/camera/mic/network/display.
"""

from dataclasses import dataclass

import numpy as np

from src.gestures import GestureState
from src.interaction import GrabState, InteractionState
from src.main import PipelineRuntime, parse_args
from src.scene import Layer, Scene, Transform, rect_px
from src.sources import ArraySource
from src.sources.base import Size, Source
from src.spawn import (
    DISMISS_SECONDS,
    PHASE_GRABBED,
    PHASE_IDLE,
    PULL_DOWN_DISTANCE,
    PreviewInbox,
    PreviewState,
    SpawnAction,
    advance_spawn,
    dock_preview,
    dock_rect,
    pinch_over_dock,
    pull_down_distance,
    render_dock,
    spawn_transform,
    suppress_spawn_pinch,
)

FRAME_W = FRAME_H = 200
FRAME_SIZE = (FRAME_W, FRAME_H)


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #

@dataclass
class _FakeAsset:
    """A minimal RetrievedAsset stand-in: only ``bgra`` (+ ``source_name``)."""

    bgra: np.ndarray
    source_name: str = "wikimedia"


def _solid_bgra(w: int, h: int, color=(0, 0, 255)) -> np.ndarray:
    """An opaque solid-color HxWx4 BGRA image."""
    img = np.zeros((h, w, 4), dtype=np.uint8)
    img[:, :, 0], img[:, :, 1], img[:, :, 2] = color
    img[:, :, 3] = 255
    return img


class _FakeSource(Source):
    """A source with a fixed base_size (read() unused by the tested paths)."""

    def __init__(self, base_size: Size = (40, 40)) -> None:
        self._base_size = base_size

    def read(self):
        return None

    @property
    def base_size(self) -> Size:
        return self._base_size


class _FakeWebcam:
    def latest_packet(self):
        return None


class _Tracking:
    """Minimal TrackingResult stand-in for the runtime lifecycle tests."""

    def __init__(self, gesture_states, hands=()):
        self.gesture_states = tuple(gesture_states)
        self.hands = tuple(hands)


def _pinch(point) -> GestureState:
    return GestureState(is_pinching=True, pinch_point=point, is_grab=False)


def _idle() -> GestureState:
    return GestureState(is_pinching=False, pinch_point=(0.0, 0.0), is_grab=False)


def _asset(w: int = 20, h: int = 20, color=(0, 0, 255)) -> _FakeAsset:
    return _FakeAsset(_solid_bgra(w, h, color))


def _runtime(inbox=None, scene=None, interaction=None) -> PipelineRuntime:
    args = parse_args(["--mode", "image", "--image", "assets/test.png"])
    return PipelineRuntime(
        args,
        _FakeWebcam(),
        None,
        scene if scene is not None else Scene(),
        interaction if interaction is not None else InteractionState.initial(),
        None,
        None,
        inbox=inbox,
    )


# --------------------------------------------------------------------------- #
# PreviewInbox: the single atomic slot (speech thread -> main loop)
# --------------------------------------------------------------------------- #

def test_inbox_take_empty_is_none():
    assert PreviewInbox().take() is None


def test_inbox_publish_then_take_returns_and_clears():
    inbox = PreviewInbox()
    asset = _asset()
    inbox.publish(asset)
    assert inbox.take() is asset
    assert inbox.take() is None  # cleared after taking


def test_inbox_publish_overwrites_latest_wins():
    inbox = PreviewInbox()
    first, second = _asset(), _asset()
    inbox.publish(first)
    inbox.publish(second)  # un-taken first is dropped
    assert inbox.take() is second


# --------------------------------------------------------------------------- #
# dock_rect: scaled-to-fit the top band, centered, aspect kept
# --------------------------------------------------------------------------- #

def test_dock_rect_square_asset_fits_band_centered():
    x, y, w, h = dock_rect(FRAME_SIZE, (20, 20))
    # 200x200 frame: band 44 tall, max width 70 -> height-bound scale 2.2 -> 44x44.
    assert (w, h) == (44, 44)
    assert y == 6  # top margin (3% of 200)
    assert x == (FRAME_W - w) // 2  # horizontally centered


def test_dock_rect_wide_asset_is_width_bound():
    x, y, w, h = dock_rect(FRAME_SIZE, (100, 20))
    assert w == 70  # capped at 35% of frame width
    assert h == 14  # aspect preserved (100:20 == 70:14)
    assert abs(w / h - 100 / 20) < 0.2


def test_dock_rect_tall_asset_is_height_bound():
    x, y, w, h = dock_rect(FRAME_SIZE, (20, 100))
    assert h == 44  # capped at the band height (22% of 200)
    assert w == 9   # aspect preserved (~0.2)


def test_dock_rect_degenerate_asset_is_safe():
    x, y, w, h = dock_rect(FRAME_SIZE, (0, 0))
    assert (w, h) == (1, 1)  # no crash / no divide-by-zero


# --------------------------------------------------------------------------- #
# pinch_over_dock: half-open pixel hit-test
# --------------------------------------------------------------------------- #

def test_pinch_over_dock_inside_and_outside():
    rect = dock_rect(FRAME_SIZE, (20, 20))  # (78, 6, 44, 44)
    assert pinch_over_dock((0.5, 0.14), rect, FRAME_SIZE) is True   # (100, 28)
    assert pinch_over_dock((0.5, 0.6), rect, FRAME_SIZE) is False   # below the band
    assert pinch_over_dock((0.2, 0.14), rect, FRAME_SIZE) is False  # left of dock


def test_pinch_over_dock_edges_are_half_open():
    rect = (78, 6, 44, 44)
    assert pinch_over_dock((78 / FRAME_W, 6 / FRAME_H), rect, FRAME_SIZE) is True   # top-left incl.
    assert pinch_over_dock((122 / FRAME_W, 28 / FRAME_H), rect, FRAME_SIZE) is False  # right excl.


# --------------------------------------------------------------------------- #
# pull_down_distance: normalized downward travel
# --------------------------------------------------------------------------- #

def test_pull_down_distance_signed_vertical():
    assert abs(pull_down_distance((0.5, 0.2), (0.5, 0.4)) - 0.2) < 1e-9  # down = positive
    assert pull_down_distance((0.5, 0.4), (0.5, 0.2)) < 0               # up = negative


# --------------------------------------------------------------------------- #
# spawn_transform: centered at pinch, round-trips through the compositor
# --------------------------------------------------------------------------- #

def test_spawn_transform_centers_at_pinch_and_fits_width():
    t = spawn_transform((0.5, 0.6), (20, 20), FRAME_SIZE)
    assert abs(t.center_x - 0.5) < 1e-9 and abs(t.center_y - 0.6) < 1e-9
    assert abs(t.scale - (FRAME_W * 0.4 / 20)) < 1e-9  # width fit


def test_spawn_transform_round_trips_against_compositor_placement():
    # Like Phase 11's placement test: the compositor rect's center must land back
    # on the pinch point in pixels.
    pinch = (0.5, 0.6)
    t = spawn_transform(pinch, (20, 20), FRAME_SIZE)
    x0, y0, w, h = rect_px(t, (20, 20), FRAME_SIZE)
    assert abs((x0 + w / 2) - pinch[0] * FRAME_W) < 1.0
    assert abs((y0 + h / 2) - pinch[1] * FRAME_H) < 1.0


# --------------------------------------------------------------------------- #
# advance_spawn state machine
# --------------------------------------------------------------------------- #

def test_advance_spawn_no_preview_is_noop():
    state = PreviewState()  # nothing docked
    new, action = advance_spawn(state, _pinch((0.5, 0.14)), 0.0, FRAME_SIZE)
    assert new is state and action is None


def test_advance_spawn_rising_edge_over_dock_enters_grabbed():
    state = dock_preview(_asset(), now=0.0, was_pinching=False)
    new, action = advance_spawn(state, _pinch((0.5, 0.14)), 0.0, FRAME_SIZE)
    assert new.phase == PHASE_GRABBED
    assert new.grab_point == (0.5, 0.14)
    assert action is None


def test_advance_spawn_pinch_over_dock_without_rising_edge_stays_idle():
    # was_pinching True -> not a fresh pinch -> no grab.
    state = dock_preview(_asset(), now=0.0, was_pinching=True)
    new, action = advance_spawn(state, _pinch((0.5, 0.14)), 0.0, FRAME_SIZE)
    assert new.phase == PHASE_IDLE and action is None


def test_advance_spawn_rising_edge_off_dock_stays_idle():
    state = dock_preview(_asset(), now=0.0, was_pinching=False)
    new, action = advance_spawn(state, _pinch((0.5, 0.9)), 0.0, FRAME_SIZE)
    assert new.phase == PHASE_IDLE and action is None


def test_advance_spawn_pull_past_threshold_emits_action_and_clears():
    asset = _asset()
    state = PreviewState(
        asset=asset, phase=PHASE_GRABBED, grab_point=(0.5, 0.14),
        was_pinching=True, dismiss_at=1e18,
    )
    pinch_y = 0.14 + PULL_DOWN_DISTANCE + 0.01
    new, action = advance_spawn(state, _pinch((0.5, pinch_y)), 0.0, FRAME_SIZE)
    assert isinstance(action, SpawnAction)
    assert action.asset is asset
    assert abs(action.transform.center_x - 0.5) < 1e-9
    assert abs(action.transform.center_y - pinch_y) < 1e-9
    assert new.asset is None  # preview cleared on spawn


def test_advance_spawn_release_before_threshold_returns_to_idle_docked():
    asset = _asset()
    state = PreviewState(
        asset=asset, phase=PHASE_GRABBED, grab_point=(0.5, 0.14),
        was_pinching=True, dismiss_at=1e18,
    )
    new, action = advance_spawn(state, _idle(), 0.0, FRAME_SIZE)
    assert new.phase == PHASE_IDLE
    assert new.asset is asset  # still docked
    assert action is None


def test_advance_spawn_grabbed_short_pull_stays_grabbed():
    state = PreviewState(
        asset=_asset(), phase=PHASE_GRABBED, grab_point=(0.5, 0.14),
        was_pinching=True, dismiss_at=1e18,
    )
    new, action = advance_spawn(state, _pinch((0.5, 0.16)), 0.0, FRAME_SIZE)
    assert new.phase == PHASE_GRABBED and action is None


def test_advance_spawn_idle_past_deadline_dismisses():
    state = dock_preview(_asset(), now=0.0, was_pinching=False)  # dismiss_at = DISMISS_SECONDS
    new, action = advance_spawn(state, _idle(), DISMISS_SECONDS + 1.0, FRAME_SIZE)
    assert new.asset is None  # dismissed
    assert action is None


def test_advance_spawn_grabbed_ignores_dismiss_deadline():
    state = PreviewState(
        asset=_asset(), phase=PHASE_GRABBED, grab_point=(0.5, 0.14),
        was_pinching=True, dismiss_at=0.0,  # already past
    )
    new, action = advance_spawn(state, _pinch((0.5, 0.16)), 999.0, FRAME_SIZE)
    assert new.phase == PHASE_GRABBED  # a drag is never auto-dismissed
    assert new.asset is not None


# --------------------------------------------------------------------------- #
# suppress_spawn_pinch: precedence while dragging the preview
# --------------------------------------------------------------------------- #

def test_suppress_spawn_pinch_masks_primary_when_grabbing():
    primary, secondary = _pinch((0.5, 0.3)), _pinch((0.2, 0.2))
    out = suppress_spawn_pinch([primary, secondary], grabbing=True)
    assert out[0].is_pinching is False
    assert out[0].pinch_point == (0.5, 0.3)  # point preserved
    assert out[1] is secondary  # other hand untouched
    assert primary.is_pinching is True  # input not mutated


def test_suppress_spawn_pinch_noop_when_not_grabbing():
    states = [_pinch((0.5, 0.3)), _idle()]
    assert suppress_spawn_pinch(states, grabbing=False) is states


def test_suppress_spawn_pinch_noop_on_empty():
    assert suppress_spawn_pinch([], grabbing=True) == []


# --------------------------------------------------------------------------- #
# dock_preview + render_dock
# --------------------------------------------------------------------------- #

def test_dock_preview_sets_idle_and_dismiss_deadline():
    asset = _asset()
    state = dock_preview(asset, now=10.0, was_pinching=True)
    assert state.asset is asset
    assert state.phase == PHASE_IDLE
    assert state.dismiss_at == 10.0 + DISMISS_SECONDS
    assert state.was_pinching is True


def test_render_dock_noop_without_asset():
    frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    render_dock(frame, PreviewState())
    assert frame.sum() == 0


def test_render_dock_draws_preview_pixels():
    frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    render_dock(frame, dock_preview(_asset(color=(0, 0, 255)), 0.0, False))
    assert frame.sum() > 0  # the preview + border landed on the frame


def test_render_dock_zero_dim_asset_is_noop():
    # Defensive: a 0-dimension asset must not reach cv2.resize (which would raise
    # and kill the render loop). Unreachable via the real retriever, guarded anyway.
    frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    render_dock(frame, dock_preview(_asset(w=0, h=0), 0.0, False))
    assert frame.sum() == 0  # nothing drawn, no crash


# --------------------------------------------------------------------------- #
# PipelineRuntime lifecycle: inbox -> dock -> composite -> pull -> spawn
# --------------------------------------------------------------------------- #

def test_runtime_drains_inbox_and_docks_preview():
    inbox = PreviewInbox()
    rt = _runtime(inbox=inbox)
    inbox.publish(_asset())
    rt._advance_spawn([_idle(), _idle()], FRAME_SIZE)
    assert rt._preview_state.is_docked is True
    assert rt._preview_state.dismiss_at > 0.0  # dismiss clock started


def test_runtime_new_asset_mid_grab_redocks_without_instant_spawn():
    # A new asset arriving while the current preview is being DRAGGED replaces it
    # (re-docked IDLE) and re-seeds was_pinching from the held pinch, so the still-
    # held pinch is NOT a rising edge and can't instantly spawn the new asset.
    inbox = PreviewInbox()
    rt = _runtime(inbox=inbox)
    asset_a, asset_b = _asset(w=20), _asset(w=30)
    rt._preview_state = PreviewState(
        asset=asset_a, phase=PHASE_GRABBED, grab_point=(0.5, 0.14),
        was_pinching=True, dismiss_at=1e18,
    )
    inbox.publish(asset_b)
    held = _pinch((0.5, 0.14 + PULL_DOWN_DISTANCE + 0.1))  # well past threshold
    grabbing = rt._advance_spawn([held, _idle()], FRAME_SIZE)
    assert rt._preview_state.asset is asset_b  # replaced by the newcomer
    assert rt._preview_state.phase == PHASE_IDLE
    assert len(rt.scene.layers) == 0  # no instant spawn from the held pinch
    assert grabbing is False


def test_runtime_docked_preview_composites_into_vcam_frame():
    inbox = PreviewInbox()
    rt = _runtime(inbox=inbox)
    inbox.publish(_asset(color=(0, 0, 255)))
    tracking = _Tracking([_idle(), _idle()])
    rt._advance_scene(tracking, FRAME_SIZE)  # docks it (single writer path)
    assert rt._preview_state.is_docked
    frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    rt._composite(frame, tracking)  # the same frame fed to preview + virtualcam
    assert frame.sum() > 0  # preview pixels present before any spawn


def test_runtime_pull_down_spawns_layer_and_resets_interaction():
    grabbed = InteractionState(grab=GrabState(is_grabbed=True, was_pinching=True))
    rt = _runtime(interaction=grabbed)
    asset = _asset()
    rt._preview_state = PreviewState(
        asset=asset, phase=PHASE_GRABBED, grab_point=(0.5, 0.14),
        was_pinching=True, dismiss_at=1e18,
    )
    pinch_y = 0.14 + PULL_DOWN_DISTANCE + 0.02
    grabbing = rt._advance_spawn([_pinch((0.5, pinch_y)), _idle()], FRAME_SIZE)
    assert grabbing is False  # spawned -> no longer dragging the preview
    assert len(rt.scene.layers) == 1
    layer = rt.scene.selected_layer
    assert isinstance(layer.source, ArraySource)  # wrapped the asset bgra
    assert abs(layer.transform.center_x - 0.5) < 0.02
    assert abs(layer.transform.center_y - pinch_y) < 0.02
    assert rt._preview_state.is_docked is False  # preview cleared
    assert rt.interaction.grab.is_grabbed is False  # interaction reset to idle


def test_runtime_grabbing_preview_suppresses_grabbing_a_scene_layer():
    # An existing layer near the top; a pinch over it while the preview is being
    # dragged must NOT grab it (spawn drag has precedence).
    scene = Scene.single(Layer(source=_FakeSource((40, 40)), transform=Transform(0.5, 0.2)))
    rt = _runtime(scene=scene)
    rt._preview_state = PreviewState(
        asset=_asset(), phase=PHASE_GRABBED, grab_point=(0.5, 0.15),
        was_pinching=True, dismiss_at=1e18,
    )
    over_layer = _pinch((0.5, 0.2))  # short pull -> still GRABBED; over the layer
    rt._advance_scene(_Tracking([over_layer, _idle()]), FRAME_SIZE)
    assert rt.interaction.grab.is_grabbed is False  # suppressed: layer not grabbed
    assert rt._preview_state.is_grabbing is True     # still dragging the preview


def test_runtime_pinch_grabs_scene_layer_when_no_preview_is_dragged():
    # Control for the precedence test: the identical pinch DOES grab the layer
    # when nothing is being dragged (proves the suppression is the cause).
    scene = Scene.single(Layer(source=_FakeSource((40, 40)), transform=Transform(0.5, 0.2)))
    rt = _runtime(scene=scene)  # no preview docked
    rt._advance_scene(_Tracking([_pinch((0.5, 0.2)), _idle()]), FRAME_SIZE)
    assert rt.interaction.grab.is_grabbed is True


# --------------------------------------------------------------------------- #
# build_speech_on_phrase inbox wiring (Phase 15 seam)
# --------------------------------------------------------------------------- #

def test_build_speech_on_phrase_publishes_asset_to_inbox(monkeypatch, capsys):
    import src.parser as parser_module
    import src.retriever as retriever_module
    from src.launch import build_speech_on_phrase
    from src.parser import ParsedRequest

    asset = _FakeAsset(_solid_bgra(6, 4), source_name="wikimedia")

    class _Parser:
        def parse(self, text):
            return ParsedRequest("insert", "CFTR", "retrieve", "wikimedia")

    class _Retriever:
        def retrieve(self, request):
            return asset

    monkeypatch.setattr(parser_module, "build_request_parser", lambda: _Parser())
    monkeypatch.setattr(retriever_module, "build_retriever", lambda *a, **k: _Retriever())

    inbox = PreviewInbox()
    on_phrase = build_speech_on_phrase(True, inbox)
    on_phrase("we're looking at CFTR")
    out = capsys.readouterr().out
    assert "[retriever] got 6x4 image" in out  # still logs (Phase 14 behavior kept)
    assert inbox.take() is asset               # AND publishes to the inbox


def test_build_speech_on_phrase_without_inbox_is_log_only(monkeypatch, capsys):
    import src.parser as parser_module
    import src.retriever as retriever_module
    from src.launch import build_speech_on_phrase
    from src.parser import ParsedRequest

    class _Parser:
        def parse(self, text):
            return ParsedRequest("insert", "CFTR", "retrieve", "wikimedia")

    class _Retriever:
        def retrieve(self, request):
            return _FakeAsset(_solid_bgra(6, 4))

    monkeypatch.setattr(parser_module, "build_request_parser", lambda: _Parser())
    monkeypatch.setattr(retriever_module, "build_retriever", lambda *a, **k: _Retriever())

    on_phrase = build_speech_on_phrase(True)  # no inbox -> behaves as before
    on_phrase("we're looking at CFTR")  # must not raise
    assert "[retriever] got 6x4 image" in capsys.readouterr().out

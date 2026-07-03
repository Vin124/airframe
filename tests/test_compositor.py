"""Unit tests for src.compositor: alpha blend, edge clipping, and z-ordered
compositing of scene layers. Pure/headless -- synthetic numpy frames only.
"""

import numpy as np
import pytest

from src.compositor import draw_selection_highlight, render
from src.scene import Layer, Scene, Transform
from src.sources.base import Source
from src.sources.image import ImageSource

FRAME_SIZE = (100, 100)  # (width, height)

_HIGHLIGHT_COLOR = (0, 165, 255)  # must match src.compositor._HIGHLIGHT_COLOR


def _solid_image(size=(10, 10), bgr=(200, 160, 80), alpha=255) -> np.ndarray:
    """A ``size`` (w,h) BGRA image filled with one color/alpha."""
    w, h = size
    img = np.zeros((h, w, 4), dtype=np.uint8)
    img[:, :, 0] = bgr[0]
    img[:, :, 1] = bgr[1]
    img[:, :, 2] = bgr[2]
    img[:, :, 3] = alpha
    return img


def _layer(size=(10, 10), bgr=(200, 160, 80), alpha=255, transform=None, z_index=0) -> Layer:
    return Layer(
        source=ImageSource(_solid_image(size, bgr, alpha)),
        transform=transform if transform is not None else Transform(),
        z_index=z_index,
        selected=True,
    )


def _blank_frame(color=(0, 0, 0)) -> np.ndarray:
    frame = np.zeros((FRAME_SIZE[1], FRAME_SIZE[0], 3), dtype=np.uint8)
    frame[:, :] = color
    return frame


class _NoneSource(Source):
    """A layer source with no frame available yet."""

    def read(self):
        return None

    @property
    def base_size(self):
        return (10, 10)


# --------------------------------------------------------------------------- #
# Single-layer alpha compositing
# --------------------------------------------------------------------------- #

def test_composite_opaque_inside_paints_and_leaves_outside_untouched():
    frame = _blank_frame((0, 0, 0))
    render(frame, Scene.single(_layer(bgr=(10, 20, 30), alpha=255)))  # rect px [45,55)
    assert tuple(int(v) for v in frame[50, 50]) == (10, 20, 30)  # inside
    assert tuple(int(v) for v in frame[0, 0]) == (0, 0, 0)       # outside untouched


def test_composite_alpha_blends_known_pixel():
    alpha = 128
    frame = _blank_frame((0, 0, 0))
    render(frame, Scene.single(_layer(bgr=(200, 160, 80), alpha=alpha)))

    ratio = alpha / 255.0
    expected = tuple(int(ratio * c) for c in (200, 160, 80))
    got = tuple(int(v) for v in frame[50, 50])
    assert all(abs(g - e) <= 1 for g, e in zip(got, expected))


@pytest.mark.parametrize(
    "center, probe",
    [
        ((0.0, 0.5), (50, 2)),    # clipped at left edge, right half visible
        ((1.0, 0.5), (50, 97)),   # clipped at right edge
        ((0.5, 0.0), (2, 50)),    # clipped at top edge
        ((0.5, 1.0), (97, 50)),   # clipped at bottom edge
    ],
)
def test_composite_partially_offscreen_clips_without_crashing(center, probe):
    frame = _blank_frame((0, 0, 0))
    layer = _layer(bgr=(11, 22, 33), transform=Transform(center_x=center[0], center_y=center[1]))
    render(frame, Scene.single(layer))
    row, col = probe
    assert tuple(int(v) for v in frame[row, col]) == (11, 22, 33)


def test_composite_fully_offscreen_is_a_noop():
    # A 1px scaled image parked past the last column -> no overlap.
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    original = frame.copy()
    layer = Layer(
        source=ImageSource(_solid_image(size=(10, 10), bgr=(255, 255, 255))),
        transform=Transform(center_x=1.0, center_y=1.0, scale=0.1),
        selected=True,
    )
    render(frame, Scene.single(layer))
    assert np.array_equal(frame, original)


# --------------------------------------------------------------------------- #
# Scene-level compositing (z-order, empty scene, unavailable source)
# --------------------------------------------------------------------------- #

def test_empty_scene_leaves_background_untouched():
    frame = _blank_frame((7, 8, 9))
    original = frame.copy()
    result = render(frame, Scene())
    assert result is frame  # same array back
    assert np.array_equal(frame, original)


def test_higher_z_index_paints_on_top():
    # Two overlapping opaque layers at the center; the higher z_index wins.
    back = _layer(bgr=(10, 10, 10), alpha=255, z_index=0)
    front = _layer(bgr=(90, 90, 90), alpha=255, z_index=1)
    frame = _blank_frame((0, 0, 0))
    # Pass them in "front first" so list order != z order, proving z drives it.
    render(frame, Scene(layers=(front, back)))
    assert tuple(int(v) for v in frame[50, 50]) == (90, 90, 90)


def test_lower_z_index_is_covered_by_higher():
    back = _layer(bgr=(10, 10, 10), alpha=255, z_index=5)
    front = _layer(bgr=(90, 90, 90), alpha=255, z_index=0)
    frame = _blank_frame((0, 0, 0))
    render(frame, Scene(layers=(back, front)))
    # z=5 layer 'back' paints last -> it wins at the overlap.
    assert tuple(int(v) for v in frame[50, 50]) == (10, 10, 10)


def test_layer_with_no_frame_is_skipped():
    frame = _blank_frame((3, 3, 3))
    original = frame.copy()
    layer = Layer(source=_NoneSource(), transform=Transform(), selected=True)
    render(frame, Scene(layers=(layer,)))
    assert np.array_equal(frame, original)  # None source -> skipped, no crash


# --------------------------------------------------------------------------- #
# Scaled compositing (covers both interpolation branches)
# --------------------------------------------------------------------------- #

def test_composite_upscaled_layer_covers_enlarged_rect():
    # scale 2.0 -> 20x20 rect [40,60) via the INTER_LINEAR branch.
    frame = _blank_frame((0, 0, 0))
    layer = _layer(bgr=(40, 50, 60), transform=Transform(scale=2.0))
    render(frame, Scene.single(layer))
    assert tuple(int(v) for v in frame[45, 45]) == (40, 50, 60)  # inside grown rect
    assert tuple(int(v) for v in frame[38, 38]) == (0, 0, 0)     # outside it


def test_composite_downscaled_layer_shrinks_rect():
    # scale 0.5 -> 5x5 rect [48,53) via the INTER_AREA branch.
    frame = _blank_frame((0, 0, 0))
    layer = _layer(bgr=(40, 50, 60), transform=Transform(scale=0.5))
    render(frame, Scene.single(layer))
    assert tuple(int(v) for v in frame[50, 50]) == (40, 50, 60)  # inside shrunk rect
    assert tuple(int(v) for v in frame[45, 45]) == (0, 0, 0)     # scale-1.0 pixel now bare


# --------------------------------------------------------------------------- #
# Selection highlight (Phase 8): thin box around the selected layer
# --------------------------------------------------------------------------- #

class _SizelessSource(Source):
    """A layer source with no frame yet, so its rect is undefined."""

    def read(self):
        return None

    @property
    def base_size(self):
        return None


def test_draw_selection_highlight_marks_selected_layer():
    frame = _blank_frame((0, 0, 0))
    original = frame.copy()
    scene = Scene.single(_layer(bgr=(10, 20, 30)))  # centered 10x10 rect
    result = draw_selection_highlight(frame, scene)
    assert result is frame  # same array back (chaining)
    assert not np.array_equal(frame, original)  # something was drawn
    # The exact highlight color appears (cv2.rectangle isn't anti-aliased here).
    assert np.any(np.all(frame == _HIGHLIGHT_COLOR, axis=2))


def test_draw_selection_highlight_noop_without_selection():
    frame = _blank_frame((0, 0, 0))
    original = frame.copy()
    draw_selection_highlight(frame, Scene())  # nothing selected
    assert np.array_equal(frame, original)


def test_draw_selection_highlight_noop_when_size_unknown():
    frame = _blank_frame((0, 0, 0))
    original = frame.copy()
    scene = Scene(layers=(Layer(source=_SizelessSource(), selected=True),))
    draw_selection_highlight(frame, scene)  # base_size None -> no rect -> skip
    assert np.array_equal(frame, original)

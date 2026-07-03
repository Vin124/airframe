"""Unit tests for src.scene: Transform math/clamping, layer geometry + hit-test,
and Layer/Scene ordering & selection. All pure/headless -- no webcam, no window.
"""

from typing import Optional

import numpy as np
import pytest

from src.scene import (
    CENTER_MAX,
    CENTER_MIN,
    MAX_SCALE,
    MIN_SCALE,
    Layer,
    Scene,
    Transform,
    contains,
    rect_px,
    scaled_size,
    top_left_px,
)
from src.sources.base import Size, Source

FRAME_SIZE = (100, 100)  # (width, height)


class _FakeSource(Source):
    """Minimal Source whose only meaningful attribute is base_size."""

    def __init__(self, base_size: Size, frame: Optional[np.ndarray] = None) -> None:
        self._base_size = base_size
        self._frame = frame

    def read(self):
        return self._frame

    @property
    def base_size(self) -> Size:
        return self._base_size


def _layer(size=(10, 10), transform=None, z_index=0, selected=False) -> Layer:
    return Layer(
        source=_FakeSource(size),
        transform=transform if transform is not None else Transform(),
        z_index=z_index,
        selected=selected,
    )


# --------------------------------------------------------------------------- #
# Transform math + clamping
# --------------------------------------------------------------------------- #

def test_default_transform_is_centered_unit_scale():
    t = Transform()
    assert t.center_x == 0.5
    assert t.center_y == 0.5
    assert t.scale == 1.0


def test_moved_to_clamps_center_into_frame():
    t = Transform().moved_to(2.0, -1.0)
    assert t.center_x == CENTER_MAX
    assert t.center_y == CENTER_MIN


def test_moved_to_preserves_scale():
    t = Transform(scale=2.0).moved_to(0.3, 0.7)
    assert t.scale == 2.0
    assert (t.center_x, t.center_y) == (0.3, 0.7)


def test_scaled_by_clamps_to_max():
    t = Transform(scale=1.0).scaled_by(1000.0)
    assert t.scale == MAX_SCALE


def test_scaled_by_clamps_to_min():
    t = Transform(scale=1.0).scaled_by(0.00001)
    assert t.scale == MIN_SCALE


def test_scaled_by_preserves_center():
    t = Transform(center_x=0.2, center_y=0.8, scale=1.0).scaled_by(2.0)
    assert (t.center_x, t.center_y) == (0.2, 0.8)
    assert t.scale == 2.0


def test_with_scale_is_absolute_and_clamped():
    assert Transform(scale=1.0).with_scale(3.0).scale == 3.0
    assert Transform(scale=1.0).with_scale(-5.0).scale == MIN_SCALE


def test_scaled_size_multiplies_and_floors_to_one():
    assert scaled_size((10, 20), 2.0) == (20, 40)
    # A tiny scale must never produce a zero dimension.
    assert scaled_size((10, 10), 0.0001) == (1, 1)


def test_top_left_px_centers_the_image():
    # 10x10 centered on a 100x100 frame -> top-left at (45, 45).
    assert top_left_px(Transform(), (10, 10), FRAME_SIZE) == (45, 45)


def test_rect_px_reports_scaled_rect():
    assert rect_px(Transform(), (10, 10), FRAME_SIZE) == (45, 45, 10, 10)
    # Double the scale doubles size and keeps it centered.
    assert rect_px(Transform(scale=2.0), (10, 10), FRAME_SIZE) == (40, 40, 20, 20)


# --------------------------------------------------------------------------- #
# contains() hit-test (module function + Layer method)
# --------------------------------------------------------------------------- #

def test_contains_center_point_is_inside():
    assert contains(Transform(), (10, 10), (0.5, 0.5), FRAME_SIZE) is True


def test_contains_top_left_edge_is_inclusive():
    # rect px [45,55) x [45,55)
    assert contains(Transform(), (10, 10), (0.45, 0.45), FRAME_SIZE) is True


def test_contains_bottom_right_edge_is_exclusive():
    assert contains(Transform(), (10, 10), (0.55, 0.55), FRAME_SIZE) is False


def test_contains_point_just_outside_left_is_false():
    assert contains(Transform(), (10, 10), (0.44, 0.5), FRAME_SIZE) is False


def test_contains_far_point_is_false():
    assert contains(Transform(), (10, 10), (0.9, 0.9), FRAME_SIZE) is False


def test_layer_contains_delegates_to_transform_and_base_size():
    layer = _layer(size=(10, 10))
    assert layer.contains((0.5, 0.5), FRAME_SIZE) is True
    assert layer.contains((0.9, 0.9), FRAME_SIZE) is False


def test_layer_rect_px_uses_source_base_size():
    assert _layer(size=(10, 10)).rect_px(FRAME_SIZE) == (45, 45, 10, 10)


def test_layer_base_size_delegates_to_source():
    assert _layer(size=(7, 3)).base_size == (7, 3)


# --------------------------------------------------------------------------- #
# Layer immutability / functional updates
# --------------------------------------------------------------------------- #

def test_layer_with_transform_returns_new_copy():
    layer = _layer()
    moved = layer.with_transform(Transform(center_x=0.2, center_y=0.8))
    assert moved is not layer
    assert (moved.transform.center_x, moved.transform.center_y) == (0.2, 0.8)
    # Original is untouched (immutability).
    assert (layer.transform.center_x, layer.transform.center_y) == (0.5, 0.5)
    # Source/z/selection preserved.
    assert moved.source is layer.source
    assert moved.z_index == layer.z_index
    assert moved.selected == layer.selected


def test_layer_with_selected_and_with_z_index():
    layer = _layer()
    assert layer.with_selected(True).selected is True
    assert layer.with_z_index(5).z_index == 5
    # Original unchanged.
    assert layer.selected is False
    assert layer.z_index == 0


# --------------------------------------------------------------------------- #
# Scene ordering + selection
# --------------------------------------------------------------------------- #

def test_empty_scene_has_no_layers_or_selection():
    scene = Scene()
    assert scene.layers == ()
    assert scene.selected_layer is None
    assert scene.layers_by_z() == ()


def test_scene_single_marks_layer_selected():
    scene = Scene.single(_layer())
    assert len(scene.layers) == 1
    assert scene.layers[0].selected is True
    assert scene.selected_layer is scene.layers[0]


def test_layers_by_z_orders_ascending_and_is_stable():
    back = _layer(z_index=0)
    mid_a = _layer(z_index=1)
    mid_b = _layer(z_index=1)  # equal z -> keep insertion order (stable sort)
    front = _layer(z_index=5)
    scene = Scene(layers=(front, mid_a, back, mid_b))
    ordered = scene.layers_by_z()
    assert [layer.z_index for layer in ordered] == [0, 1, 1, 5]
    # Stable: mid_a stays before mid_b despite equal z_index.
    assert ordered[1] is mid_a
    assert ordered[2] is mid_b


def test_selected_layer_returns_first_selected():
    a = _layer(selected=False)
    b = _layer(selected=True)
    scene = Scene(layers=(a, b))
    assert scene.selected_layer is b


def test_replace_selected_swaps_only_the_selected_layer():
    a = _layer(selected=False, z_index=0)
    b = _layer(selected=True, z_index=1)
    scene = Scene(layers=(a, b))
    new_b = b.with_transform(Transform(center_x=0.1, center_y=0.1))
    updated = scene.replace_selected(new_b)
    assert updated.layers[0] is a  # untouched
    assert updated.layers[1] is new_b
    # Original scene unchanged (immutability).
    assert scene.layers[1] is b


def test_replace_selected_is_noop_when_nothing_selected():
    scene = Scene(layers=(_layer(selected=False),))
    updated = scene.replace_selected(_layer())
    assert updated.layers == scene.layers


def test_with_layers_replaces_layer_tuple():
    scene = Scene()
    layer = _layer(selected=True)
    updated = scene.with_layers((layer,))
    assert updated.layers == (layer,)
    assert scene.layers == ()  # original untouched


def test_layer_contains_is_false_when_source_has_no_size():
    # A source with no frame yet (base_size None) has no rect, so nothing hits it.
    layer = Layer(source=_FakeSource(None))
    assert layer.contains((0.5, 0.5), FRAME_SIZE) is False


def test_layer_rect_px_raises_when_source_has_no_size():
    layer = Layer(source=_FakeSource(None))
    with pytest.raises(ValueError):
        layer.rect_px(FRAME_SIZE)

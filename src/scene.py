"""The scene graph: ``Transform`` geometry, ``Layer``, and ``Scene``.

A ``Scene`` is an ordered set of ``Layer`` objects the compositor paints over
the webcam background. Each ``Layer`` pairs a frame ``Source`` with a
``Transform`` (where/how big it is), a ``z_index`` (paint order), and a
``selected`` flag (which layer interaction currently drives).

Everything here is immutable (frozen dataclasses + functional updates), per the
project's hard immutability rule: advancing a frame produces a *new* scene, so
there's no hidden shared state between frames and every transition is trivially
testable without a camera.

Coordinates are normalized 0-1 (center + scale multiplier); pixel conversion
happens only in the geometry helpers below, called at composite/hit-test time.

Design note (deviation from the phase prompt): ``Transform`` lives here rather
than in ``interaction.py``. A ``Layer`` must embed a ``Transform``, so putting
it here keeps the dependency acyclic -- ``interaction`` imports ``scene``, never
the reverse. The grab/move/resize *state machines* that mutate transforms are
what live in ``interaction.py``.
"""

from dataclasses import dataclass, field, replace
from typing import Optional, Tuple

from src.sources.base import Size, Source

Point2D = Tuple[float, float]

# Scale is a multiplier on a layer's native pixel size. Clamped so a layer can
# neither vanish nor swallow the whole frame.
MIN_SCALE = 0.1
MAX_SCALE = 6.0
DEFAULT_SCALE = 1.0

# Center is clamped to the frame so a layer can't be dragged fully off-screen:
# the center point always stays inside the frame, keeping part of it visible.
CENTER_MIN = 0.0
CENTER_MAX = 1.0

# A layer's ``layer_id`` before ``Scene.add_layer`` stamps a real, monotonically
# increasing one onto it. Handles bare ``Layer(...)`` built outside a scene.
UNASSIGNED_ID = -1


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass(frozen=True)
class Transform:
    """Immutable layer transform: center (normalized) + scale multiplier.

    Values are clamped on construction, so every instance is valid and the
    functional update helpers below don't need to re-clamp.
    """

    center_x: float = 0.5
    center_y: float = 0.5
    scale: float = DEFAULT_SCALE

    def __post_init__(self) -> None:
        object.__setattr__(self, "center_x", _clamp(self.center_x, CENTER_MIN, CENTER_MAX))
        object.__setattr__(self, "center_y", _clamp(self.center_y, CENTER_MIN, CENTER_MAX))
        object.__setattr__(self, "scale", _clamp(self.scale, MIN_SCALE, MAX_SCALE))

    def moved_to(self, center_x: float, center_y: float) -> "Transform":
        """Return a copy centered at ``(center_x, center_y)`` (normalized, clamped)."""
        return Transform(center_x, center_y, self.scale)

    def scaled_by(self, factor: float) -> "Transform":
        """Return a copy with the scale multiplied by ``factor`` (clamped)."""
        return Transform(self.center_x, self.center_y, self.scale * factor)

    def with_scale(self, scale: float) -> "Transform":
        """Return a copy with an absolute ``scale`` (clamped)."""
        return Transform(self.center_x, self.center_y, scale)


def scaled_size(base_size: Size, scale: float) -> Size:
    """Return a layer's pixel size at ``scale``, floored to at least 1x1."""
    base_w, base_h = base_size
    return max(1, int(round(base_w * scale))), max(1, int(round(base_h * scale)))


def top_left_px(transform: Transform, scaled: Size, frame_size: Size) -> Tuple[int, int]:
    """Return the top-left pixel of the scaled layer for ``transform``."""
    scaled_w, scaled_h = scaled
    frame_w, frame_h = frame_size
    cx = transform.center_x * frame_w
    cy = transform.center_y * frame_h
    return int(round(cx - scaled_w / 2.0)), int(round(cy - scaled_h / 2.0))


def rect_px(
    transform: Transform, base_size: Size, frame_size: Size
) -> Tuple[int, int, int, int]:
    """Return a layer's on-frame rect as ``(x0, y0, width, height)`` in pixels."""
    scaled_w, scaled_h = scaled_size(base_size, transform.scale)
    x0, y0 = top_left_px(transform, (scaled_w, scaled_h), frame_size)
    return x0, y0, scaled_w, scaled_h


def contains(
    transform: Transform, base_size: Size, point: Point2D, frame_size: Size
) -> bool:
    """Hit-test: is the normalized ``point`` inside the layer's current rect?

    Top/left edges are inclusive, bottom/right exclusive (half-open pixel rect),
    matching the compositing region so a pinch on the visible edge counts.
    """
    x0, y0, w, h = rect_px(transform, base_size, frame_size)
    frame_w, frame_h = frame_size
    px = point[0] * frame_w
    py = point[1] * frame_h
    return x0 <= px < x0 + w and y0 <= py < y0 + h


@dataclass(frozen=True)
class Layer:
    """One composited element: a frame ``source`` plus its placement.

    ``transform`` is the geometry (center + scale); ``z_index`` sets paint order
    (higher paints later, i.e. on top); ``selected`` marks the layer that
    interaction currently drives. ``layer_id`` is a stable identity assigned by
    ``Scene.add_layer`` (``UNASSIGNED_ID`` until then) so remove/raise/fit can
    target a specific layer across the functional copies grab/move produce --
    tests key off ``layer_id`` rather than fragile object identity. Immutable --
    the ``with_*`` helpers return updated copies.
    """

    source: Source
    transform: Transform = field(default_factory=Transform)
    z_index: int = 0
    selected: bool = False
    layer_id: int = UNASSIGNED_ID

    @property
    def base_size(self) -> Optional[Size]:
        """The layer's native (width, height), delegated to its source."""
        return self.source.base_size

    def with_transform(self, transform: Transform) -> "Layer":
        """Return a copy with a replaced transform (keeps source/z/id/selection)."""
        return replace(self, transform=transform)

    def with_selected(self, selected: bool) -> "Layer":
        """Return a copy with the ``selected`` flag set."""
        return replace(self, selected=selected)

    def with_z_index(self, z_index: int) -> "Layer":
        """Return a copy with a replaced ``z_index``."""
        return replace(self, z_index=z_index)

    def rect_px(self, frame_size: Size) -> Tuple[int, int, int, int]:
        """This layer's on-frame ``(x0, y0, w, h)`` rect in pixels.

        Raises ``ValueError`` when the source has no frame yet (``base_size``
        is ``None``) -- a size-less layer has no rect, and callers that need
        one should skip the layer instead.
        """
        base_size = self.base_size
        if base_size is None:
            raise ValueError("layer source has no frame yet; its rect is undefined")
        return rect_px(self.transform, base_size, frame_size)

    def contains(self, point: Point2D, frame_size: Size) -> bool:
        """Is the normalized ``point`` inside this layer's current rect?

        A layer whose source has no frame yet (``base_size`` is ``None``) has
        no rect, so nothing hits it.
        """
        base_size = self.base_size
        if base_size is None:
            return False
        return contains(self.transform, base_size, point, frame_size)


@dataclass(frozen=True)
class Scene:
    """An ordered set of layers painted over the webcam background.

    Compositing order is strictly by ``z_index`` (see ``layers_by_z``), not list
    order, so raising a layer is just a ``z_index`` change. At most one layer is
    ``selected`` (the one interaction drives); ``select_layer`` / ``top_layer_at``
    move the selection, ``add_layer`` / ``remove_selected`` / ``raise_selected``
    edit the set. ``next_id`` is the source of the monotonically increasing
    ``layer_id`` stamped onto each added layer, so identity survives the
    functional copies grab/move/resize produce.
    """

    layers: Tuple[Layer, ...] = ()
    next_id: int = 0

    def layers_by_z(self) -> Tuple[Layer, ...]:
        """Layers ordered back-to-front (ascending ``z_index``, stable)."""
        return tuple(sorted(self.layers, key=lambda layer: layer.z_index))

    @property
    def selected_layer(self) -> Optional[Layer]:
        """The first layer flagged ``selected``, or ``None`` if none is."""
        for layer in self.layers:
            if layer.selected:
                return layer
        return None

    def layer_by_id(self, layer_id: int) -> Optional[Layer]:
        """The layer whose ``layer_id`` matches, or ``None`` if none does."""
        for layer in self.layers:
            if layer.layer_id == layer_id:
                return layer
        return None

    def with_layers(self, layers: Tuple[Layer, ...]) -> "Scene":
        """Return a copy whose layers are replaced wholesale (keeps ``next_id``)."""
        return replace(self, layers=tuple(layers))

    def replace_selected(self, new_layer: Layer) -> "Scene":
        """Return a copy with the selected layer swapped for ``new_layer``.

        No-op (returns ``self``) when nothing is selected. Only the first
        selected layer is replaced -- at most one is selected at a time.
        """
        for i, layer in enumerate(self.layers):
            if layer.selected:
                layers = self.layers[:i] + (new_layer,) + self.layers[i + 1 :]
                return replace(self, layers=layers)
        return self

    def update_layer(self, layer_id: int, new_layer: Layer) -> "Scene":
        """Return a copy with the layer identified by ``layer_id`` replaced.

        No-op (returns a copy of the same layers) when no layer has that id.
        Used to update a layer that isn't necessarily the selected one (e.g. a
        just-added live layer getting its one-time initial fit).
        """
        layers = tuple(
            new_layer if layer.layer_id == layer_id else layer for layer in self.layers
        )
        return replace(self, layers=layers)

    def select_layer(self, index: int) -> "Scene":
        """Return a copy with exactly the layer at list ``index`` selected.

        Every other layer is deselected, preserving the at-most-one-selected
        invariant. An out-of-range ``index`` simply deselects everything.
        """
        layers = tuple(
            layer.with_selected(i == index) for i, layer in enumerate(self.layers)
        )
        return replace(self, layers=layers)

    def top_layer_at(self, point: Point2D, frame_size: Size) -> Optional[int]:
        """List index of the top-most layer whose rect contains ``point``.

        "Top-most" is the layer painted last at that spot: highest ``z_index``,
        ties broken by later list position (matching ``layers_by_z``'s stable
        sort, where an equal-z layer later in the list paints on top). Returns
        ``None`` when no layer contains the point (a pinch over bare background).
        Size-less layers (``base_size`` None) never contain anything.
        """
        top_index: Optional[int] = None
        top_key: Optional[Tuple[int, int]] = None
        for i, layer in enumerate(self.layers):
            if layer.contains(point, frame_size):
                key = (layer.z_index, i)
                if top_key is None or key > top_key:
                    top_key, top_index = key, i
        return top_index

    def add_layer(self, layer: Layer) -> "Scene":
        """Return a copy with ``layer`` added on top, selected, and given an id.

        The new layer is stamped with a fresh monotonic ``layer_id`` (from
        ``next_id``), placed at ``z_index = max(existing) + 1`` (front-most), and
        marked selected while every existing layer is deselected. Its source and
        transform are kept; its incoming ``z_index`` / ``selected`` / ``layer_id``
        are overwritten by this placement.
        """
        top_z = max((existing.z_index for existing in self.layers), default=-1)
        placed = replace(layer, layer_id=self.next_id, z_index=top_z + 1, selected=True)
        deselected = tuple(existing.with_selected(False) for existing in self.layers)
        return replace(self, layers=deselected + (placed,), next_id=self.next_id + 1)

    def remove_selected(self) -> "Scene":
        """Return a copy with the selected layer removed, then re-select the top.

        No-op (returns ``self``) when nothing is selected. After removal the new
        top-most remaining layer becomes selected; removing the last layer leaves
        an empty, selection-less scene (legal at runtime). ``next_id`` is kept so
        ids never repeat even after removals.
        """
        kept = tuple(layer for layer in self.layers if not layer.selected)
        if len(kept) == len(self.layers):
            return self
        return replace(self, layers=kept)._select_top()

    def raise_selected(self) -> "Scene":
        """Return a copy with the selected layer's ``z_index`` above all others.

        No-op (returns ``self``) when nothing is selected. Sets the selected
        layer's ``z_index`` to ``max(all) + 1`` so it paints last (front-most).
        """
        layer = self.selected_layer
        if layer is None:
            return self
        top_z = max(existing.z_index for existing in self.layers)
        return self.replace_selected(layer.with_z_index(top_z + 1))

    def _select_top(self) -> "Scene":
        """Select the front-most layer (highest ``z_index``, then list order)."""
        if not self.layers:
            return self
        top_index = max(
            range(len(self.layers)),
            key=lambda i: (self.layers[i].z_index, i),
        )
        return self.select_layer(top_index)

    @classmethod
    def single(cls, layer: Layer) -> "Scene":
        """Build a one-layer scene with ``layer`` added, selected, and id-stamped.

        Convenience for the launch-time single-layer scene; runtime multi-layer
        edits go through ``add_layer`` / ``remove_selected`` directly.
        """
        return cls().add_layer(layer)

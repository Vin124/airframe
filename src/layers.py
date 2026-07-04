"""Layer/source construction and the scene-transition helpers built on them.

This module owns everything about turning a chosen overlay *source* (a static
image, a live app window, or a monitor screen) into a scene ``Layer`` and the
pure transitions that add/remove/fit those layers -- the logic behind the
image-mode hotkeys ('a'/'w'/'m' add, 'x' remove, 'f' front) and the control
panel's add/reset buttons. It also owns the mode/source/layer-key *vocabulary*
(``MODE_*``, ``SOURCE_*``, the add/remove key codes, the initial-fit fraction)
because both the entry point (``src.main``: argument parsing + the runtime) and
``src.launch`` (panel prefill) share it, and this is the lowest module in the
dependency graph they can all import from.

Everything here is camera-free and side-effect-light (a failed add prints to
stderr and returns the scene unchanged; nothing exits), so the transitions are
unit-testable without a webcam or a real capture source. To keep the graph
acyclic this module imports only from ``src.scene`` / ``src.sources`` /
``src.interaction`` / the stdlib -- never from ``src.main``. ``src.main``
re-exports these names so ``from src.main import _build_layer_source`` (etc.)
keeps working; tests that monkeypatch the capture classes patch
``src.layers.WindowSource`` / ``src.layers.MonitorSource``.
"""

import argparse
import sys
from typing import Callable, Iterable, Optional, Set, Tuple

from src.interaction import InteractionState
from src.scene import DEFAULT_SCALE, Layer, Scene
from src.sources import (
    ImageSource,
    MonitorSource,
    Source,
    SourceError,
    WindowSource,
)

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

# Image-mode multi-layer keys (Phase 8). 'a'/'w'/'m' add an image / window /
# screen layer, reusing the launch args (--image / --window-title /
# --monitor-index) for the new source; each new layer lands on top and becomes
# selected. 'x' removes the selected layer (selecting the new top-most); 'f'
# brings the selected layer to the front. Chosen to avoid the other runtime keys.
ADD_IMAGE_KEY = ord("a")
ADD_WINDOW_KEY = ord("w")
ADD_SCREEN_KEY = ord("m")
REMOVE_LAYER_KEY = ord("x")
BRING_TO_FRONT_KEY = ord("f")


# --------------------------------------------------------------------------- #
# Explicit-value source builders. These are the single implementation; the
# hotkey path wraps them to read the launch args, and the panel path passes
# values from its fields (see PipelineRuntime.add_*_layer).
# --------------------------------------------------------------------------- #

def _image_source_from_path(image_path: Optional[str]) -> Source:
    """Build an image overlay source from an explicit path.

    Raises ``SourceError`` when the path is empty (so the caller can surface a
    notice and keep running) or when the path can't be loaded.
    """
    if not image_path:
        raise SourceError("Cannot add an image layer: no image path provided.")
    return ImageSource.load(image_path)


def _window_source_from_title(window_title: Optional[str]) -> Source:
    """Build a live window source from an explicit title substring.

    Raises ``SourceError`` when the title is empty, or (from ``open()``) when no
    visible window matches it.
    """
    if not window_title:
        raise SourceError("Cannot add a window layer: no window title provided.")
    return WindowSource(window_title).open()


def _monitor_source_from_index(monitor_index: int) -> Source:
    """Build a live monitor source from an explicit 1-based index.

    Raises ``SourceError`` (from ``open()``) on a bad monitor index.
    """
    return MonitorSource(monitor_index).open()


def _build_image_layer_source(args: argparse.Namespace) -> Source:
    """Hotkey ('a') wrapper: build an image source from ``--image``."""
    return _image_source_from_path(args.image)


def _build_window_layer_source(args: argparse.Namespace) -> Source:
    """Hotkey ('w') wrapper: build a window source from ``--window-title``."""
    return _window_source_from_title(args.window_title)


def _build_monitor_layer_source(args: argparse.Namespace) -> Source:
    """Hotkey ('m') wrapper: build a monitor source from ``--monitor-index``."""
    return _monitor_source_from_index(args.monitor_index)


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


# Dispatch for the add-layer keys: key -> (source builder, is-live-source). Live
# sources need a lazy initial fit; image sources already know their size.
_ADD_LAYER_BUILDERS = {
    ADD_IMAGE_KEY: (_build_image_layer_source, False),
    ADD_WINDOW_KEY: (_build_window_layer_source, True),
    ADD_SCREEN_KEY: (_build_monitor_layer_source, True),
}


def _add_source_layer(
    scene: Scene,
    interaction: InteractionState,
    fit_ids: Set[int],
    build_source: Callable[[], Source],
    live: bool,
) -> Tuple[Scene, InteractionState, Set[int], Optional[str]]:
    """Core add-layer transition shared by the hotkey and panel paths.

    Returns ``(scene, interaction, fit_ids, status)`` where ``status`` is the
    ``SourceError`` message on failure (scene unchanged) or ``None`` on success.
    Deliberately does NOT print or exit -- adding a layer is a runtime action, so
    the callers decide how to surface ``status`` (stderr for the hotkey; stderr +
    the panel's status label for the panel). On success the new layer is selected
    and interaction is reset to idle so an in-progress grab on the previously-
    selected layer can't leak onto the new one; a live layer's id joins
    ``fit_ids`` for its one-time fit.
    """
    try:
        source = build_source()
    except SourceError as exc:
        return scene, interaction, fit_ids, str(exc)
    scene = scene.add_layer(Layer(source=source))
    if live:
        fit_ids = fit_ids | {scene.selected_layer.layer_id}
    return scene, InteractionState.initial(), fit_ids, None


def _add_layer(
    scene: Scene,
    interaction: InteractionState,
    fit_ids: Set[int],
    build_source: Callable[[], Source],
    live: bool,
) -> Tuple[Scene, InteractionState, Set[int]]:
    """Hotkey add-layer: like ``_add_source_layer`` but prints failures to stderr.

    A ``SourceError`` is printed and the scene is left unchanged (the app keeps
    running, same principle as the virtual-cam 'v' toggle).
    """
    scene, interaction, fit_ids, error = _add_source_layer(
        scene, interaction, fit_ids, build_source, live
    )
    if error is not None:
        print(f"Error: {error}", file=sys.stderr)
    return scene, interaction, fit_ids


def _close_layer_sources(layers: Iterable[Layer]) -> None:
    """``close()`` every layer's source (stops live capture threads; no-op for images)."""
    for layer in layers:
        layer.source.close()


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
    _close_layer_sources((layer,))
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



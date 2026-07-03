"""Compositor: paint a scene's layers over the webcam background.

The webcam frame is the background; each ``Layer`` is alpha-blended on top in
strict ``z_index`` order (back to front). Blending happens in place on the
background frame -- the same frame then carries the debug skeleton, gesture
overlay, and HUD -- which preserves the pre-refactor behavior exactly (the old
``Overlay.composite`` mutated the mirrored webcam frame directly).

Pixel math (edge clipping + alpha blend) is lifted verbatim from the old
``overlay.py`` so image mode looks identical; it now operates on a layer's
``source`` image + ``Transform`` instead of a bespoke ``Overlay`` object.
"""

import cv2
import numpy as np

from src.scene import Scene, Transform, rect_px, scaled_size, top_left_px

# Thin box drawn around the selected layer AFTER compositing, so the active
# layer is obvious when several overlap. Orange (BGR) reads as a distinct
# "selection" accent against the red skeleton dots / green connections / yellow
# pinch dot; kept thin (2 px) so it frames without covering the layer.
_HIGHLIGHT_COLOR = (0, 165, 255)
_HIGHLIGHT_THICKNESS = 2


def _alpha_composite(frame: np.ndarray, image_bgra: np.ndarray, transform: Transform) -> None:
    """Alpha-blend a BGRA ``image_bgra`` onto ``frame`` in place at ``transform``.

    Scales the image to ``transform.scale``, clips it at all four frame edges
    (partially off-screen is fine; fully off-screen is a no-op), and blends by
    the per-pixel alpha. Mutates ``frame``.
    """
    frame_h, frame_w = frame.shape[:2]
    base_w, base_h = image_bgra.shape[1], image_bgra.shape[0]
    scaled_w, scaled_h = scaled_size((base_w, base_h), transform.scale)
    if scaled_w < 1 or scaled_h < 1:
        return

    interpolation = cv2.INTER_AREA if transform.scale < 1.0 else cv2.INTER_LINEAR
    scaled = cv2.resize(image_bgra, (scaled_w, scaled_h), interpolation=interpolation)

    x0, y0 = top_left_px(transform, (scaled_w, scaled_h), (frame_w, frame_h))

    # Intersection of the layer rect with the frame, in frame pixels.
    fx0, fy0 = max(0, x0), max(0, y0)
    fx1, fy1 = min(frame_w, x0 + scaled_w), min(frame_h, y0 + scaled_h)
    if fx0 >= fx1 or fy0 >= fy1:
        return  # fully off-screen

    # Matching region inside the scaled layer.
    ox0, oy0 = fx0 - x0, fy0 - y0
    ox1, oy1 = ox0 + (fx1 - fx0), oy0 + (fy1 - fy0)

    overlay_region = scaled[oy0:oy1, ox0:ox1]
    frame_region = frame[fy0:fy1, fx0:fx1]

    alpha = overlay_region[:, :, 3:4].astype(np.float32) / 255.0
    overlay_bgr = overlay_region[:, :, :3].astype(np.float32)
    blended = alpha * overlay_bgr + (1.0 - alpha) * frame_region.astype(np.float32)
    frame[fy0:fy1, fx0:fx1] = blended.astype(frame.dtype)


def render(background: np.ndarray, scene: Scene) -> np.ndarray:
    """Composite ``scene`` onto the webcam ``background`` in place, back-to-front.

    Layers are painted in ascending ``z_index`` order. A layer whose source has
    no frame yet (``read()`` returns ``None``) is skipped. An empty scene leaves
    the background untouched (skeleton mode). Returns the same ``background``
    array for convenient chaining.
    """
    for layer in scene.layers_by_z():
        image = layer.source.read()
        if image is None:
            continue
        _alpha_composite(background, image, layer.transform)
    return background


def draw_selection_highlight(frame: np.ndarray, scene: Scene) -> np.ndarray:
    """Outline the selected layer with a thin box, in place, AFTER compositing.

    Drawn after ``render`` so the frame is visible over the layer it marks.
    A no-op when nothing is selected or the selected layer has no rect yet
    (its source hasn't produced a frame, so ``base_size`` is ``None``). Returns
    the same ``frame`` for convenient chaining.
    """
    layer = scene.selected_layer
    if layer is None:
        return frame
    # Snapshot base_size ONCE: for a live layer the capture thread can null it
    # (window closed) between a check and a later re-read, and Layer.rect_px
    # re-reads it and raises. One read + the module-level rect_px is race-free.
    base_size = layer.base_size
    if base_size is None:
        return frame
    frame_h, frame_w = frame.shape[:2]
    x0, y0, w, h = rect_px(layer.transform, base_size, (frame_w, frame_h))
    cv2.rectangle(
        frame, (x0, y0), (x0 + w, y0 + h), _HIGHLIGHT_COLOR, _HIGHLIGHT_THICKNESS
    )
    return frame

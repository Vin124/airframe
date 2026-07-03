"""Image generator: turn a parsed ``generate`` request into a made-up image.

Phase 16 (OPTIONAL, GPU). The parser (``src.parser``) routes an *imagined* or
*described* thing ("a friendly robot", "a simple A shape") to
``route="generate"`` / ``source="generator"``. This module runs a fast few-step
text-to-image diffusion model on the GPU and hands back a ready ``GeneratedAsset``
that is DUCK-COMPATIBLE with ``retriever.RetrievedAsset`` (same ``.bgra`` /
``.source_name`` / ``.query`` contract), so it spawns through the EXACT SAME
Phase 15 path (``PreviewInbox`` -> dock -> pinch-pull) as a retrieved image.

DESIGN (the app's "optional dep, degrade never die" rule):
  - ALL heavy imports (``torch``, ``diffusers``) are DEFERRED inside the engine's
    methods, so the base pipeline (and this module's unit tests) import cleanly
    with NEITHER installed. Only ``--generate`` (opt-in) ever touches them.
  - GPU detection is ``torch.cuda.is_available()``. No CUDA GPU (or torch/diffusers
    absent) -> ONE clear stderr notice + the engine is marked unavailable, so
    every generate degrades to ``None`` (never a crash).
  - The diffusion pipeline is LAZY-loaded on the FIRST ``generate`` (not at
    construction), so merely enabling the flag for a session that never speaks a
    generate request doesn't hold ~7 GB of VRAM. The first generate is therefore
    SLOW (model download on a cold cache + load); warm generates are fast.
  - Targeted at an 8 GB card (RTX 4060): a turbo model at fp16/512px with
    ``enable_model_cpu_offload`` + VAE/attention slicing so it fits comfortably.
    The north star is "a described object reliably appears in frame", not maximum
    fidelity. ``sdxl-turbo`` is the default; ``sd-turbo`` (~2.5 GB) is the lighter
    fallback if the default OOMs even with offload.
  - ``make_placeholder`` renders a "materializing: <query>" card (pure cv2, NO
    GPU) that docks IMMEDIATELY while generation blocks; the finished image is
    published second and REPLACES it via the ``PreviewInbox``'s latest-wins slot.

THREADING: ``RequestGenerator.generate`` is called from the SPEECH thread's
``on_phrase`` callback (see ``src.launch.build_speech_on_phrase``), never the
render loop. It BLOCKS for seconds on the GPU; that is fine off the render thread
(same as the retriever/LLM) and can never stall render/tracking FPS. It catches
EVERYTHING (OOM/model/runtime) and NEVER raises to the speech thread.
"""

import sys
from dataclasses import dataclass
from typing import Optional, Protocol

import cv2
import numpy as np

# --------------------------------------------------------------------------- #
# Model + inference constants (target the 8 GB RTX 4060; reliability over max
# fidelity). The model id is a module constant so it's swappable in one place.
# --------------------------------------------------------------------------- #

# Default turbo model: SDXL-Turbo (fp16, 1-4 steps, 512 px) with memory savings
# enabled so it fits 8 GB comfortably.
DEFAULT_MODEL_ID = "stabilityai/sdxl-turbo"
# Lighter fallback (~2.5 GB fp16) if the default OOMs even with CPU offload.
FALLBACK_MODEL_ID = "stabilityai/sd-turbo"

# Turbo models are distilled for very few steps and NO classifier-free guidance.
NUM_INFERENCE_STEPS = 4
GUIDANCE_SCALE = 0.0
# Square output; the dock/scene scale it down anyway.
IMAGE_SIZE = 512

# Stamped onto every produced asset (matches the parser's SOURCE_GENERATOR).
SOURCE_NAME = "generator"

# The ONE clear message printed when no CUDA GPU / torch+diffusers is present.
GPU_UNAVAILABLE_MESSAGE = (
    "[generator] needs a CUDA GPU / torch+diffusers; generate requests will be "
    "ignored (pip install the Phase 16 block from requirements.txt)."
)

# --------------------------------------------------------------------------- #
# Placeholder card constants (pure cv2 -- the "materializing" preview).
# --------------------------------------------------------------------------- #

PLACEHOLDER_SIZE = (IMAGE_SIZE, IMAGE_SIZE)  # (width, height)
PLACEHOLDER_BG_COLOR = (60, 60, 60)          # BGR dark gray card
PLACEHOLDER_TEXT_COLOR = (240, 240, 240)     # BGR near-white label
PLACEHOLDER_PREFIX = "materializing: "
PLACEHOLDER_MAX_TEXT_CHARS = 48              # keep the label on one readable line
_PLACEHOLDER_FONT = cv2.FONT_HERSHEY_SIMPLEX
_PLACEHOLDER_TEXT_SCALE = 0.7
_PLACEHOLDER_TEXT_THICKNESS = 2
_PLACEHOLDER_TEXT_MARGIN = 8                 # min left inset if the label is wide


# --------------------------------------------------------------------------- #
# GeneratedAsset: the immutable, ready-to-spawn result (duck-compat RetrievedAsset)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class GeneratedAsset:
    """A generated image ready to become a scene ``Layer`` (like ``RetrievedAsset``).

    ``bgra`` is an ``HxWx4`` uint8 BGRA array (the overlay-layer contract);
    ``query`` is the prompt used; ``source_name`` is always ``"generator"``.
    ``is_placeholder`` distinguishes the transient "materializing" card from the
    finished image so the main loop / logs can tell them apart. Immutable; the
    ``.bgra`` / ``.source_name`` / ``.query`` trio is exactly what the Phase 15
    spawn path reads, so a generated asset spawns identically to a retrieved one.
    """

    bgra: np.ndarray
    query: str
    source_name: str = SOURCE_NAME
    is_placeholder: bool = False


# --------------------------------------------------------------------------- #
# Pure helpers (unit-testable, NO GPU/torch): placeholder card + RGB->BGRA
# --------------------------------------------------------------------------- #

def rgb_to_bgra(rgb: np.ndarray) -> np.ndarray:
    """Convert an ``HxWx3`` RGB uint8 array to an ``HxWx4`` BGRA (opaque alpha).

    The diffusion pipeline yields PIL/RGB; the overlay contract is BGRA. This
    swaps R<->B and appends a fully-opaque alpha channel via ``cv2.cvtColor``
    (contiguous uint8 output). Pure + testable with a synthetic array.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"rgb_to_bgra expects HxWx3, got shape {rgb.shape!r}")
    array = np.ascontiguousarray(rgb, dtype=np.uint8)
    return cv2.cvtColor(array, cv2.COLOR_RGB2BGRA)


def make_placeholder(query: str, size=PLACEHOLDER_SIZE) -> np.ndarray:
    """Render a gray "materializing: <query>" card as an ``HxWx4`` BGRA array.

    Shown IMMEDIATELY in the top dock while generation blocks (pure cv2, no GPU),
    then replaced by the finished image. The label is truncated to one line and
    centered; the solid card guarantees non-zero pixels even for an empty query.
    """
    width, height = size
    width = max(1, int(width))
    height = max(1, int(height))
    # Draw on a BGR card, then append a fully-opaque alpha: this keeps the whole
    # card opaque (the anti-aliased text would otherwise blend the alpha channel).
    card = np.empty((height, width, 3), dtype=np.uint8)
    card[:, :] = PLACEHOLDER_BG_COLOR

    text = f"{PLACEHOLDER_PREFIX}{query}".strip()[:PLACEHOLDER_MAX_TEXT_CHARS]
    (text_w, text_h), _baseline = cv2.getTextSize(
        text, _PLACEHOLDER_FONT, _PLACEHOLDER_TEXT_SCALE, _PLACEHOLDER_TEXT_THICKNESS
    )
    x = max(_PLACEHOLDER_TEXT_MARGIN, (width - text_w) // 2)
    y = (height + text_h) // 2
    cv2.putText(
        card, text, (x, y), _PLACEHOLDER_FONT, _PLACEHOLDER_TEXT_SCALE,
        PLACEHOLDER_TEXT_COLOR, _PLACEHOLDER_TEXT_THICKNESS, cv2.LINE_AA,
    )
    return cv2.cvtColor(card, cv2.COLOR_BGR2BGRA)  # opaque alpha


# --------------------------------------------------------------------------- #
# GeneratorEngine: the one interface a backend implements (prompt -> BGRA/None)
# --------------------------------------------------------------------------- #

class GeneratorEngine(Protocol):
    """A swappable text-to-image backend: ``generate(prompt)`` -> BGRA or ``None``.

    Returns an ``HxWx4`` uint8 BGRA array on success, or ``None`` on ANY failure
    (no GPU, model load error, OOM, ...) -- it PREFERS ``None`` over raising so the
    caller degrades gracefully.
    """

    def generate(self, prompt: str) -> Optional[np.ndarray]: ...


# --------------------------------------------------------------------------- #
# DiffusionGenerator: the real engine (all heavy imports DEFERRED)
# --------------------------------------------------------------------------- #

def _is_oom(exc: Exception) -> bool:
    """True iff ``exc`` looks like a CUDA out-of-memory error (name or message).

    Avoids importing torch just to reference ``torch.cuda.OutOfMemoryError``:
    matches the class name and the canonical message text instead.
    """
    return (
        type(exc).__name__ == "OutOfMemoryError"
        or "out of memory" in str(exc).lower()
    )


def _empty_cuda_cache() -> None:
    """Best-effort ``torch.cuda.empty_cache()`` so a fallback reload has room."""
    try:
        import torch  # deferred

        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 - cache cleanup is best-effort
        pass


class DiffusionGenerator:
    """Run a fast few-step diffusion model on the GPU; ``None`` if unavailable.

    ALL heavy imports (``torch``, ``diffusers``) are deferred: ``__init__`` only
    probes the GPU (importing ``torch`` to call ``cuda.is_available()``), and the
    pipeline itself is LAZY-loaded on the first ``generate`` so enabling the flag
    doesn't hold VRAM for a session that never generates. No GPU / no torch marks
    the engine unavailable (one stderr notice) and every ``generate`` returns
    ``None``. Never raises out of ``generate`` on the normal failure paths.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        steps: int = NUM_INFERENCE_STEPS,
        guidance: float = GUIDANCE_SCALE,
        image_size: int = IMAGE_SIZE,
    ) -> None:
        self._model_id = model_id
        self._steps = steps
        self._guidance = guidance
        self._image_size = image_size
        self._pipe = None  # lazy-loaded on first generate
        self._tried_models = set()  # models we've attempted to load (OOM fallback)
        self._available = self._detect_gpu()

    @property
    def available(self) -> bool:
        """True iff a CUDA GPU + torch were found (else every generate is None)."""
        return self._available

    def _detect_gpu(self) -> bool:
        """Probe for a usable CUDA GPU; print ONE notice + return False if absent."""
        try:
            import torch  # deferred: base app imports without it
        except ImportError:
            print(GPU_UNAVAILABLE_MESSAGE, file=sys.stderr)
            return False
        try:
            usable = bool(torch.cuda.is_available())
        except Exception:  # noqa: BLE001 - a broken torch build can't crash us
            usable = False
        if not usable:
            print(GPU_UNAVAILABLE_MESSAGE, file=sys.stderr)
            return False
        return True

    def generate(self, prompt: str) -> Optional[np.ndarray]:
        """Generate a BGRA image for ``prompt``; ``None`` if unavailable/failed.

        An out-of-memory error mid-inference triggers a ONE-TIME fallback to the
        lighter model (reload + retry) before giving up -- the 8 GB target may not
        hold SDXL-Turbo under other VRAM pressure.
        """
        if not self._available:
            return None
        rgb = self._run_inference(prompt)
        if rgb is None and self._advance_to_fallback():
            rgb = self._run_inference(prompt)  # retry once on the lighter model
        if rgb is None:
            return None
        return rgb_to_bgra(rgb[:, :, :3])

    def _run_inference(self, prompt: str) -> Optional[np.ndarray]:
        """One inference attempt on the current pipeline; ``None`` on OOM/failure.

        An OOM is swallowed here (empties the cache) so ``generate`` can retry on
        the fallback model; any other failure also returns ``None`` (the caller
        degrades). Non-OOM failures are announced once.
        """
        pipe = self._ensure_pipeline()
        if pipe is None:
            return None
        import torch  # deferred

        try:
            with torch.inference_mode():
                output = pipe(
                    prompt=prompt,
                    num_inference_steps=self._steps,
                    guidance_scale=self._guidance,
                    height=self._image_size,
                    width=self._image_size,
                )
        except Exception as exc:  # noqa: BLE001 - degrade; generate() may retry on OOM
            if _is_oom(exc):
                print(f"[generator] {self._model_id} OOM during inference.", file=sys.stderr)
            else:
                print(f"[generator] inference failed ({exc}).", file=sys.stderr)
            _empty_cuda_cache()
            return None
        images = getattr(output, "images", None)
        if images is None or len(images) == 0:
            return None
        rgb = np.asarray(images[0])
        if rgb.ndim != 3 or rgb.shape[2] < 3:
            return None
        return rgb[:, :, :3]

    def _advance_to_fallback(self) -> bool:
        """Switch to the lighter fallback model (drop the current pipeline) once.

        Returns True iff a not-yet-tried fallback exists, so ``generate`` should
        retry. A no-op (False) when already on the fallback or none is configured.
        """
        if self._model_id == FALLBACK_MODEL_ID or FALLBACK_MODEL_ID in self._tried_models:
            return False
        print(
            f"[generator] falling back from {self._model_id} to {FALLBACK_MODEL_ID}.",
            file=sys.stderr,
        )
        self._model_id = FALLBACK_MODEL_ID
        self._pipe = None  # force a reload of the lighter model on next inference
        _empty_cuda_cache()
        return True

    def _ensure_pipeline(self):
        """Return the diffusion pipeline, loading it on first use (slow, once).

        A load-time OOM on the primary model transparently falls back to the
        lighter one; any other load failure disables the engine (degrade, never
        crash).
        """
        if self._pipe is not None:
            return self._pipe
        self._tried_models.add(self._model_id)
        try:
            self._pipe = self._load_pipeline(self._model_id)
        except Exception as exc:  # noqa: BLE001 - load failure disables/falls back
            if _is_oom(exc) and self._advance_to_fallback():
                return self._ensure_pipeline()  # retry once with the lighter model
            print(f"[generator] model load failed ({exc}); generate disabled.", file=sys.stderr)
            self._available = False
            return None
        return self._pipe

    def _load_pipeline(self, model_id: str):
        """Load the fp16 turbo pipeline (``model_id``) with 8 GB memory savings on.

        Loads the ``fp16`` weight variant explicitly -- both target turbo models
        (sdxl-turbo / sd-turbo) ship it, roughly HALVING the download + VRAM vs the
        fp32 weights. There is deliberately NO silent no-variant fallback: a model
        without an fp16 variant (or a disk/network failure) surfaces to the caller,
        which degrades to ``None`` rather than pulling the far larger fp32 weights
        (which would double the download and can exhaust disk on a cold cache).
        """
        import torch  # deferred
        from diffusers import AutoPipelineForText2Image  # deferred

        pipe = AutoPipelineForText2Image.from_pretrained(
            model_id, torch_dtype=torch.float16, variant="fp16"
        )
        self._apply_memory_savings(pipe)
        return pipe

    @staticmethod
    def _apply_memory_savings(pipe) -> None:
        """Enable CPU offload + VAE/attention slicing so the model fits 8 GB.

        Each optimization is best-effort: an unsupported method on a given
        pipeline is skipped with a notice rather than failing the load.
        """
        for method in (
            "enable_model_cpu_offload",
            "enable_vae_slicing",
            "enable_attention_slicing",
        ):
            fn = getattr(pipe, method, None)
            if fn is None:
                continue
            try:
                fn()
            except Exception as exc:  # noqa: BLE001 - optional optimization
                print(f"[generator] {method} unavailable ({exc}).", file=sys.stderr)


# --------------------------------------------------------------------------- #
# RequestGenerator: engine + safe wrapping into a GeneratedAsset (public entry)
# --------------------------------------------------------------------------- #

@dataclass
class RequestGenerator:
    """Tie a ``GeneratorEngine`` to safe ``GeneratedAsset`` production.

    ``generate(query)`` returns a ``GeneratedAsset`` on success, or ``None`` for a
    blank query OR any failure (an engine returning ``None``, or an unexpected
    engine exception -- OOM/model/runtime). It catches EVERYTHING and NEVER raises,
    printing at most one stderr notice on an unexpected failure: the speech thread
    that calls it must keep running.
    """

    engine: GeneratorEngine

    def generate(self, query: str) -> Optional[GeneratedAsset]:
        query = (query or "").strip()
        if not query:
            return None
        try:
            bgra = self.engine.generate(query)
        except Exception as exc:  # noqa: BLE001 - a bad engine can't crash the app
            print(f"Warning: generator failed ({exc}); no image.", file=sys.stderr)
            return None
        if bgra is None:
            return None
        return GeneratedAsset(bgra=bgra, query=query, source_name=SOURCE_NAME)


def build_request_generator(model_id: str = DEFAULT_MODEL_ID) -> RequestGenerator:
    """Assemble a production ``RequestGenerator`` backed by a ``DiffusionGenerator``.

    Construction probes the GPU (one notice if absent) but does NOT load the model
    -- the pipeline lazy-loads on the first ``generate``.
    """
    return RequestGenerator(DiffusionGenerator(model_id))

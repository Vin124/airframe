"""Launch-time settings glue: panel prefill, config <-> args, and panel build.

These are the seams between the CLI/config layer and the runtime: resolving the
control panel's initial field values (CLI > config > default), turning launch
args into a persistable config dict, deciding what to persist on exit, and
building the Tk panel (imported lazily so a headless import never pulls in
Tkinter). Kept out of ``src.main`` so the entry point stays a thin
setup/teardown shell; ``src.main`` re-exports these names for the test surface.
"""

import argparse
import sys
from typing import TYPE_CHECKING, Any, Dict, Optional

from src.layers import (
    DEFAULT_MONITOR_INDEX,
    LIVE_SOURCES,
    MODE_IMAGE,
    MODES,
    SOURCE_IMAGE,
    SOURCE_WINDOW,
    SOURCES,
)
from src.tuning import (
    DEFAULT_MEDIAPIPE_CONFIDENCE,
    DEFAULT_PINCH_ENTER,
    DEFAULT_PINCH_EXIT,
    GestureTuning,
    TrackerConfidence,
    clamp_tuning,
)

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids Tk import at runtime
    from src.main import PipelineRuntime
    from src.panel import ControlPanel
    from src.spawn import PreviewInbox
    from src.speech import PhraseCallback


def resolve_panel_prefill(args: argparse.Namespace, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve the panel's initial field values with precedence CLI > config > default.

    argparse can't tell an explicitly-passed default from the built-in default,
    so "CLI wins" is applied as "a non-default CLI value wins"; when a CLI option
    is left at its built-in default, the remembered ``cfg`` value is used, which
    itself already defaults (see ``src.config``). Known limitation (accepted for
    this debug harness): a CLI value equal to the built-in default can't override
    a persisted config value. ``--virtualcam`` is store_true, so it can only
    force the cam ON; to keep it OFF you simply don't pass it and config
    remembers the last state.
    """
    source = args.source if args.source != SOURCE_IMAGE else cfg.get("source", SOURCE_IMAGE)
    monitor_index = (
        args.monitor_index
        if args.monitor_index != DEFAULT_MONITOR_INDEX
        else cfg.get("monitor_index", DEFAULT_MONITOR_INDEX)
    )
    return {
        "source": source,
        "image_path": args.image or cfg.get("image_path", ""),
        "window_title": args.window_title or cfg.get("window_title", ""),
        "monitor_index": monitor_index,
        "virtualcam": bool(args.virtualcam or cfg.get("virtualcam", False)),
        # Speech opt-in: store_true CLI can only force ON, else remembered config
        # value (same precedence rule as --virtualcam).
        "speech": bool(args.speech or cfg.get("speech", False)),
        # Parser opt-in (Phase 13): same store_true precedence as --speech. Only
        # acted on when speech is also on (see build_speech_on_phrase).
        "parse": bool(args.parse or cfg.get("parse", False)),
        # Generator opt-in (Phase 16, GPU-only): same store_true precedence. Only
        # acted on when speech + parse are also on (see build_speech_on_phrase).
        "generate": bool(args.generate or cfg.get("generate", False)),
        # Pinch tuning has no CLI flag (it's the live sliders): config > default.
        # cfg already validated/clamped these through merge_config.
        "pinch_enter": cfg.get("pinch_enter", DEFAULT_PINCH_ENTER),
        "pinch_exit": cfg.get("pinch_exit", DEFAULT_PINCH_EXIT),
        # Startup-only confidences: non-default CLI wins, else config, else
        # default -- the same "non-default CLI value wins" rule as source/monitor.
        "min_detection_confidence": _confidence_prefill(
            args.min_detection_confidence, cfg, "min_detection_confidence"
        ),
        "min_presence_confidence": _confidence_prefill(
            args.min_presence_confidence, cfg, "min_presence_confidence"
        ),
        "min_tracking_confidence": _confidence_prefill(
            args.min_tracking_confidence, cfg, "min_tracking_confidence"
        ),
    }


def _confidence_prefill(cli_value: float, cfg: Dict[str, Any], key: str) -> float:
    """Resolve one confidence knob: a non-default CLI value wins, else config/default.

    argparse can't distinguish an explicitly-passed default from the built-in
    one, so (as with ``--monitor-index``) a CLI value left at MediaPipe's default
    yields to the remembered config value.
    """
    if cli_value != DEFAULT_MEDIAPIPE_CONFIDENCE:
        return cli_value
    return cfg.get(key, DEFAULT_MEDIAPIPE_CONFIDENCE)


def _config_from_args(args: argparse.Namespace, virtualcam_active: bool) -> Dict[str, Any]:
    """Build a config dict from the launch args (the ``--no-panel`` persist path)."""
    return {
        "source": args.source,
        "image_path": args.image or "",
        "window_title": args.window_title or "",
        "monitor_index": args.monitor_index,
        "virtualcam": bool(virtualcam_active),
    }


def tuning_from_prefill(prefill: Dict[str, Any]) -> GestureTuning:
    """Build the startup ``GestureTuning`` from a resolved prefill, clamped.

    Routed through ``clamp_tuning`` -- the single hysteresis validation
    boundary -- so no code path can construct an un-clamped startup tuning.
    """
    return clamp_tuning(prefill["pinch_enter"], prefill["pinch_exit"])


def confidence_from_prefill(prefill: Dict[str, Any]) -> TrackerConfidence:
    """Build the startup ``TrackerConfidence`` from a resolved prefill."""
    return TrackerConfidence(
        detection=prefill["min_detection_confidence"],
        presence=prefill["min_presence_confidence"],
        tracking=prefill["min_tracking_confidence"],
    )


def _final_settings(
    panel,
    args: argparse.Namespace,
    vcam_state: bool,
    tuning: GestureTuning,
    confidence: TrackerConfidence,
    speech: bool = False,
    parse: bool = False,
    generate: bool = False,
) -> Dict[str, Any]:
    """The settings dict persisted on exit.

    Prefers the panel's live field values, but the RUNTIME's virtual-cam state
    (``vcam_state``) is authoritative: the panel's checkbox freezes once its
    window is closed, so a 'v' hotkey flip after that would otherwise persist
    stale. Falls back to args-derived settings when no panel was built. The pinch
    ``tuning`` (runtime-authoritative, driven by the sliders), the startup
    ``confidence`` (never changes at runtime), and the resolved ``speech`` /
    ``parse`` / ``generate`` opt-ins are overlaid last so all round-trip -- the
    panel has no widget for any of them.
    """
    settings = _config_from_args(args, vcam_state) if panel is None else panel.current_settings()
    settings["virtualcam"] = bool(vcam_state)
    settings["speech"] = bool(speech)
    settings["parse"] = bool(parse)
    settings["generate"] = bool(generate)
    settings["pinch_enter"] = tuning.enter
    settings["pinch_exit"] = tuning.exit
    settings["min_detection_confidence"] = confidence.detection
    settings["min_presence_confidence"] = confidence.presence
    settings["min_tracking_confidence"] = confidence.tracking
    return settings


def build_speech_on_phrase(
    parse_enabled: bool,
    inbox: Optional["PreviewInbox"] = None,
    generate_enabled: bool = False,
) -> Optional["PhraseCallback"]:
    """Compose the speech ``on_phrase`` callback (the Phase 13/14/15/16 wiring seam).

    Returns ``None`` when parsing is off, so ``speech_session`` keeps its default
    callback (which prints the live ``[speech]`` transcript). When on, returns a
    callback that prints that live transcript, runs the LLM parser (logging
    ``[parser] {...}`` for actionable phrases only), then routes each actionable
    request:
      - ``retrieve`` (Phase 14): fetch the image and log ``[retriever] got <WxH>
        image ...`` (or ``[retriever] no image ...``).
      - ``generate`` (Phase 16): only when ``generate_enabled`` -- dock a
        "materializing" placeholder at once, run the GPU generator, then publish
        the finished image (which REPLACES the placeholder). Logs ``[generator]
        generated <WxH> for '<query>'`` / ``[generator] failed ...``. With
        ``--generate`` OFF a generate request logs ``[generator] disabled (pass
        --generate) ...`` and is ignored (retrieval stays the default).

    Phase 15: when an ``inbox`` (``PreviewInbox``) is supplied, a produced asset is
    ALSO published to it (in addition to the log line) so the main loop can dock it
    as a spawnable preview. With ``inbox=None`` this behaves exactly as before
    (log-only), so the earlier-phase unit tests are unaffected.

    THREADING: this callback runs on the SPEECH thread (``SpeechWorker`` invokes
    ``on_phrase`` there, never the render loop). ``RequestParser.parse``,
    ``Retriever.retrieve`` and ``RequestGenerator.generate`` all BLOCK (network /
    GPU) for up to their bounded budgets, which is fine off the render thread and
    can't stall render/tracking FPS; the worker's bounded ``stop()`` join keeps a
    slow call from wedging shutdown. Each ``inbox.publish`` is a single atomic
    reference store (no lock) -- the speech thread never touches scene/interaction,
    only drops the asset in the slot. The parser/retriever/generator/speech imports
    are deferred so a headless import of this module (and its unit tests) doesn't
    pull them (or torch/diffusers) in.
    """
    if not parse_enabled:
        return None
    from src.parser import ROUTE_GENERATE, build_request_parser
    from src.retriever import build_retriever
    from src.speech import default_on_phrase

    parser = build_request_parser()
    retriever = build_retriever()
    generator = _build_request_generator() if generate_enabled else None

    def on_phrase(text: str) -> None:
        default_on_phrase(text)  # preserve the Phase 12 live-transcript line
        request = parser.parse(text)
        if request is None or not request.is_actionable:
            return
        print(f"[parser] {request}")
        if request.route == ROUTE_GENERATE:
            _run_generate(request, generator, inbox)
        else:
            _run_retrieve(request, retriever, inbox)

    return on_phrase


def _build_request_generator():
    """Compose the Phase 16 ``RequestGenerator`` (deferred import; probes the GPU)."""
    from src.generator import build_request_generator

    return build_request_generator()


def _run_retrieve(request, retriever, inbox: Optional["PreviewInbox"]) -> None:
    """Fetch + log a ``retrieve`` request and publish any asset (SPEECH thread)."""
    asset = retriever.retrieve(request)
    if asset is None:
        print(f"[retriever] no image for '{request.query}'")
        return
    height, width = asset.bgra.shape[:2]
    print(
        f"[retriever] got {width}x{height} image for "
        f"'{request.query}' from {asset.source_name}"
    )
    if inbox is not None:
        inbox.publish(asset)  # Phase 15: hand it to the main loop to dock


def _run_generate(request, generator, inbox: Optional["PreviewInbox"]) -> None:
    """Generate + log a ``generate`` request (SPEECH thread); placeholder then result.

    With no generator (``--generate`` off) the request is ignored with a notice.
    Otherwise a "materializing" placeholder docks IMMEDIATELY, generation blocks
    on the GPU, and on success the finished image is published -- REPLACING the
    placeholder via the inbox's latest-wins slot. On failure the placeholder is
    left to auto-dismiss (Phase 15's timer). Never raises to the speech thread.
    """
    query = request.query
    if generator is None:
        print(f"[generator] disabled (pass --generate) for '{query}'")
        return
    if inbox is not None:
        from src.generator import GeneratedAsset, make_placeholder

        inbox.publish(
            GeneratedAsset(bgra=make_placeholder(query), query=query, is_placeholder=True)
        )
    asset = generator.generate(query)
    if asset is None:
        print(f"[generator] failed for '{query}'")
        return
    height, width = asset.bgra.shape[:2]
    print(f"[generator] generated {width}x{height} for '{query}'")
    if inbox is not None:
        inbox.publish(asset)  # Phase 15: replaces the placeholder (latest-wins)


def _build_panel(runtime: "PipelineRuntime", prefill: Dict[str, Any]) -> Optional["ControlPanel"]:
    """Create the Tk control panel bound to ``runtime``; None if Tk is unavailable.

    Imported lazily so a headless/pure import of this module (and its unit tests)
    never pulls in Tkinter. A ``PanelUnavailable`` (no display / Tk not built)
    prints a notice and returns None so the app still runs with hotkeys only.
    """
    from src.panel import ControlPanel, PanelActions, PanelUnavailable

    try:
        return ControlPanel(PanelActions.from_runtime(runtime), prefill)
    except PanelUnavailable as exc:
        print(
            f"Warning: control panel unavailable ({exc}); running with hotkeys only.",
            file=sys.stderr,
        )
        return None


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
        "--speech",
        action="store_true",
        help=(
            "Enable the always-listening speech layer: capture mic audio on a "
            "background thread and print a live transcript to the console "
            "(Moonshine, falling back to faster-whisper). Off by default; the "
            "last state is remembered in the config."
        ),
    )
    parser.add_argument(
        "--parse",
        action="store_true",
        help=(
            "Enable the Phase 13 LLM parser: turn each finalized speech phrase "
            "into a structured request (action/query/route/source) via a local "
            "Ollama model and print it to the console. Only meaningful with "
            "--speech; needs Ollama running + a small model pulled (qwen2.5:1.5b). "
            "Off by default; the last state is remembered in the config."
        ),
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help=(
            "Enable the Phase 16 image GENERATOR: a parsed 'generate'-routed "
            "request (an imagined/described thing) runs a fast few-step diffusion "
            "model on the GPU and spawns the result like a retrieved image. Only "
            "meaningful with --speech --parse; needs a CUDA GPU + the optional "
            "torch/diffusers block (see requirements.txt). Off by default; the "
            "last state is remembered in the config. Retrieval stays the default "
            "for real, nameable things."
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
    parser.add_argument(
        "--no-panel",
        dest="no_panel",
        action="store_true",
        help=(
            "Run without the Tkinter control panel (hotkeys only). The panel is "
            "on by default; use this for headless smokes or a keyboard-only run."
        ),
    )
    # Startup-only MediaPipe min-confidence knobs. These bake into the landmarker
    # (a change needs a tracker re-init), so they are CLI/config-resolved, NOT
    # live sliders -- unlike the pinch thresholds. Default = MediaPipe's own 0.5.
    parser.add_argument(
        "--min-detection-confidence",
        type=float,
        default=DEFAULT_MEDIAPIPE_CONFIDENCE,
        help=(
            "MediaPipe minimum hand-DETECTION confidence (0-1, default 0.5). "
            "Startup-only (needs a tracker re-init), so it's a flag, not a slider."
        ),
    )
    parser.add_argument(
        "--min-presence-confidence",
        type=float,
        default=DEFAULT_MEDIAPIPE_CONFIDENCE,
        help=(
            "MediaPipe minimum hand-PRESENCE confidence (0-1, default 0.5). "
            "Startup-only (needs a tracker re-init), so it's a flag, not a slider."
        ),
    )
    parser.add_argument(
        "--min-tracking-confidence",
        type=float,
        default=DEFAULT_MEDIAPIPE_CONFIDENCE,
        help=(
            "MediaPipe minimum hand-TRACKING confidence (0-1, default 0.5). "
            "Startup-only (needs a tracker re-init), so it's a flag, not a slider."
        ),
    )
    args = parser.parse_args(argv)

    if args.source in LIVE_SOURCES and args.mode != MODE_IMAGE:
        parser.error("--source window/screen is only valid with --mode image")
    if args.mode == MODE_IMAGE and args.source == SOURCE_IMAGE and not args.image:
        parser.error("--image is required when --mode is 'image' and --source is 'image'")
    if args.source == SOURCE_WINDOW and not args.window_title:
        parser.error("--window-title is required when --source is 'window'")
    for name, value in (
        ("--min-detection-confidence", args.min_detection_confidence),
        ("--min-presence-confidence", args.min_presence_confidence),
        ("--min-tracking-confidence", args.min_tracking_confidence),
    ):
        if not 0.0 <= value <= 1.0:
            parser.error(f"{name} must be between 0 and 1")

    return args

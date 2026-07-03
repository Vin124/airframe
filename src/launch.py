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

from src.layers import DEFAULT_MONITOR_INDEX, SOURCE_IMAGE

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids Tk import at runtime
    from src.main import PipelineRuntime
    from src.panel import ControlPanel


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
    }


def _config_from_args(args: argparse.Namespace, virtualcam_active: bool) -> Dict[str, Any]:
    """Build a config dict from the launch args (the ``--no-panel`` persist path)."""
    return {
        "source": args.source,
        "image_path": args.image or "",
        "window_title": args.window_title or "",
        "monitor_index": args.monitor_index,
        "virtualcam": bool(virtualcam_active),
    }


def _final_settings(panel, args: argparse.Namespace, vcam_state: bool) -> Dict[str, Any]:
    """The settings dict persisted on exit.

    Prefers the panel's live field values, but the RUNTIME's virtual-cam state
    (``vcam_state``) is authoritative: the panel's checkbox freezes once its
    window is closed, so a 'v' hotkey flip after that would otherwise persist
    stale. Falls back to args-derived settings when no panel was built.
    """
    if panel is None:
        return _config_from_args(args, vcam_state)
    settings = panel.current_settings()
    settings["virtualcam"] = bool(vcam_state)
    return settings


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

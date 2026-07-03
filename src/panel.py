"""Lightweight Tkinter control panel that runs alongside the preview window.

Phase 10 stops the app being CLI-and-hotkey-only: a small always-available panel
lets you pick a source and add it as a layer, toggle the virtual cam, reset the
scene, and watch a live FPS readout.

Threading / integration (important):
    Tkinter is **not** thread-safe, so the panel does NOT run on its own thread.
    ``ControlPanel`` creates its ``Tk`` root on the MAIN thread, and the pipeline
    loop pumps it once per tick via ``pump()`` (a non-blocking ``root.update()``).
    Because of that, a button callback fires *inline on the main thread*,
    interleaved between frames -- which means it may call the very same
    ``PipelineRuntime`` methods the hotkeys already call, with no cross-thread
    queue and no duplicated scene/interaction logic. The runtime stays the single
    writer of scene/interaction (see ``src.main``).

Structure:
    - ``PanelActions`` -- a tiny controller: a bundle of callables wired to a
      runtime. Pure Python, no Tkinter, so the wiring is unit-testable without a
      display (``PanelActions.from_runtime`` + a fake runtime).
    - ``format_fps`` -- pure formatter for the FPS label.
    - ``ControlPanel`` -- the thin Tk view. It only binds widgets to a
      ``PanelActions`` and refreshes read-only state (FPS, vcam) each ``pump()``.
      Constructing it needs a display; nothing else here does.

Closing the panel window (the OS "X") marks the panel closed and lets the app
keep running with hotkeys only -- ``pump()`` then becomes a no-op. A ``TclError``
after the root is destroyed is swallowed for the same reason.
"""

import sys
import tkinter as tk
from dataclasses import dataclass
from tkinter import filedialog, ttk
from typing import Any, Callable, Dict, Optional

from src import hud

# Source-choice values, mirroring src.main / src.config (same literals).
SOURCE_IMAGE = "image"
SOURCE_WINDOW = "window"
SOURCE_SCREEN = "screen"
_SOURCE_CHOICES = (SOURCE_IMAGE, SOURCE_WINDOW, SOURCE_SCREEN)

_NO_FPS_TEXT = "--"


class PanelUnavailable(RuntimeError):
    """Raised when a Tk root can't be created (no display / Tk not built).

    Lets the launcher fall back to a headless, hotkey-only run instead of
    crashing.
    """


def format_fps(fps: Optional[float]) -> str:
    """Format the runtime's last FPS for the readout label.

    Pure (no Tk) so it's unit-testable. ``None`` (before the first one-second
    window elapses) shows as a dash, matching the HUD.
    """
    if fps is None:
        return _NO_FPS_TEXT
    return f"{fps:.1f}"


@dataclass(frozen=True)
class PanelActions:
    """The controller: callables the view binds to, wired to a runtime.

    Kept free of Tkinter so it can be built and exercised in tests with a fake
    runtime. The action callables mutate scene/interaction via the runtime (the
    single writer); the ``get_*`` callables read live state for the view to
    display.
    """

    add_image_layer: Callable[[str], None]
    add_window_layer: Callable[[str], None]
    add_screen_layer: Callable[[int], None]
    toggle_virtualcam: Callable[[], None]
    reset_scene: Callable[[], None]
    get_fps: Callable[[], Optional[float]]
    get_virtualcam_active: Callable[[], bool]
    get_status: Callable[[], Optional[str]]

    @classmethod
    def from_runtime(cls, runtime: Any) -> "PanelActions":
        """Bind the panel actions to a ``PipelineRuntime`` (or a fake in tests)."""
        return cls(
            add_image_layer=runtime.add_image_layer,
            add_window_layer=runtime.add_window_layer,
            add_screen_layer=runtime.add_screen_layer,
            toggle_virtualcam=runtime.toggle_virtualcam,
            reset_scene=runtime.reset_scene,
            get_fps=lambda: runtime.last_fps,
            get_virtualcam_active=lambda: runtime.virtualcam_active,
            get_status=lambda: runtime.panel_status,
        )


class ControlPanel:
    """Thin Tk view: source picker + add, vcam toggle, reset, FPS readout.

    Bind widgets to a ``PanelActions``; refresh read-only state each ``pump()``.
    All three source-input fields (image path + Browse, window title, monitor
    index) are always visible; ``Add layer`` reads whichever matches the selected
    source. Pre-filled from ``prefill`` (resolved CLI > config > defaults upstream).
    """

    def __init__(self, actions: PanelActions, prefill: Optional[Dict[str, Any]] = None) -> None:
        self._actions = actions
        prefill = prefill or {}
        # Fallback snapshot for current_settings() if the Tk vars become
        # unreadable after the root is destroyed (platform-dependent).
        self._prefill = dict(prefill)
        self._closed = False
        try:
            self._root = tk.Tk()
        except tk.TclError as exc:  # pragma: no cover - needs a headless box
            raise PanelUnavailable(str(exc)) from exc

        self._root.title("handy control")
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._source_var = tk.StringVar(value=prefill.get("source", SOURCE_IMAGE))
        self._path_var = tk.StringVar(value=prefill.get("image_path", ""))
        self._title_var = tk.StringVar(value=prefill.get("window_title", ""))
        self._monitor_var = tk.StringVar(value=str(prefill.get("monitor_index", 1)))
        self._vcam_var = tk.BooleanVar(value=bool(prefill.get("virtualcam", False)))
        self._fps_var = tk.StringVar(value=_NO_FPS_TEXT)
        self._status_var = tk.StringVar(value="")

        self._build_widgets()

    # -- construction ----------------------------------------------------- #

    def _build_widgets(self) -> None:
        """Lay out the panel. All rows are simple ttk widgets in a padded frame."""
        frame = ttk.Frame(self._root, padding=10)
        frame.grid(sticky="nsew")

        ttk.Label(frame, text="Source:").grid(row=0, column=0, sticky="w")
        ttk.Combobox(
            frame,
            textvariable=self._source_var,
            values=list(_SOURCE_CHOICES),
            state="readonly",
            width=12,
        ).grid(row=0, column=1, sticky="w", pady=2)

        ttk.Label(frame, text="Image:").grid(row=1, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self._path_var, width=28).grid(
            row=1, column=1, sticky="we", pady=2
        )
        ttk.Button(frame, text="Browse...", command=self._on_browse).grid(
            row=1, column=2, padx=4
        )

        ttk.Label(frame, text="Window title:").grid(row=2, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self._title_var, width=28).grid(
            row=2, column=1, sticky="we", pady=2
        )

        ttk.Label(frame, text="Monitor:").grid(row=3, column=0, sticky="w")
        ttk.Spinbox(frame, from_=1, to=16, textvariable=self._monitor_var, width=6).grid(
            row=3, column=1, sticky="w", pady=2
        )

        ttk.Button(frame, text="Add layer", command=self._on_add_layer).grid(
            row=4, column=1, sticky="w", pady=(6, 2)
        )

        ttk.Checkbutton(
            frame,
            text="Virtual camera",
            variable=self._vcam_var,
            command=self._on_toggle_virtualcam,
        ).grid(row=5, column=0, columnspan=2, sticky="w", pady=2)

        ttk.Button(frame, text="Reset scene", command=self._on_reset_scene).grid(
            row=6, column=0, columnspan=2, sticky="w", pady=2
        )

        ttk.Separator(frame, orient="horizontal").grid(
            row=7, column=0, columnspan=3, sticky="we", pady=6
        )

        ttk.Label(frame, textvariable=self._fps_var).grid(
            row=8, column=0, columnspan=2, sticky="w"
        )
        ttk.Label(frame, textvariable=self._status_var, foreground="#888").grid(
            row=9, column=0, columnspan=3, sticky="w"
        )

        # Hotkey hint -- reuse the HUD's legend so the two never drift apart.
        for i, line in enumerate(hud.KEY_LEGEND_LINES):
            ttk.Label(frame, text=line, foreground="#888").grid(
                row=10 + i, column=0, columnspan=3, sticky="w"
            )

    # -- widget callbacks (run inline on the main thread via pump()) ------- #

    def _on_browse(self) -> None:
        # NOTE: modal dialog = nested Tk event loop on the main thread, so the
        # preview/virtual-cam output pauses on the last frame while it's open
        # (capture/tracking threads keep running). Not a hang.
        path = filedialog.askopenfilename(
            title="Choose overlay image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp"), ("All files", "*.*")],
        )
        if path:
            self._path_var.set(path)

    def _on_add_layer(self) -> None:
        source = self._source_var.get()
        if source == SOURCE_WINDOW:
            self._actions.add_window_layer(self._title_var.get())
        elif source == SOURCE_SCREEN:
            self._actions.add_screen_layer(self._monitor_index())
        else:
            self._actions.add_image_layer(self._path_var.get())
        self._refresh()

    def _on_toggle_virtualcam(self) -> None:
        self._actions.toggle_virtualcam()
        self._refresh()

    def _on_reset_scene(self) -> None:
        self._actions.reset_scene()
        self._refresh()

    def _on_close(self) -> None:
        """OS window-close: mark closed, keep the app running (hotkeys still work)."""
        self._closed = True
        try:
            self._root.destroy()
        except tk.TclError:  # pragma: no cover - already gone
            pass

    # -- loop integration -------------------------------------------------- #

    def pump(self) -> None:
        """Process pending Tk events + refresh readouts. Called once per tick.

        Non-blocking (``update`` not ``mainloop``). Becomes a no-op after the
        panel is closed; a ``TclError`` (root destroyed mid-update) flips it to
        closed rather than propagating, so a closed panel can never crash the loop.
        """
        if self._closed:
            return
        try:
            self._refresh()
            self._root.update()
        except tk.TclError:
            self._closed = True

    def _refresh(self) -> None:
        """Sync the read-only widgets (FPS, vcam checkbox, status) to live state.

        The vcam checkbox is polled here so a 'v' hotkey flip stays reflected in
        the panel without any event plumbing.
        """
        self._fps_var.set(format_fps(self._actions.get_fps()))
        active = bool(self._actions.get_virtualcam_active())
        if active != self._vcam_var.get():
            self._vcam_var.set(active)
        status = self._actions.get_status()
        if status:
            self._status_var.set(status)

    def current_settings(self) -> Dict[str, Any]:
        """Snapshot the panel fields for persistence (see ``src.config``).

        Reading Tk variables after the root was destroyed returns cached values
        on some builds but raises ``TclError`` on others; the guard makes the
        teardown persistence platform-proof (and keeps the caller's remaining
        finally-block cleanup running). The fallback is the prefill snapshot.
        """
        try:
            return {
                "source": self._source_var.get(),
                "image_path": self._path_var.get(),
                "window_title": self._title_var.get(),
                "monitor_index": self._monitor_index(),
                "virtualcam": bool(self._vcam_var.get()),
            }
        except tk.TclError:  # pragma: no cover - platform-dependent teardown
            return dict(self._prefill)

    def close(self) -> None:
        """Tear down the Tk root (idempotent; safe after the window was closed)."""
        self._closed = True
        try:
            self._root.destroy()
        except tk.TclError:  # pragma: no cover - already gone
            pass

    # -- helpers ----------------------------------------------------------- #

    def _monitor_index(self) -> int:
        """Parse the monitor spinbox to a >=1 int, defaulting to 1 on garbage."""
        try:
            value = int(self._monitor_var.get())
        except (TypeError, ValueError):
            return 1
        return value if value >= 1 else 1

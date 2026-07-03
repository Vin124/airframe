"""Unit tests for src.panel's controller layer (no Tkinter root created).

Only the display-free parts are exercised here: ``format_fps`` and the
``PanelActions`` wiring against a fake runtime. Constructing ``ControlPanel``
needs a display, so it is deliberately never instantiated -- the Tk view is a
thin binding over the actions verified below.
"""

from src.panel import PanelActions, format_fps
from src.tuning import GestureTuning, clamp_tuning


def test_format_fps_none_is_dash():
    assert format_fps(None) == "--"


def test_format_fps_rounds_to_one_decimal():
    assert format_fps(29.97) == "30.0"
    assert format_fps(30.0) == "30.0"


class _FakeRuntime:
    """Records panel-action calls and exposes read-only state like PipelineRuntime."""

    def __init__(self):
        self.calls = []
        self.last_fps = 42.0
        self.virtualcam_active = False
        self.panel_status = "ready"

    def add_image_layer(self, path):
        self.calls.append(("image", path))

    def add_window_layer(self, title):
        self.calls.append(("window", title))

    def add_screen_layer(self, index):
        self.calls.append(("screen", index))

    def toggle_virtualcam(self):
        self.calls.append(("toggle",))

    def reset_scene(self):
        self.calls.append(("reset",))

    def set_tuning(self, enter, exit):
        self.calls.append(("tuning", enter, exit))
        return clamp_tuning(enter, exit)


def test_from_runtime_wires_action_callables():
    rt = _FakeRuntime()
    actions = PanelActions.from_runtime(rt)

    actions.add_image_layer("a.png")
    actions.add_window_layer("Notepad")
    actions.add_screen_layer(2)
    actions.toggle_virtualcam()
    actions.reset_scene()

    assert rt.calls == [
        ("image", "a.png"),
        ("window", "Notepad"),
        ("screen", 2),
        ("toggle",),
        ("reset",),
    ]


def test_from_runtime_wires_set_tuning_and_returns_clamp():
    rt = _FakeRuntime()
    actions = PanelActions.from_runtime(rt)

    result = actions.set_tuning(0.06, 0.04)  # inverted -> clamp lifts exit
    assert rt.calls == [("tuning", 0.06, 0.04)]
    assert isinstance(result, GestureTuning)
    assert result.exit > result.enter  # hysteresis gap enforced by the runtime


def test_from_runtime_getters_read_live_state():
    rt = _FakeRuntime()
    actions = PanelActions.from_runtime(rt)

    assert actions.get_fps() == 42.0
    assert actions.get_virtualcam_active() is False
    assert actions.get_status() == "ready"

    # Getters are live: they re-read the runtime each call.
    rt.last_fps = 12.0
    rt.virtualcam_active = True
    rt.panel_status = "Added window layer."
    assert actions.get_fps() == 12.0
    assert actions.get_virtualcam_active() is True
    assert actions.get_status() == "Added window layer."

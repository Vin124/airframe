"""Unit tests for src.config: load/save/merge of the panel settings JSON.

Pure functions over ``tmp_path`` -- no UI, no display, no real config file in
the repo. Covers the round-trip, missing/corrupt fallbacks, and the
``merge_config`` validation boundary (unknown keys, wrong types, precedence).
"""

import json

from src import config
from src.tuning import DEFAULT_PINCH_ENTER, DEFAULT_PINCH_EXIT


def test_default_config_has_expected_keys_and_values():
    cfg = config.default_config()
    assert cfg == {
        "source": "image",
        "image_path": "",
        "window_title": "",
        "monitor_index": 1,
        "virtualcam": False,
        "speech": False,
        "parse": False,
        "generate": False,
        "pinch_enter": DEFAULT_PINCH_ENTER,
        "pinch_exit": DEFAULT_PINCH_EXIT,
        "min_detection_confidence": 0.5,
        "min_presence_confidence": 0.5,
        "min_tracking_confidence": 0.5,
    }


def test_default_config_returns_fresh_copy():
    a = config.default_config()
    a["source"] = "window"
    assert config.default_config()["source"] == "image"  # shared dict untouched


def test_merge_overlays_loaded_values_over_defaults():
    merged = config.merge_config(
        {
            "source": "window",
            "image_path": "a.png",
            "window_title": "Notepad",
            "monitor_index": 2,
            "virtualcam": True,
            "speech": True,
            "parse": True,
            "generate": True,
            "pinch_enter": 0.25,
            "pinch_exit": 0.55,
            "min_detection_confidence": 0.6,
            "min_presence_confidence": 0.7,
            "min_tracking_confidence": 0.8,
        }
    )
    assert merged == {
        "source": "window",
        "image_path": "a.png",
        "window_title": "Notepad",
        "monitor_index": 2,
        "virtualcam": True,
        "speech": True,
        "parse": True,
        "generate": True,
        "pinch_enter": 0.25,
        "pinch_exit": 0.55,
        "min_detection_confidence": 0.6,
        "min_presence_confidence": 0.7,
        "min_tracking_confidence": 0.8,
    }


def test_merge_ignores_unknown_keys():
    merged = config.merge_config({"bogus": 1, "source": "screen"})
    assert "bogus" not in merged
    assert merged["source"] == "screen"


def test_merge_rejects_invalid_source():
    assert config.merge_config({"source": "webcam"})["source"] == "image"


def test_merge_rejects_wrong_types():
    merged = config.merge_config(
        {"image_path": 5, "window_title": None, "virtualcam": "yes", "speech": "on"}
    )
    assert merged["image_path"] == ""
    assert merged["window_title"] == ""
    assert merged["virtualcam"] is False
    assert merged["speech"] is False  # non-bool speech falls back to default


def test_merge_accepts_speech_bool():
    assert config.merge_config({"speech": True})["speech"] is True


def test_merge_accepts_parse_bool():
    assert config.merge_config({"parse": True})["parse"] is True


def test_merge_rejects_non_bool_parse():
    assert config.merge_config({"parse": "yes"})["parse"] is False  # falls back


def test_merge_accepts_generate_bool():
    assert config.merge_config({"generate": True})["generate"] is True


def test_merge_rejects_non_bool_generate():
    assert config.merge_config({"generate": "yes"})["generate"] is False  # falls back


def test_generate_defaults_off():
    assert config.default_config()["generate"] is False


def test_merge_rejects_bool_and_out_of_range_monitor_index():
    assert config.merge_config({"monitor_index": True})["monitor_index"] == 1
    assert config.merge_config({"monitor_index": 0})["monitor_index"] == 1
    assert config.merge_config({"monitor_index": -3})["monitor_index"] == 1
    assert config.merge_config({"monitor_index": 3})["monitor_index"] == 3


def test_merge_non_dict_returns_defaults():
    assert config.merge_config(["not", "a", "dict"]) == config.default_config()


def test_load_missing_file_returns_defaults(tmp_path):
    missing = tmp_path / "nope.json"
    assert config.load_config(missing) == config.default_config()


def test_load_corrupt_file_returns_defaults_with_notice(tmp_path, capsys):
    bad = tmp_path / "handy_config.json"
    bad.write_text("{ this is not json", encoding="utf-8")
    assert config.load_config(bad) == config.default_config()
    assert "config" in capsys.readouterr().err.lower()


def test_save_then_load_roundtrips(tmp_path):
    path = tmp_path / "handy_config.json"
    settings = {
        "source": "screen",
        "image_path": "x.png",
        "window_title": "Term",
        "monitor_index": 2,
        "virtualcam": True,
        "speech": True,
        "parse": True,
        "generate": True,
        "pinch_enter": 0.25,
        "pinch_exit": 0.55,
        "min_detection_confidence": 0.6,
        "min_presence_confidence": 0.7,
        "min_tracking_confidence": 0.8,
    }
    config.save_config(path, settings)
    assert config.load_config(path) == settings


# --------------------------------------------------------------------------- #
# New keys (pinch tuning + startup confidences) validation
# --------------------------------------------------------------------------- #

def test_merge_defaults_pinch_when_absent():
    merged = config.merge_config({"source": "image"})
    assert merged["pinch_enter"] == DEFAULT_PINCH_ENTER
    assert merged["pinch_exit"] == DEFAULT_PINCH_EXIT


def test_merge_rejects_out_of_range_or_wrongtype_pinch():
    # Wrong type / out of range -> that key falls back to its default.
    merged = config.merge_config({"pinch_enter": "lots", "pinch_exit": 99.0})
    assert merged["pinch_enter"] == DEFAULT_PINCH_ENTER
    assert merged["pinch_exit"] == DEFAULT_PINCH_EXIT


def test_merge_rejects_stale_frame_fraction_pinch_values():
    # Values persisted under the old frame-fraction metric (typically 0.01-0.09)
    # fall below the ratio metric's minimum and reset to the new defaults.
    merged = config.merge_config({"pinch_enter": 0.05, "pinch_exit": 0.07})
    assert merged["pinch_enter"] == DEFAULT_PINCH_ENTER
    assert merged["pinch_exit"] == DEFAULT_PINCH_EXIT


def test_merge_enforces_hysteresis_gap_on_pinch_pair():
    # Both in range but inverted -> the shared clamp lifts exit above enter.
    merged = config.merge_config({"pinch_enter": 0.50, "pinch_exit": 0.40})
    assert merged["pinch_enter"] == 0.50
    assert merged["pinch_exit"] > merged["pinch_enter"]


def test_merge_rejects_bool_pinch_values():
    merged = config.merge_config({"pinch_enter": True, "pinch_exit": False})
    assert merged["pinch_enter"] == DEFAULT_PINCH_ENTER
    assert merged["pinch_exit"] == DEFAULT_PINCH_EXIT


def test_merge_validates_confidence_range():
    merged = config.merge_config(
        {
            "min_detection_confidence": 0.9,
            "min_presence_confidence": 1.5,   # out of range -> default
            "min_tracking_confidence": "x",   # wrong type -> default
        }
    )
    assert merged["min_detection_confidence"] == 0.9
    assert merged["min_presence_confidence"] == 0.5
    assert merged["min_tracking_confidence"] == 0.5


def test_merge_rejects_bool_confidence():
    assert config.merge_config({"min_detection_confidence": True})[
        "min_detection_confidence"
    ] == 0.5


def test_save_merges_incomplete_dict(tmp_path):
    path = tmp_path / "handy_config.json"
    config.save_config(path, {"source": "window"})  # partial
    written = json.loads(path.read_text(encoding="utf-8"))
    # Missing keys are filled from defaults on the way out.
    assert written["source"] == "window"
    assert written["monitor_index"] == 1
    assert written["virtualcam"] is False


def test_save_drops_invalid_values(tmp_path):
    path = tmp_path / "handy_config.json"
    config.save_config(path, {"source": "bogus", "monitor_index": 0})
    written = json.loads(path.read_text(encoding="utf-8"))
    assert written["source"] == "image"
    assert written["monitor_index"] == 1

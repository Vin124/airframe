"""Unit tests for src.config: load/save/merge of the panel settings JSON.

Pure functions over ``tmp_path`` -- no UI, no display, no real config file in
the repo. Covers the round-trip, missing/corrupt fallbacks, and the
``merge_config`` validation boundary (unknown keys, wrong types, precedence).
"""

import json

from src import config


def test_default_config_has_expected_keys_and_values():
    cfg = config.default_config()
    assert cfg == {
        "source": "image",
        "image_path": "",
        "window_title": "",
        "monitor_index": 1,
        "virtualcam": False,
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
        }
    )
    assert merged == {
        "source": "window",
        "image_path": "a.png",
        "window_title": "Notepad",
        "monitor_index": 2,
        "virtualcam": True,
    }


def test_merge_ignores_unknown_keys():
    merged = config.merge_config({"bogus": 1, "source": "screen"})
    assert "bogus" not in merged
    assert merged["source"] == "screen"


def test_merge_rejects_invalid_source():
    assert config.merge_config({"source": "webcam"})["source"] == "image"


def test_merge_rejects_wrong_types():
    merged = config.merge_config(
        {"image_path": 5, "window_title": None, "virtualcam": "yes"}
    )
    assert merged["image_path"] == ""
    assert merged["window_title"] == ""
    assert merged["virtualcam"] is False


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
    }
    config.save_config(path, settings)
    assert config.load_config(path) == settings


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

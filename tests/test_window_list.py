"""Unit tests for src.window_list: title filtering/dedup/sort + failure -> [].

The Win32 enumeration is injected as a fake so the pure logic is exercised
without a display or real windows; the failure path uses an enumerator that
raises.
"""

from src.window_list import list_window_titles


def test_filters_empty_and_whitespace_titles():
    titles = list_window_titles(lambda: ["Notepad", "", "   ", "Calculator"])
    assert titles == ["Calculator", "Notepad"]


def test_excludes_the_apps_own_windows_case_insensitively():
    titles = list_window_titles(
        lambda: ["handy", "HANDY CONTROL", "Handy Control", "Notepad"]
    )
    assert titles == ["Notepad"]  # every casing of the own titles dropped


def test_dedupes_case_insensitively_keeping_first_casing():
    titles = list_window_titles(lambda: ["Notepad", "notepad", "NOTEPAD"])
    assert titles == ["Notepad"]  # collapsed to one, first-seen casing


def test_sorts_case_insensitively():
    titles = list_window_titles(lambda: ["zebra", "Apple", "banana"])
    assert titles == ["Apple", "banana", "zebra"]


def test_strips_surrounding_whitespace_before_dedup():
    titles = list_window_titles(lambda: ["  Notepad  ", "Notepad"])
    assert titles == ["Notepad"]


def test_ignores_non_string_entries():
    titles = list_window_titles(lambda: ["Notepad", None, 5, "Calc"])
    assert titles == ["Calc", "Notepad"]


def test_enumerator_failure_returns_empty_list():
    def _boom():
        raise OSError("EnumWindows exploded")

    assert list_window_titles(_boom) == []


def test_empty_enumeration_returns_empty_list():
    assert list_window_titles(lambda: []) == []

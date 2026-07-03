"""Unit tests for src.hud: pure text composition, no drawing surface needed."""

from src.hud import (
    GESTURE_LABEL_GRAB,
    GESTURE_LABEL_NONE,
    GESTURE_LABEL_OPEN,
    GESTURE_LABEL_PINCH,
    KEY_LEGEND_LINES,
    STATUS_GRABBED,
    STATUS_RESIZING,
    VIRTUALCAM_OFF,
    VIRTUALCAM_ON,
    build_hud_lines,
    build_timing_line,
    gesture_label,
    resize_or_grab_status,
)


# --------------------------------------------------------------------------- #
# gesture_label
# --------------------------------------------------------------------------- #

def test_gesture_label_no_hand_is_none_label():
    assert gesture_label(is_grab=False, is_pinching=False, hand_present=False) == GESTURE_LABEL_NONE


def test_gesture_label_no_hand_ignores_stale_flags():
    # Even if is_grab/is_pinching were somehow left True, no hand -> NONE.
    assert gesture_label(is_grab=True, is_pinching=True, hand_present=False) == GESTURE_LABEL_NONE


def test_gesture_label_open_hand():
    assert gesture_label(is_grab=False, is_pinching=False, hand_present=True) == GESTURE_LABEL_OPEN


def test_gesture_label_pinching():
    assert gesture_label(is_grab=False, is_pinching=True, hand_present=True) == GESTURE_LABEL_PINCH


def test_gesture_label_grab_takes_priority_over_pinch():
    assert gesture_label(is_grab=True, is_pinching=True, hand_present=True) == GESTURE_LABEL_GRAB


# --------------------------------------------------------------------------- #
# resize_or_grab_status
# --------------------------------------------------------------------------- #

def test_status_idle_is_none():
    assert resize_or_grab_status(is_grabbed=False, is_resizing=False) is None


def test_status_grabbed():
    assert resize_or_grab_status(is_grabbed=True, is_resizing=False) == STATUS_GRABBED


def test_status_resizing():
    assert resize_or_grab_status(is_grabbed=False, is_resizing=True) == STATUS_RESIZING


def test_status_resizing_takes_priority_over_grabbed():
    assert resize_or_grab_status(is_grabbed=True, is_resizing=True) == STATUS_RESIZING


# --------------------------------------------------------------------------- #
# build_hud_lines
# --------------------------------------------------------------------------- #

def test_build_hud_lines_includes_mode_fps_gesture_and_legend():
    lines = build_hud_lines("skeleton", 29.97, GESTURE_LABEL_OPEN)
    assert lines[0] == "Mode: skeleton"
    assert lines[1] == "FPS: 30.0"
    assert lines[2] == f"Gesture: {GESTURE_LABEL_OPEN}"
    assert tuple(lines[-2:]) == KEY_LEGEND_LINES


def test_build_hud_lines_no_fps_yet_shows_placeholder():
    lines = build_hud_lines("skeleton", None, GESTURE_LABEL_NONE)
    assert lines[1] == "FPS: --"


def test_build_hud_lines_status_omitted_when_none():
    lines = build_hud_lines("image", 60.0, GESTURE_LABEL_OPEN, status=None)
    assert not any(line.startswith("Status:") for line in lines)


def test_build_hud_lines_status_included_when_present():
    lines = build_hud_lines("image", 60.0, GESTURE_LABEL_PINCH, status=STATUS_GRABBED)
    assert f"Status: {STATUS_GRABBED}" in lines
    # Legend still comes last even with an extra status line.
    assert tuple(lines[-2:]) == KEY_LEGEND_LINES


def test_build_hud_lines_virtualcam_omitted_by_default():
    lines = build_hud_lines("skeleton", 30.0, GESTURE_LABEL_OPEN)
    assert not any(line.startswith("VirtualCam:") for line in lines)


def test_build_hud_lines_virtualcam_on():
    lines = build_hud_lines("skeleton", 30.0, GESTURE_LABEL_OPEN, virtualcam_active=True)
    assert f"VirtualCam: {VIRTUALCAM_ON}" in lines
    assert tuple(lines[-2:]) == KEY_LEGEND_LINES


def test_build_hud_lines_virtualcam_off():
    lines = build_hud_lines("skeleton", 30.0, GESTURE_LABEL_OPEN, virtualcam_active=False)
    assert f"VirtualCam: {VIRTUALCAM_OFF}" in lines
    assert tuple(lines[-2:]) == KEY_LEGEND_LINES


def test_build_hud_lines_virtualcam_and_status_both_present_before_legend():
    lines = build_hud_lines(
        "image", 60.0, GESTURE_LABEL_GRAB, status=STATUS_GRABBED, virtualcam_active=True
    )
    assert f"Status: {STATUS_GRABBED}" in lines
    assert f"VirtualCam: {VIRTUALCAM_ON}" in lines
    assert tuple(lines[-2:]) == KEY_LEGEND_LINES


# --------------------------------------------------------------------------- #
# build_timing_line + timing_line param (Phase 9)
# --------------------------------------------------------------------------- #

def test_build_timing_line_rounds_and_labels_each_stage():
    line = build_timing_line(
        capture_ms=48.6, track_ms=41.2, composite_ms=2.7, sink_ms=0.9, loop_ms=33.4
    )
    assert line == "cap49 trk41 cmp3 snk1 lp33"


def test_build_hud_lines_timing_line_omitted_by_default():
    lines = build_hud_lines("skeleton", 30.0, GESTURE_LABEL_OPEN)
    assert not any(line.startswith("cap") for line in lines)


def test_build_hud_lines_includes_timing_line_before_legend():
    timing = build_timing_line(1, 2, 3, 4, 5)
    lines = build_hud_lines(
        "image", 30.0, GESTURE_LABEL_OPEN, virtualcam_active=False, timing_line=timing
    )
    assert timing in lines
    # Legend still the last two lines even with the timing line added.
    assert tuple(lines[-2:]) == KEY_LEGEND_LINES
    # Timing line sits just above the legend.
    assert lines[-3] == timing

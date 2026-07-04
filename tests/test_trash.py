"""Unit tests for src.trash: trash-zone geometry + drawing. Headless."""

import numpy as np

from src.trash import TRASH_ZONE_FRACTION, draw_trash_zone, in_trash_zone, trash_rect

FRAME_SIZE = (100, 100)


def test_trash_rect_is_bottom_right_corner():
    x0, y0, x1, y1 = trash_rect(FRAME_SIZE)
    side = int(round(100 * TRASH_ZONE_FRACTION))
    assert (x0, y0, x1, y1) == (100 - side, 100 - side, 100, 100)


def test_in_trash_zone_inside_and_outside():
    assert in_trash_zone((0.99, 0.99), FRAME_SIZE) is True   # deep in the corner
    assert in_trash_zone((0.5, 0.5), FRAME_SIZE) is False    # center
    assert in_trash_zone((0.99, 0.1), FRAME_SIZE) is False   # right edge, but high up
    assert in_trash_zone((0.1, 0.99), FRAME_SIZE) is False   # bottom, but left


def test_in_trash_zone_uses_shorter_side_for_a_wide_frame():
    # 200x100 frame: the zone is min(200,100)*frac on each side, anchored bottom-right.
    assert in_trash_zone((0.99, 0.99), (200, 100)) is True
    assert in_trash_zone((0.55, 0.99), (200, 100)) is False  # left of the zone


def test_draw_trash_zone_marks_the_corner():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    draw_trash_zone(frame, armed=False)
    x0, y0, _x1, _y1 = trash_rect((100, 100))
    assert frame[y0:, x0:].sum() > 0   # the corner got painted
    assert frame[:y0, :x0].sum() == 0  # the rest is untouched


def test_draw_trash_zone_armed_is_more_opaque():
    idle = np.zeros((100, 100, 3), dtype=np.uint8)
    armed = np.zeros((100, 100, 3), dtype=np.uint8)
    draw_trash_zone(idle, armed=False)
    draw_trash_zone(armed, armed=True)
    # The armed fill is brighter/stronger, so its corner sums to more.
    x0, y0, _x1, _y1 = trash_rect((100, 100))
    assert armed[y0:, x0:].sum() > idle[y0:, x0:].sum()

"""Unit tests for src.skeleton: pure coordinate conversion and topology sanity.

No webcam, no OpenCV window, no drawing surface required.
"""

from src.skeleton import HAND_CONNECTIONS, normalized_to_pixel
from src.hand_tracker import LANDMARKS_PER_HAND


def test_normalized_center_maps_to_frame_center():
    assert normalized_to_pixel(0.5, 0.5, 640, 480) == (320, 240)


def test_normalized_origin_maps_to_pixel_origin():
    assert normalized_to_pixel(0.0, 0.0, 640, 480) == (0, 0)


def test_normalized_max_maps_to_frame_extent():
    assert normalized_to_pixel(1.0, 1.0, 640, 480) == (640, 480)


def test_conversion_rounds_to_nearest_pixel():
    # 0.44 * 100 = 44.0 -> 44; 0.666 * 100 = 66.6 -> 67 (nearest)
    assert normalized_to_pixel(0.44, 0.666, 100, 100) == (44, 67)


def test_connection_indices_are_within_landmark_range():
    for start_idx, end_idx in HAND_CONNECTIONS:
        assert 0 <= start_idx < LANDMARKS_PER_HAND
        assert 0 <= end_idx < LANDMARKS_PER_HAND
        assert start_idx != end_idx


def test_topology_covers_all_21_landmarks():
    referenced = {idx for pair in HAND_CONNECTIONS for idx in pair}
    assert referenced == set(range(LANDMARKS_PER_HAND))


def test_expected_connection_count():
    # 6 palm + 3 per finger * 5 fingers = 21 connections.
    assert len(HAND_CONNECTIONS) == 21

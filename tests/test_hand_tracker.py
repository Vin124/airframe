"""Unit tests for src.hand_tracker pure logic.

Covers the MediaPipe-result -> plain-tuple conversion and the model-download
guard. No webcam, no MediaPipe model load, and no network access: MediaPipe
landmark objects are faked and the download path is exercised with a local
file / a stubbed urlopen.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest
from mediapipe.tasks.python.core.base_options import BaseOptions

from src.hand_tracker import (
    HandTrackerError,
    LANDMARKS_PER_HAND,
    create_landmarker,
    ensure_model,
    landmarks_to_hands,
)


def _fake_landmark(x, y, z):
    return SimpleNamespace(x=x, y=y, z=z)


_MODEL = Path("unused.task")


def _factory(record, *, fail_gpu=False, fail_all=False):
    """A landmarker factory recording the delegate of each options it's built with.

    ``fail_gpu`` raises when handed GPU options (simulating no GPU delegate on
    Windows, the usual case); ``fail_all`` raises for any delegate.
    """

    def factory(options):
        delegate = options.base_options.delegate
        record.append(delegate)
        if fail_all:
            raise RuntimeError("no delegate available")
        if fail_gpu and delegate == BaseOptions.Delegate.GPU:
            raise RuntimeError("GPU delegate not supported on this platform")
        return delegate  # stand-in "landmarker" tagged with the delegate it used

    return factory


def test_landmarks_to_hands_empty_when_no_hands():
    assert landmarks_to_hands([]) == []


def test_landmarks_to_hands_converts_single_hand_to_tuples():
    hand = [_fake_landmark(i * 0.01, i * 0.02, i * 0.03) for i in range(LANDMARKS_PER_HAND)]

    result = landmarks_to_hands([hand])

    assert len(result) == 1
    assert len(result[0]) == LANDMARKS_PER_HAND
    assert result[0][0] == (0.0, 0.0, 0.0)
    assert result[0][5] == pytest.approx((0.05, 0.10, 0.15))
    # Every landmark is a plain (x, y, z) float tuple, not a MediaPipe object.
    assert all(isinstance(lm, tuple) and len(lm) == 3 for lm in result[0])
    assert all(isinstance(v, float) for lm in result[0] for v in lm)


def test_landmarks_to_hands_handles_multiple_hands():
    hand_a = [_fake_landmark(0.1, 0.2, 0.3)]
    hand_b = [_fake_landmark(0.4, 0.5, 0.6)]

    result = landmarks_to_hands([hand_a, hand_b])

    assert result == [[(0.1, 0.2, 0.3)], [(0.4, 0.5, 0.6)]]


def test_ensure_model_returns_existing_nonempty_file(tmp_path):
    model = tmp_path / "hand_landmarker.task"
    model.write_bytes(b"already here")

    # Existing non-empty file must be returned without any download attempt.
    assert ensure_model(model_path=model, url="http://unused.invalid") == model


def test_ensure_model_downloads_when_missing(tmp_path, monkeypatch):
    model = tmp_path / "models" / "hand_landmarker.task"

    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b"fake-model-bytes"

    monkeypatch.setattr(
        "src.hand_tracker.urllib.request.urlopen",
        lambda url, timeout=None: _FakeResponse(),
    )

    result = ensure_model(model_path=model, url="http://example.invalid/model.task")

    assert result == model
    assert model.read_bytes() == b"fake-model-bytes"


def test_ensure_model_raises_on_empty_download(tmp_path, monkeypatch):
    model = tmp_path / "hand_landmarker.task"

    class _EmptyResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b""

    monkeypatch.setattr(
        "src.hand_tracker.urllib.request.urlopen",
        lambda url, timeout=None: _EmptyResponse(),
    )

    with pytest.raises(HandTrackerError):
        ensure_model(model_path=model, url="http://example.invalid/model.task")


def test_ensure_model_raises_clear_error_on_network_failure(tmp_path, monkeypatch):
    model = tmp_path / "hand_landmarker.task"

    def _boom(url, timeout=None):
        raise OSError("network down")

    monkeypatch.setattr("src.hand_tracker.urllib.request.urlopen", _boom)

    with pytest.raises(HandTrackerError):
        ensure_model(model_path=model, url="http://example.invalid/model.task")


# --------------------------------------------------------------------------- #
# create_landmarker: GPU delegate request + CPU fallback (Phase 9)
# --------------------------------------------------------------------------- #

def test_create_landmarker_cpu_by_default_uses_cpu_delegate():
    record = []
    result = create_landmarker(_factory(record), _MODEL, num_hands=1, gpu=False)
    assert record == [BaseOptions.Delegate.CPU]  # GPU never attempted
    assert result == BaseOptions.Delegate.CPU


def test_create_landmarker_gpu_success_uses_gpu_delegate(capsys):
    record = []
    result = create_landmarker(_factory(record), _MODEL, num_hands=1, gpu=True)
    assert record == [BaseOptions.Delegate.GPU]
    assert result == BaseOptions.Delegate.GPU
    assert capsys.readouterr().err == ""  # no fallback notice when GPU works


def test_create_landmarker_gpu_failure_falls_back_to_cpu(capsys):
    record = []
    result = create_landmarker(
        _factory(record, fail_gpu=True), _MODEL, num_hands=1, gpu=True
    )
    # Tried GPU first, then fell back to CPU.
    assert record == [BaseOptions.Delegate.GPU, BaseOptions.Delegate.CPU]
    assert result == BaseOptions.Delegate.CPU
    err = capsys.readouterr().err
    assert "GPU delegate unavailable" in err
    assert "falling back to CPU" in err


def test_create_landmarker_cpu_failure_is_fatal():
    record = []
    with pytest.raises(HandTrackerError):
        create_landmarker(_factory(record, fail_all=True), _MODEL, num_hands=1, gpu=False)

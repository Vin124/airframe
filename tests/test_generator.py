"""Unit tests for Phase 16 image generator (src.generator + its speech wiring).

All PURE / faked -- NO real torch, diffusers, GPU, or network. The heavy imports
in ``DiffusionGenerator`` are deferred, so a fake ``torch`` module injected via
``sys.modules`` drives the GPU-detection path; the diffusion pipeline itself is
never touched here. Covers: the placeholder card, the RGB->BGRA helper,
``GeneratedAsset`` immutability + duck-compat, the engine-unavailable degrade,
``RequestGenerator`` never raising, and the ``build_speech_on_phrase`` generate
branch (placeholder-then-result via a real ``PreviewInbox``).
"""

import sys
import types

import numpy as np
import pytest

from src.generator import (
    DEFAULT_MODEL_ID,
    GPU_UNAVAILABLE_MESSAGE,
    SOURCE_NAME,
    DiffusionGenerator,
    GeneratedAsset,
    RequestGenerator,
    build_request_generator,
    make_placeholder,
    rgb_to_bgra,
)
from src.spawn import PreviewInbox


# --------------------------------------------------------------------------- #
# make_placeholder: valid BGRA, right size, non-zero pixels
# --------------------------------------------------------------------------- #

def test_make_placeholder_shape_and_dtype():
    card = make_placeholder("a friendly robot", size=(320, 200))
    assert card.shape == (200, 320, 4)  # (h, w, 4)
    assert card.dtype == np.uint8


def test_make_placeholder_is_opaque_and_nonzero():
    card = make_placeholder("a simple A shape")
    assert card[:, :, 3].min() == 255  # fully opaque alpha
    assert card.sum() > 0              # the gray card + label are visible


def test_make_placeholder_empty_query_still_valid():
    card = make_placeholder("")
    assert card.shape[2] == 4
    assert card.sum() > 0  # solid card even with no label text


def test_make_placeholder_long_query_does_not_crash():
    card = make_placeholder("x" * 500)  # truncated internally, no overflow
    assert card.shape[2] == 4


# --------------------------------------------------------------------------- #
# rgb_to_bgra: opaque alpha + correct channel order
# --------------------------------------------------------------------------- #

def test_rgb_to_bgra_channel_order_and_alpha():
    rgb = np.zeros((1, 1, 3), dtype=np.uint8)
    rgb[0, 0] = (10, 20, 30)  # R=10, G=20, B=30
    bgra = rgb_to_bgra(rgb)
    assert bgra.shape == (1, 1, 4)
    assert tuple(int(v) for v in bgra[0, 0]) == (30, 20, 10, 255)  # B, G, R, A


def test_rgb_to_bgra_preserves_dimensions():
    rgb = np.random.randint(0, 256, (7, 5, 3), dtype=np.uint8)
    bgra = rgb_to_bgra(rgb)
    assert bgra.shape == (7, 5, 4)
    # Blue plane of BGRA == red plane of the RGB input.
    assert np.array_equal(bgra[:, :, 0], rgb[:, :, 2])


def test_rgb_to_bgra_rejects_non_rgb():
    with pytest.raises(ValueError):
        rgb_to_bgra(np.zeros((4, 4, 4), dtype=np.uint8))


# --------------------------------------------------------------------------- #
# GeneratedAsset: immutability + duck-compat with RetrievedAsset
# --------------------------------------------------------------------------- #

def test_generated_asset_defaults_and_duck_compat():
    bgra = make_placeholder("q")
    asset = GeneratedAsset(bgra=bgra, query="q")
    assert asset.source_name == SOURCE_NAME == "generator"
    assert asset.is_placeholder is False
    # Duck-compat: the exact trio the Phase 15 spawn path reads.
    assert asset.bgra is bgra
    assert asset.query == "q"


def test_generated_asset_is_frozen():
    asset = GeneratedAsset(bgra=make_placeholder("q"), query="q")
    with pytest.raises(Exception):  # FrozenInstanceError (dataclasses)
        asset.query = "other"  # type: ignore[misc]


def test_generated_asset_placeholder_flag():
    asset = GeneratedAsset(bgra=make_placeholder("q"), query="q", is_placeholder=True)
    assert asset.is_placeholder is True


# --------------------------------------------------------------------------- #
# Fake torch modules to drive DiffusionGenerator's GPU probe (no real torch)
# --------------------------------------------------------------------------- #

def _fake_torch(cuda_available: bool) -> types.ModuleType:
    module = types.ModuleType("torch")
    cuda = types.SimpleNamespace(is_available=lambda: cuda_available)
    module.cuda = cuda  # type: ignore[attr-defined]
    return module


# --------------------------------------------------------------------------- #
# DiffusionGenerator: engine-unavailable degrade path
# --------------------------------------------------------------------------- #

def test_diffusion_generator_no_cuda_is_unavailable(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda_available=False))
    engine = DiffusionGenerator()
    assert engine.available is False
    assert engine.generate("anything") is None  # degrades, never loads a model
    assert GPU_UNAVAILABLE_MESSAGE in capsys.readouterr().err


def test_diffusion_generator_torch_absent_is_unavailable(monkeypatch, capsys):
    # Simulate torch not installed: importing it raises ImportError.
    monkeypatch.setitem(sys.modules, "torch", None)  # `import torch` -> ImportError
    engine = DiffusionGenerator()
    assert engine.available is False
    assert engine.generate("anything") is None
    assert GPU_UNAVAILABLE_MESSAGE in capsys.readouterr().err


def test_diffusion_generator_cuda_present_is_available(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda_available=True))
    engine = DiffusionGenerator()
    assert engine.available is True  # would lazy-load the pipeline on first generate


# --------------------------------------------------------------------------- #
# OOM fallback: primary model OOMs -> retry on the lighter fallback model
# --------------------------------------------------------------------------- #

def test_is_oom_recognizes_oom_and_rejects_others():
    from src.generator import _is_oom

    class OutOfMemoryError(Exception):
        pass

    assert _is_oom(OutOfMemoryError("boom")) is True          # by class name
    assert _is_oom(RuntimeError("CUDA out of memory")) is True  # by message
    assert _is_oom(RuntimeError("some other error")) is False


def test_advance_to_fallback_switches_model_once(monkeypatch):
    from src.generator import DEFAULT_MODEL_ID, FALLBACK_MODEL_ID

    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda_available=True))
    engine = DiffusionGenerator()  # starts on DEFAULT_MODEL_ID
    assert engine._model_id == DEFAULT_MODEL_ID
    assert engine._advance_to_fallback() is True
    assert engine._model_id == FALLBACK_MODEL_ID
    assert engine._advance_to_fallback() is False  # already on the fallback


def test_load_time_oom_falls_back_to_lighter_model(monkeypatch):
    from src.generator import DEFAULT_MODEL_ID, FALLBACK_MODEL_ID

    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda_available=True))
    engine = DiffusionGenerator()
    loaded = []

    def fake_load(model_id):
        loaded.append(model_id)
        if model_id == DEFAULT_MODEL_ID:
            raise RuntimeError("CUDA out of memory")  # primary OOMs on load
        return f"pipe::{model_id}"  # fallback loads fine

    monkeypatch.setattr(engine, "_load_pipeline", fake_load)
    pipe = engine._ensure_pipeline()
    assert pipe == f"pipe::{FALLBACK_MODEL_ID}"
    assert loaded == [DEFAULT_MODEL_ID, FALLBACK_MODEL_ID]
    assert engine.available is True  # fallback kept the engine alive


def test_non_oom_load_failure_disables_without_fallback(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda_available=True))
    engine = DiffusionGenerator()

    def fake_load(model_id):
        raise RuntimeError("model files corrupt")  # not an OOM

    monkeypatch.setattr(engine, "_load_pipeline", fake_load)
    assert engine._ensure_pipeline() is None
    assert engine.available is False  # a non-OOM load failure disables, no retry


def test_inference_oom_retries_on_fallback_model(monkeypatch):
    from src.generator import DEFAULT_MODEL_ID, FALLBACK_MODEL_ID

    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda_available=True))
    engine = DiffusionGenerator()
    calls = []

    def fake_run(prompt):
        calls.append(engine._model_id)
        if engine._model_id == DEFAULT_MODEL_ID:
            return None  # simulate an OOM-swallowed miss on the primary
        return np.zeros((8, 8, 3), dtype=np.uint8)  # fallback succeeds

    monkeypatch.setattr(engine, "_run_inference", fake_run)
    bgra = engine.generate("a friendly robot")
    assert bgra is not None and bgra.shape == (8, 8, 4)
    assert calls == [DEFAULT_MODEL_ID, FALLBACK_MODEL_ID]  # retried on the lighter model


# --------------------------------------------------------------------------- #
# RequestGenerator: never raises; wraps a BGRA into a GeneratedAsset
# --------------------------------------------------------------------------- #

class _OkEngine:
    def generate(self, prompt: str):
        return make_placeholder(prompt, size=(16, 12))


class _NoneEngine:
    def generate(self, prompt: str):
        return None


class _RaisingEngine:
    def generate(self, prompt: str):
        raise RuntimeError("CUDA out of memory")


def test_request_generator_wraps_bgra_into_asset():
    asset = RequestGenerator(_OkEngine()).generate("a red letter A")
    assert isinstance(asset, GeneratedAsset)
    assert asset.query == "a red letter A"
    assert asset.source_name == "generator"
    assert asset.bgra.shape == (12, 16, 4)
    assert asset.is_placeholder is False


def test_request_generator_blank_query_is_none():
    assert RequestGenerator(_OkEngine()).generate("   ") is None


def test_request_generator_engine_none_is_none():
    assert RequestGenerator(_NoneEngine()).generate("x") is None


def test_request_generator_never_raises(capsys):
    result = RequestGenerator(_RaisingEngine()).generate("x")
    assert result is None  # swallowed
    assert "generator failed" in capsys.readouterr().err


def test_build_request_generator_probes_gpu(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda_available=False))
    gen = build_request_generator()
    assert isinstance(gen, RequestGenerator)
    assert gen.generate("x") is None  # unavailable engine -> None
    assert GPU_UNAVAILABLE_MESSAGE in capsys.readouterr().err


def test_default_model_id_is_a_turbo_model():
    assert DEFAULT_MODEL_ID == "stabilityai/sdxl-turbo"


# --------------------------------------------------------------------------- #
# build_speech_on_phrase generate branch (placeholder FIRST, then result)
# --------------------------------------------------------------------------- #

class _FakeParser:
    """Returns one fixed ParsedRequest for every phrase."""

    def __init__(self, request):
        self._request = request

    def parse(self, text):
        return self._request


class _FakeRetriever:
    """Records whether it was reached and returns a distinct retrieve asset."""

    def __init__(self):
        self.calls = []

    def retrieve(self, request):
        self.calls.append(request.query)
        return GeneratedAsset(bgra=make_placeholder("retrieved", size=(4, 3)), query="retrieved")


class _RecordingGenerator:
    """A RequestGenerator stand-in: returns a fixed 8x6 generated asset."""

    def __init__(self, result=True):
        self.calls = []
        self._result = result

    def generate(self, query):
        self.calls.append(query)
        if not self._result:
            return None
        return GeneratedAsset(bgra=make_placeholder(query, size=(8, 6)), query=query)


def _generate_request():
    from src.parser import ROUTE_GENERATE, SOURCE_GENERATOR, ParsedRequest

    return ParsedRequest("insert", "a friendly robot", ROUTE_GENERATE, SOURCE_GENERATOR)


def _retrieve_request():
    from src.parser import ROUTE_RETRIEVE, SOURCE_WIKIMEDIA, ParsedRequest

    return ParsedRequest("insert", "CFTR", ROUTE_RETRIEVE, SOURCE_WIKIMEDIA)


def _wire(monkeypatch, request, generator, generate_enabled, inbox, retriever=None):
    """Wire build_speech_on_phrase with fake parser/retriever/generator."""
    import src.generator as generator_module
    import src.parser as parser_module
    import src.retriever as retriever_module
    from src.launch import build_speech_on_phrase

    monkeypatch.setattr(parser_module, "build_request_parser", lambda: _FakeParser(request))
    monkeypatch.setattr(
        retriever_module, "build_retriever",
        lambda *a, **k: retriever if retriever is not None else _FakeRetriever(),
    )
    monkeypatch.setattr(generator_module, "build_request_generator", lambda *a, **k: generator)
    return build_speech_on_phrase(True, inbox, generate_enabled)


def test_generate_branch_publishes_placeholder_then_result(monkeypatch, capsys):
    inbox = PreviewInbox()
    generator = _RecordingGenerator()
    on_phrase = _wire(monkeypatch, _generate_request(), generator, True, inbox)

    published = []
    real_publish = inbox.publish
    monkeypatch.setattr(
        inbox, "publish", lambda asset: (published.append(asset), real_publish(asset))[1]
    )

    on_phrase("can you conjure a friendly robot")

    # Two publishes: the placeholder FIRST, then the finished result.
    assert len(published) == 2
    assert published[0].is_placeholder is True
    assert published[1].is_placeholder is False
    assert published[0].query == "a friendly robot"
    # Latest-wins: the result is what the main loop takes (replaces the placeholder).
    assert inbox.take() is published[1]
    assert generator.calls == ["a friendly robot"]
    assert "[generator] generated 8x6 for 'a friendly robot'" in capsys.readouterr().out


def test_generate_branch_disabled_ignores_and_logs(monkeypatch, capsys):
    inbox = PreviewInbox()
    on_phrase = _wire(monkeypatch, _generate_request(), None, False, inbox)
    on_phrase("can you conjure a friendly robot")
    out = capsys.readouterr().out
    assert "[generator] disabled (pass --generate) for 'a friendly robot'" in out
    assert inbox.take() is None  # nothing docked when generation is off


def test_generate_off_still_retrieves(monkeypatch, capsys):
    # With --generate OFF, a RETRIEVE request must still work end to end.
    inbox = PreviewInbox()
    retriever = _FakeRetriever()
    on_phrase = _wire(
        monkeypatch, _retrieve_request(), None, False, inbox, retriever=retriever
    )
    on_phrase("we're looking at CFTR")
    assert retriever.calls == ["CFTR"]  # retrieve path reached
    docked = inbox.take()
    assert docked is not None and docked.query == "retrieved"
    assert "[retriever] got 4x3 image for 'CFTR'" in capsys.readouterr().out


def test_generate_branch_failure_leaves_placeholder(monkeypatch, capsys):
    inbox = PreviewInbox()
    on_phrase = _wire(
        monkeypatch, _generate_request(), _RecordingGenerator(result=False), True, inbox
    )
    on_phrase("conjure a friendly robot")
    # The placeholder was published and remains for the timer to auto-dismiss.
    docked = inbox.take()
    assert docked is not None and docked.is_placeholder is True
    assert "[generator] failed for 'a friendly robot'" in capsys.readouterr().out

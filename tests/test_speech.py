"""Unit tests for src.speech: VAD segmentation, rolling transcript, worker, wiring.

All pure/faked -- no real microphone, ASR model, or network. The VAD math and the
rolling-buffer snapshot are exercised as plain functions; the SpeechWorker is
driven through a fake mic + fake engine (via ``poll_once`` and a real thread
lifecycle); engine selection uses injected factories that raise on demand.
"""

import numpy as np
import pytest

from src.speech import (
    MAX_PHRASE_SECONDS,
    ROLLING_TRANSCRIPT_SIZE,
    SILENCE_GAP_SECONDS,
    VAD_RMS_THRESHOLD,
    PhraseSegmenter,
    SpeechWorker,
    Transcript,
    append_phrase,
    build_speech_worker,
    default_on_phrase,
    is_voiced,
    rms,
    select_engine,
    speech_session,
)


# --------------------------------------------------------------------------- #
# Helpers: synthetic audio blocks
# --------------------------------------------------------------------------- #

def _voiced(n: int = 8, amp: float = 0.3) -> np.ndarray:
    """A constant-amplitude block whose RMS is comfortably above the threshold."""
    return np.full(n, amp, dtype=np.float32)


def _silence(n: int = 8) -> np.ndarray:
    """A zero block (RMS 0)."""
    return np.zeros(n, dtype=np.float32)


def _wait_until(predicate, timeout: float = 1.0) -> bool:
    """Poll ``predicate`` until true or ``timeout`` elapses; return the result."""
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.002)
    return predicate()


# --------------------------------------------------------------------------- #
# VAD primitives: rms / is_voiced
# --------------------------------------------------------------------------- #

def test_rms_of_silence_is_zero():
    assert rms(_silence()) == 0.0


def test_rms_of_empty_block_is_zero():
    assert rms(np.array([], dtype=np.float32)) == 0.0


def test_rms_of_constant_block_equals_amplitude():
    assert rms(np.full(16, 0.5, dtype=np.float32)) == pytest.approx(0.5)


def test_is_voiced_true_above_threshold():
    assert is_voiced(_voiced()) is True


def test_is_voiced_false_for_silence():
    assert is_voiced(_silence()) is False


def test_is_voiced_respects_custom_threshold():
    quiet = np.full(16, 0.02, dtype=np.float32)  # rms 0.02
    assert is_voiced(quiet, threshold=0.01) is True
    assert is_voiced(quiet, threshold=0.05) is False


def test_default_threshold_separates_quiet_noise_from_speech():
    # Quiet-room noise sits below the default; speech-level energy above it.
    assert is_voiced(np.full(16, 0.004, dtype=np.float32), VAD_RMS_THRESHOLD) is False
    assert is_voiced(np.full(16, 0.05, dtype=np.float32), VAD_RMS_THRESHOLD) is True


# --------------------------------------------------------------------------- #
# PhraseSegmenter: finalize on silence gap / max cap; skip pure silence
# --------------------------------------------------------------------------- #

def _segmenter(**kw) -> PhraseSegmenter:
    # sample_rate 8 with 8-sample blocks -> each block is exactly 1.0s, so the
    # timing math is trivial to reason about in tests.
    defaults = dict(sample_rate=8, silence_gap_seconds=2.0, max_phrase_seconds=5.0)
    defaults.update(kw)
    return PhraseSegmenter(**defaults)


def test_segmenter_skips_leading_silence():
    seg = _segmenter()
    assert seg.push(_silence()) is None
    assert seg.push(_silence()) is None  # no phrase started -> nothing buffered


def test_segmenter_accumulates_voiced_until_silence_gap():
    seg = _segmenter(silence_gap_seconds=2.0)  # 2 silent 1.0s blocks finalize
    assert seg.push(_voiced()) is None  # phrase starts
    assert seg.push(_voiced()) is None
    assert seg.push(_silence()) is None  # 1.0s silence < 2.0s gap
    result = seg.push(_silence())  # 2.0s silence >= gap -> finalize
    assert result is not None
    assert result.shape[0] == 8 * 4  # all four blocks concatenated (incl. silence)


def test_segmenter_finalizes_on_max_length_cap():
    # No trailing silence at all: a long monologue still flushes at the cap.
    seg = _segmenter(silence_gap_seconds=100.0, max_phrase_seconds=3.0)
    assert seg.push(_voiced()) is None  # 1.0s
    assert seg.push(_voiced()) is None  # 2.0s
    result = seg.push(_voiced())  # 3.0s >= cap -> finalize
    assert result is not None
    assert result.shape[0] == 8 * 3


def test_segmenter_resets_after_finalizing():
    seg = _segmenter(silence_gap_seconds=1.0)
    seg.push(_voiced())
    seg.push(_silence())  # finalize (1.0s silence >= 1.0s gap)
    # A fresh phrase can start again immediately after.
    assert seg.push(_voiced()) is None
    final = seg.push(_silence())
    assert final is not None
    assert final.shape[0] == 8 * 2


def test_segmenter_internal_silence_does_not_finalize_early():
    seg = _segmenter(silence_gap_seconds=2.0)
    seg.push(_voiced())
    assert seg.push(_silence()) is None  # 1.0s pause mid-phrase
    seg.push(_voiced())  # voiced again -> silence run resets
    assert seg.push(_silence()) is None  # only 1.0s of new silence
    assert seg.push(_silence()) is not None  # now 2.0s -> finalize


def test_module_constants_are_sane():
    assert SILENCE_GAP_SECONDS < MAX_PHRASE_SECONDS
    assert 0.0 < VAD_RMS_THRESHOLD < 1.0
    assert ROLLING_TRANSCRIPT_SIZE >= 1


# --------------------------------------------------------------------------- #
# Transcript + append_phrase: immutable, last-N rolling buffer
# --------------------------------------------------------------------------- #

def test_transcript_defaults_to_empty():
    t = Transcript()
    assert t.phrases == ()
    assert t.text == ""


def test_transcript_text_joins_phrases():
    assert Transcript(("hello", "world")).text == "hello world"


def test_transcript_is_immutable():
    t = Transcript(("a",))
    with pytest.raises(AttributeError):
        t.phrases = ("b",)  # frozen


def test_transcript_equality():
    assert Transcript(("a", "b")) == Transcript(("a", "b"))
    assert Transcript(("a",)) != Transcript(("b",))


def test_append_phrase_returns_new_tuple():
    original = ("a",)
    grown = append_phrase(original, "b", max_len=10)
    assert grown == ("a", "b")
    assert original == ("a",)  # not mutated


def test_append_phrase_trims_to_last_n():
    phrases = ("1", "2", "3")
    grown = append_phrase(phrases, "4", max_len=3)
    assert grown == ("2", "3", "4")  # oldest dropped


def test_append_phrase_no_trim_when_max_len_nonpositive():
    grown = append_phrase(("1", "2"), "3", max_len=0)
    assert grown == ("1", "2", "3")


# --------------------------------------------------------------------------- #
# Engine selection + fallback (injected factories, no real ASR)
# --------------------------------------------------------------------------- #

class _FakeEngine:
    """Stand-in TranscriberEngine: returns a deterministic label per call."""

    def __init__(self, label: str = "phrase") -> None:
        self._label = label
        self.calls = 0

    def transcribe(self, audio) -> str:
        self.calls += 1
        return f"{self._label}{self.calls}"


def _ok_factory(engine):
    return lambda: engine


def _boom_factory(message):
    def factory():
        raise RuntimeError(message)

    return factory


def test_select_engine_prefers_moonshine(capsys):
    moon = _FakeEngine("moon")
    engine = select_engine(
        moonshine_factory=_ok_factory(moon), whisper_factory=_boom_factory("unused")
    )
    assert engine is moon
    assert "Moonshine" in capsys.readouterr().out


def test_select_engine_falls_back_to_whisper(capsys):
    whisper = _FakeEngine("whisper")
    engine = select_engine(
        moonshine_factory=_boom_factory("no moonshine"),
        whisper_factory=_ok_factory(whisper),
    )
    assert engine is whisper
    captured = capsys.readouterr()
    assert "Moonshine speech engine unavailable" in captured.err
    assert "faster-whisper" in captured.out


def test_select_engine_returns_none_when_both_fail(capsys):
    engine = select_engine(
        moonshine_factory=_boom_factory("no moonshine"),
        whisper_factory=_boom_factory("no whisper"),
    )
    assert engine is None
    assert "no speech engine available" in capsys.readouterr().err


def test_select_engine_falls_back_when_first_transcribe_fails(capsys):
    # Construction succeeding is not enough: an engine whose model resolves
    # lazily can fail on the FIRST transcribe -- the warm-up inside
    # select_engine must catch that and fall back down the chain.
    class _LazyBoomEngine:
        def transcribe(self, audio):
            raise RuntimeError("model download failed")

    whisper = _FakeEngine("whisper")
    engine = select_engine(
        moonshine_factory=_ok_factory(_LazyBoomEngine()),
        whisper_factory=_ok_factory(whisper),
    )
    assert engine is whisper
    assert "Moonshine speech engine unavailable" in capsys.readouterr().err


def test_select_engine_skips_moonshine_when_not_preferred():
    whisper = _FakeEngine("whisper")
    moon_factory = _boom_factory("should not be called")
    engine = select_engine(
        prefer_moonshine=False,
        moonshine_factory=moon_factory,
        whisper_factory=_ok_factory(whisper),
    )
    assert engine is whisper  # moonshine never attempted


# --------------------------------------------------------------------------- #
# SpeechWorker: poll_once pipeline + real-thread lifecycle (fake mic + engine)
# --------------------------------------------------------------------------- #

class _FakeMic:
    """Fake MicStream: yields queued blocks, then None (idle). Records lifecycle."""

    def __init__(self, blocks=None, open_error=None) -> None:
        self._blocks = list(blocks or [])
        self._open_error = open_error
        self.opened = False
        self.closed = False

    def open(self):
        self.opened = True
        if self._open_error is not None:
            raise self._open_error
        return self

    def push(self, block):
        self._blocks.append(block)

    def read(self):
        if self._blocks:
            return self._blocks.pop(0)
        return None  # drained -> worker idles

    def close(self):
        self.closed = True


def _worker(mic, engine=None, on_phrase=None, **seg_kw):
    seg = PhraseSegmenter(sample_rate=8, silence_gap_seconds=1.0, max_phrase_seconds=5.0, **seg_kw)
    return SpeechWorker(
        engine or _FakeEngine(),
        mic,
        on_phrase=on_phrase,
        segmenter=seg,
    )


def test_poll_once_returns_false_when_no_block():
    worker = _worker(_FakeMic([]))
    assert worker.poll_once() is False
    assert worker.latest().phrases == ()


def test_poll_once_publishes_phrase_and_fires_callback():
    fired = []
    # voiced then a silence block (1.0s >= 1.0s gap) -> one finalized phrase.
    mic = _FakeMic([_voiced(), _silence()])
    worker = _worker(mic, on_phrase=fired.append)
    assert worker.poll_once() is True  # voiced buffered, nothing finalized
    assert worker.latest().phrases == ()
    assert worker.poll_once() is True  # silence -> finalize -> transcribe/publish
    assert worker.latest().phrases == ("phrase1",)
    assert fired == ["phrase1"]


def test_multiple_phrases_published_in_order():
    fired = []
    mic = _FakeMic([_voiced(), _silence(), _voiced(), _silence()])
    worker = _worker(mic, on_phrase=fired.append)
    while worker.poll_once():
        pass
    assert worker.latest().phrases == ("phrase1", "phrase2")
    assert fired == ["phrase1", "phrase2"]


def test_blank_transcription_is_dropped():
    class _BlankEngine:
        def transcribe(self, audio):
            return "   "  # whitespace only

    fired = []
    mic = _FakeMic([_voiced(), _silence()])
    worker = _worker(mic, engine=_BlankEngine(), on_phrase=fired.append)
    while worker.poll_once():
        pass
    assert worker.latest().phrases == ()  # nothing published
    assert fired == []  # callback not fired for empty text


def test_worker_thread_lifecycle_publishes_then_joins():
    fired = []
    mic = _FakeMic([_voiced(), _silence()])
    worker = _worker(mic, on_phrase=fired.append)
    worker.start()
    try:
        assert _wait_until(lambda: worker.latest().phrases == ("phrase1",))
        assert fired == ["phrase1"]
    finally:
        assert worker.stop() is True  # joins cleanly
    assert mic.closed is True  # mic released on the way out


def test_worker_exits_cleanly_when_mic_open_fails(capsys):
    from src.speech import SpeechError

    mic = _FakeMic([], open_error=SpeechError("no device"))
    worker = _worker(mic)
    worker.start()
    assert worker.stop() is True
    assert "no device" in capsys.readouterr().err
    assert worker.latest().phrases == ()  # nothing published


def test_worker_crash_is_recorded_and_announced_once(capsys):
    class _BoomEngine:
        def transcribe(self, audio):
            raise RuntimeError("engine boom")

    mic = _FakeMic([_voiced(), _silence()])
    worker = _worker(mic, engine=_BoomEngine())
    worker.poll_once()  # voiced buffered
    with pytest.raises(RuntimeError):
        worker.poll_once()  # finalize -> transcribe raises (poll propagates)
    worker._on_crash(RuntimeError("engine boom"))
    assert isinstance(worker.error, RuntimeError)
    assert "speech transcription stopped" in capsys.readouterr().err


def test_worker_thread_crash_closes_mic_and_sets_error():
    # End-to-end through the REAL thread loop: a transcribe crash must record
    # the error via _on_crash, terminate the loop, and still close the mic in
    # the finally.
    class _BoomEngine:
        def transcribe(self, audio):
            raise RuntimeError("engine boom")

    mic = _FakeMic([_voiced(), _silence()])
    worker = _worker(mic, engine=_BoomEngine())
    worker.start()
    try:
        assert _wait_until(lambda: worker.error is not None)
        assert _wait_until(lambda: mic.closed)
    finally:
        assert worker.stop() is True


def test_raising_on_phrase_does_not_kill_worker(capsys):
    def bad_callback(text):
        raise ValueError("consumer bug")

    mic = _FakeMic([_voiced(), _silence(), _voiced(), _silence()])
    worker = _worker(mic, on_phrase=bad_callback)
    while worker.poll_once():
        pass
    # Both phrases still transcribed + published; only the callback side effect
    # is lost, with a warning per raise.
    assert worker.latest().phrases == ("phrase1", "phrase2")
    assert "phrase callback raised" in capsys.readouterr().err


def test_stop_is_safe_before_start():
    worker = _worker(_FakeMic([]))
    assert worker.stop() is True  # never started -> no thread -> truthful True


# --------------------------------------------------------------------------- #
# build_speech_worker + speech_session (degradation + lifecycle)
# --------------------------------------------------------------------------- #

def test_build_speech_worker_uses_injected_engine_and_mic():
    engine = _FakeEngine()
    mic = _FakeMic([])
    worker = build_speech_worker(engine=engine, mic=mic)
    assert isinstance(worker, SpeechWorker)


def test_speech_session_disabled_yields_none():
    with speech_session(False) as worker:
        assert worker is None


def test_speech_session_yields_none_when_no_engine(monkeypatch):
    # Force engine selection to fail so the session degrades to no-op.
    monkeypatch.setattr("src.speech.select_engine", lambda *a, **k: None)
    with speech_session(True) as worker:
        assert worker is None


def test_default_on_phrase_prints_with_prefix(capsys):
    default_on_phrase("hello there")
    assert "[speech] hello there" in capsys.readouterr().out

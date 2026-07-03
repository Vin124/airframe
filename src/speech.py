"""Always-listening speech layer: mic capture + live transcription, off the main loop.

Phase 12 adds a fourth (optional) thread to the pipeline. ``SpeechWorker`` OWNS
the microphone stream and the ASR engine and runs entirely on its own daemon
thread, so a slow ~1s transcription never touches the render/tracking frame rate.
It follows the exact single-owner discipline the rest of the app uses (see
``src.tracking.TrackingWorker``, the closest template):

State ownership (the app's single-writer rule):
    - THIS thread owns the ``MicStream`` and the ``TranscriberEngine``. Nothing
      else touches them.
    - The rolling transcript is published as an IMMUTABLE ``Transcript`` snapshot
      with one atomic reference store (GIL-safe, no lock), latest-wins -- the same
      hand-off the capture/tracking threads use. ``latest()`` reads it, never
      blocking.
    - ``on_phrase`` (optional) fires once per FINALIZED phrase. IT RUNS ON THE
      SPEECH THREAD, not the caller's -- keep it cheap and thread-safe (the
      default just prints to the console).

Segmentation:
    Simple energy-based VAD in pure, unit-testable functions (``rms`` /
    ``is_voiced`` + ``PhraseSegmenter``): accumulate voiced audio, finalize a
    phrase after ~0.7s of trailing silence or a ~10s max-length cap, and skip
    pure silence entirely (a phrase only starts on a voiced block). Only finalized
    segments are transcribed -- no partial/streaming decode -- to keep CPU/latency
    low. The rolling transcript keeps the last ``ROLLING_TRANSCRIPT_SIZE`` phrases.

Engine selection + degradation (the app's failure policy: degrade, never die):
    ``select_engine`` tries Moonshine first (``useful-moonshine-onnx``), falling
    back to faster-whisper on ANY import/init failure, and to no-speech if both
    fail -- each with a one-line stderr notice. Every heavy dependency
    (``moonshine_onnx``, ``faster_whisper``, ``sounddevice``) is imported LAZILY
    inside the class that needs it, so the base pipeline runs with none of them
    installed (mirroring how ``pyvirtualcam`` is deferred in the virtual-cam sink).
    A missing/failed mic prints one stderr notice and the worker exits cleanly;
    the app keeps running.
"""

import sys
import threading
import time
from contextlib import contextmanager
from typing import Callable, Iterator, List, Optional, Protocol, Tuple

import numpy as np

# Audio format pulled from the mic and fed to every engine: 16 kHz mono float32.
SAMPLE_RATE = 16000
# Block size pulled per read (~0.2s). Small enough to keep VAD responsive, large
# enough that the capture loop isn't churning; a blocking read paces the thread.
BLOCK_SECONDS = 0.2
BLOCK_FRAMES = int(SAMPLE_RATE * BLOCK_SECONDS)

# VAD / phrase-segmentation knobs (see PhraseSegmenter). RMS is computed on the
# float32 samples in [-1, 1]: quiet-room noise sits around ~0.001-0.005, speech
# around ~0.02-0.1, so 0.01 cleanly separates the two by eye.
VAD_RMS_THRESHOLD = 0.01
# Finalize a phrase after this much TRAILING silence (a natural pause).
SILENCE_GAP_SECONDS = 0.7
# ...or after this long regardless, so a monologue still flushes periodically.
MAX_PHRASE_SECONDS = 10.0
# Rolling transcript keeps the last N finalized phrases.
ROLLING_TRANSCRIPT_SIZE = 10

# Sleep when the mic yields nothing yet, so the loop doesn't busy-spin.
_IDLE_SLEEP_SECONDS = 0.005
# How long stop() waits for the speech thread to exit (a transcribe can be in
# flight when stop is requested; the join must never hang app shutdown).
_JOIN_TIMEOUT_SECONDS = 2.0

# Default model ids for the two engines.
MOONSHINE_MODEL = "moonshine/base"
WHISPER_MODEL = "base.en"


class SpeechError(RuntimeError):
    """Raised for a mic/engine failure the worker turns into a graceful exit."""


# --------------------------------------------------------------------------- #
# Energy-based VAD + phrase segmentation (PURE -- no threads, no I/O)
# --------------------------------------------------------------------------- #

def rms(block: np.ndarray) -> float:
    """Root-mean-square amplitude of one audio block (0.0 for an empty block)."""
    arr = np.asarray(block, dtype=np.float32)
    if arr.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(arr))))


def is_voiced(block: np.ndarray, threshold: float = VAD_RMS_THRESHOLD) -> bool:
    """True iff ``block``'s energy is at/above ``threshold`` (i.e. not silence)."""
    return rms(block) >= threshold


class PhraseSegmenter:
    """Accumulate voiced audio into phrases, finalizing on a silence gap or cap.

    Fed one block at a time via ``push``; owns its running buffer (single owner:
    the speech thread). Pure of threads/I/O so it's driven directly in tests. A
    phrase only STARTS on a voiced block (leading silence is skipped) and, once
    started, keeps internal silence too so mid-phrase pauses don't chop words. It
    finalizes -- returning the concatenated float32 audio and resetting -- when
    trailing silence reaches ``silence_gap_seconds`` or the phrase reaches
    ``max_phrase_seconds``. Each block's duration is derived from its length and
    the sample rate, so variable block sizes are handled correctly.
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        silence_gap_seconds: float = SILENCE_GAP_SECONDS,
        max_phrase_seconds: float = MAX_PHRASE_SECONDS,
        energy_threshold: float = VAD_RMS_THRESHOLD,
    ) -> None:
        self._sample_rate = sample_rate
        self._silence_gap_seconds = silence_gap_seconds
        self._max_phrase_seconds = max_phrase_seconds
        self._energy_threshold = energy_threshold
        self._buffer: List[np.ndarray] = []
        self._phrase_seconds = 0.0
        self._silence_seconds = 0.0

    def push(self, block: np.ndarray) -> Optional[np.ndarray]:
        """Feed one block; return finalized phrase audio, or None if none yet."""
        voiced = is_voiced(block, self._energy_threshold)
        block_seconds = len(block) / self._sample_rate
        if not self._buffer:
            if not voiced:
                return None  # skip leading silence -- a phrase starts on voice
            self._buffer.append(block)
            self._phrase_seconds = block_seconds
            self._silence_seconds = 0.0
            return self._maybe_cap()
        self._buffer.append(block)
        self._phrase_seconds += block_seconds
        self._silence_seconds = 0.0 if voiced else self._silence_seconds + block_seconds
        if self._silence_seconds >= self._silence_gap_seconds:
            return self._finalize()
        return self._maybe_cap()

    def _maybe_cap(self) -> Optional[np.ndarray]:
        """Finalize if the phrase has hit the max-length cap; else None."""
        if self._phrase_seconds >= self._max_phrase_seconds:
            return self._finalize()
        return None

    def _finalize(self) -> np.ndarray:
        """Concatenate the buffered phrase, reset, and return it (float32)."""
        audio = np.concatenate(self._buffer).astype(np.float32)
        self.reset()
        return audio

    def reset(self) -> None:
        """Drop any in-progress phrase (idempotent)."""
        self._buffer = []
        self._phrase_seconds = 0.0
        self._silence_seconds = 0.0


# --------------------------------------------------------------------------- #
# Rolling transcript snapshot (immutable, latest-wins publish)
# --------------------------------------------------------------------------- #

class Transcript:
    """Immutable snapshot of the last N finalized phrases (published atomically)."""

    __slots__ = ("phrases",)

    def __init__(self, phrases: Tuple[str, ...] = ()) -> None:
        object.__setattr__(self, "phrases", tuple(phrases))

    def __setattr__(self, name: str, value: object) -> None:  # frozen
        raise AttributeError("Transcript is immutable")

    @property
    def text(self) -> str:
        """The phrases joined into one rolling-transcript line."""
        return " ".join(self.phrases)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Transcript) and other.phrases == self.phrases

    def __repr__(self) -> str:
        return f"Transcript(phrases={self.phrases!r})"


def append_phrase(
    phrases: Tuple[str, ...], phrase: str, max_len: int = ROLLING_TRANSCRIPT_SIZE
) -> Tuple[str, ...]:
    """Return a NEW tuple with ``phrase`` appended, trimmed to the last ``max_len``.

    Never mutates ``phrases`` (immutable rolling buffer). ``max_len <= 0`` keeps
    everything (no trim).
    """
    grown = (*phrases, phrase)
    if max_len > 0:
        return grown[-max_len:]
    return grown


# --------------------------------------------------------------------------- #
# Transcriber engines (heavy deps imported lazily; try Moonshine, then whisper)
# --------------------------------------------------------------------------- #

class TranscriberEngine(Protocol):
    """One method: transcribe 16 kHz mono float32 audio to a text string."""

    def transcribe(self, audio: np.ndarray) -> str: ...


class MoonshineEngine:
    """Moonshine ASR via ``useful-moonshine-onnx`` (imported lazily).

    Constructing it imports ``moonshine_onnx`` and resolves its ``transcribe``
    entry point (the ONNX build ships a module-level ``transcribe(audio, model)``),
    so an import/setup failure surfaces here where ``select_engine`` catches it and
    falls back to faster-whisper.
    """

    def __init__(self, model_name: str = MOONSHINE_MODEL) -> None:
        import moonshine_onnx  # deferred: optional dep

        self._model_name = model_name
        self._transcribe = moonshine_onnx.transcribe

    def transcribe(self, audio: np.ndarray) -> str:
        arr = np.asarray(audio, dtype=np.float32)
        result = self._transcribe(arr, self._model_name)
        # moonshine_onnx.transcribe returns a list of decoded strings.
        if isinstance(result, (list, tuple)):
            return " ".join(str(part) for part in result).strip()
        return str(result).strip()


class FasterWhisperEngine:
    """faster-whisper ASR (imported lazily), CPU int8, English base model."""

    def __init__(
        self,
        model_name: str = WHISPER_MODEL,
        device: str = "cpu",
        compute_type: str = "int8",
    ) -> None:
        from faster_whisper import WhisperModel  # deferred: optional dep

        self._model = WhisperModel(model_name, device=device, compute_type=compute_type)

    def transcribe(self, audio: np.ndarray) -> str:
        arr = np.asarray(audio, dtype=np.float32)
        segments, _info = self._model.transcribe(arr, language="en")
        return " ".join(seg.text.strip() for seg in segments).strip()


EngineFactory = Callable[[], TranscriberEngine]

# Synthetic audio (samples of silence) used to validate a freshly built engine
# end-to-end inside ``select_engine`` (0.5 s at SAMPLE_RATE).
_WARMUP_SAMPLES = SAMPLE_RATE // 2


def select_engine(
    prefer_moonshine: bool = True,
    moonshine_factory: EngineFactory = MoonshineEngine,
    whisper_factory: EngineFactory = FasterWhisperEngine,
) -> Optional[TranscriberEngine]:
    """Build the best available engine: Moonshine, then faster-whisper, then None.

    Tries each candidate in order and validates it with a short synthetic
    transcribe -- lazy model download/load and the array-input contract both
    surface HERE, where the chain can still fall back (construction alone is
    not enough: moonshine resolves its model on the first call). The warm-up
    also primes the model so the first real phrase transcribes fast. ANY
    exception (missing package, model download failure, init or transcribe
    error) prints a one-line stderr notice and moves on. Returns None (with a
    final notice) when every candidate fails, so the caller can run without
    speech. The factories are injectable so the fallback logic is testable
    without importing the real ASR stacks.
    """
    candidates: List[Tuple[str, EngineFactory]] = []
    if prefer_moonshine:
        candidates.append(("Moonshine", moonshine_factory))
    candidates.append(("faster-whisper", whisper_factory))

    for name, factory in candidates:
        try:
            engine = factory()
            engine.transcribe(np.zeros(_WARMUP_SAMPLES, dtype=np.float32))
        except Exception as exc:  # noqa: BLE001 - degrade to the next candidate
            print(
                f"Warning: {name} speech engine unavailable ({exc}); trying the next.",
                file=sys.stderr,
            )
            continue
        print(f"[speech] using {name} engine.")
        return engine

    print(
        "Warning: no speech engine available; speech transcription disabled.",
        file=sys.stderr,
    )
    return None


# --------------------------------------------------------------------------- #
# Microphone stream (sounddevice imported lazily)
# --------------------------------------------------------------------------- #

class MicStream:
    """Blocking 16 kHz mono float32 mic source, one block per ``read``.

    Wraps a ``sounddevice.InputStream`` (imported lazily). ``open`` may raise
    ``SpeechError`` when no input device / PortAudio is available; the worker
    turns that into a clean exit. ``read`` blocks ~``BLOCK_SECONDS`` for the next
    block (which paces the capture loop) and returns a fresh mono float32 array.
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE, block_frames: int = BLOCK_FRAMES) -> None:
        self._sample_rate = sample_rate
        self._block_frames = block_frames
        self._stream = None

    def open(self) -> "MicStream":
        try:
            import sounddevice  # deferred: optional dep

            self._stream = sounddevice.InputStream(
                samplerate=self._sample_rate,
                channels=1,
                dtype="float32",
                blocksize=self._block_frames,
            )
            self._stream.start()
        except Exception as exc:  # ImportError, PortAudioError, no device, ...
            raise SpeechError(f"microphone unavailable ({exc})") from exc
        return self

    def read(self) -> Optional[np.ndarray]:
        """Block for the next mono block, or None if the stream isn't open."""
        if self._stream is None:
            return None
        data, _overflowed = self._stream.read(self._block_frames)
        return np.asarray(data, dtype=np.float32).reshape(-1).copy()

    def close(self) -> None:
        """Stop + close the stream (idempotent, safe before open)."""
        stream = self._stream
        self._stream = None
        if stream is not None:
            stream.stop()
            stream.close()


# --------------------------------------------------------------------------- #
# SpeechWorker: mic -> segmenter -> engine -> published transcript + callback
# --------------------------------------------------------------------------- #

PhraseCallback = Callable[[str], None]


class SpeechWorker:
    """Run mic capture + segmentation + transcription on a dedicated thread.

    Owns ``mic`` and ``engine`` (single-owner rule -- nothing else touches them).
    ``latest()`` returns the most recent published ``Transcript`` (never blocks).
    ``on_phrase`` (if given) is invoked once per finalized phrase ON THIS THREAD.
    """

    def __init__(
        self,
        engine: TranscriberEngine,
        mic: MicStream,
        on_phrase: Optional[PhraseCallback] = None,
        segmenter: Optional[PhraseSegmenter] = None,
        transcript_size: int = ROLLING_TRANSCRIPT_SIZE,
    ) -> None:
        self._engine = engine
        self._mic = mic
        self._on_phrase = on_phrase
        self._segmenter = segmenter if segmenter is not None else PhraseSegmenter()
        self._transcript_size = transcript_size
        self._transcript = Transcript()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.error: Optional[Exception] = None

    def start(self) -> "SpeechWorker":
        """Start the speech thread (daemon)."""
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="speech-thread", daemon=True
        )
        self._thread.start()
        return self

    def latest(self) -> Transcript:
        """Return the most recently published transcript snapshot (never blocks)."""
        return self._transcript

    def poll_once(self) -> bool:
        """Read one block, feed the segmenter, transcribe+publish a finalized phrase.

        Returns True iff a block was read (False when the mic has none yet).
        Exposed for deterministic tests -- the thread loop is a thin wrapper.
        """
        block = self._mic.read()
        if block is None:
            return False
        finalized = self._segmenter.push(block)
        if finalized is not None:
            self._emit(finalized)
        return True

    def _emit(self, audio: np.ndarray) -> None:
        """Transcribe a finalized phrase, publish it, and fire the callback.

        Empty/blank transcriptions are dropped (no phrase, no callback). The new
        transcript is published with one atomic reference store; ``on_phrase``
        runs here, ON THE SPEECH THREAD.
        """
        text = self._engine.transcribe(audio).strip()
        if not text:
            return
        self._transcript = Transcript(  # single atomic publish (latest-wins)
            append_phrase(self._transcript.phrases, text, self._transcript_size)
        )
        if self._on_phrase is not None:
            try:
                self._on_phrase(text)
            except Exception as exc:  # noqa: BLE001 - a bad consumer costs one phrase
                print(
                    f"Warning: speech phrase callback raised ({exc}); "
                    "transcription continues.",
                    file=sys.stderr,
                )

    def _run(self) -> None:
        """Speech loop (own thread): open mic, poll until stopped, close mic.

        A mic-open failure prints one stderr notice and exits cleanly. A per-block
        exception (transcribe/segment) is recorded, announced once, and stops the
        loop -- the app keeps running without speech.
        """
        try:
            self._mic.open()
        except SpeechError as exc:
            print(
                f"Warning: {exc}; speech transcription disabled.", file=sys.stderr
            )
            return
        try:
            while not self._stop.is_set():
                try:
                    processed = self.poll_once()
                except Exception as exc:  # noqa: BLE001 - keep the app alive
                    self._on_crash(exc)
                    return
                if not processed:
                    time.sleep(_IDLE_SLEEP_SECONDS)
        finally:
            try:
                self._mic.close()
            except Exception as exc:  # noqa: BLE001 - closing is best-effort
                print(f"Warning: mic close failed ({exc}).", file=sys.stderr)

    def _on_crash(self, exc: Exception) -> None:
        """Record + announce a speech-loop crash once (the loop then stops)."""
        self.error = exc
        print(
            f"Error: speech transcription stopped after an exception ({exc}); "
            "the app keeps running without speech.",
            file=sys.stderr,
        )

    def stop(self) -> bool:
        """Signal the thread to stop and join it with a timeout. Idempotent.

        Returns True when the thread exited (or never ran), False when the join
        timed out (a transcribe was wedged past the timeout) -- the daemon thread
        is reclaimed at process exit. Never hangs app shutdown.
        """
        self._stop.set()
        thread = self._thread
        self._thread = None
        if thread is None:
            return True
        thread.join(timeout=_JOIN_TIMEOUT_SECONDS)
        return not thread.is_alive()


# --------------------------------------------------------------------------- #
# Launch glue: build a worker (or None), and a session context manager
# --------------------------------------------------------------------------- #

def default_on_phrase(text: str) -> None:
    """Default phrase callback: print the live transcript line to the console."""
    print(f"[speech] {text}")


def build_speech_worker(
    on_phrase: Optional[PhraseCallback] = None,
    engine: Optional[TranscriberEngine] = None,
    mic: Optional[MicStream] = None,
) -> Optional[SpeechWorker]:
    """Assemble a SpeechWorker, selecting an engine if one isn't supplied.

    Returns None when no engine is available (so the caller runs without speech).
    ``engine``/``mic`` are injectable for tests; production passes neither.
    """
    engine = engine if engine is not None else select_engine()
    if engine is None:
        return None
    return SpeechWorker(engine, mic if mic is not None else MicStream(), on_phrase=on_phrase)


@contextmanager
def speech_session(
    enabled: bool, on_phrase: Optional[PhraseCallback] = None
) -> Iterator[Optional[SpeechWorker]]:
    """Start the speech worker for the ``with`` body, stopping it cleanly on exit.

    A no-op (yields None) when ``enabled`` is False or no engine could be built,
    so ``run()`` can bracket the main loop unconditionally. The default phrase
    callback prints ``[speech] <text>`` to the console. The worker's ``stop`` is
    bounded so exiting the block never hangs shutdown.
    """
    if not enabled:
        yield None
        return
    worker = build_speech_worker(on_phrase=on_phrase or default_on_phrase)
    if worker is None:
        yield None
        return
    worker.start()
    try:
        yield worker
    finally:
        worker.stop()

"""Local-LLM request parser: turn a spoken phrase into a structured AR command.

Phase 13. The speech layer (``src.speech``) finalizes phrases off the main loop;
this module takes ONE such phrase and asks a small local LLM (Ollama, model
``qwen2.5:1.5b`` by default) to classify it into a strict, tiny JSON command:

    {"action": "insert"|"none",   # is the speaker asking to bring a visual in?
     "query": "<clean search text>",
     "route": "retrieve"|"generate",  # a real nameable thing vs an imagined one
     "source": "wikimedia"|"websearch"|"generator"}

Design + safety (the app's "never trust external data, degrade never die" rule):
  - The LLM only CLASSIFIES. Every field it returns is re-validated against a
    fixed enum in pure code (``parse_llm_json`` is the security boundary): an
    unknown/missing ``route`` falls back to "retrieve", an unknown ``source`` to
    "wikimedia", and the final ``source`` is *derived* deterministically from the
    route (``normalize_source``) so Phase 14/16 can rely on it. Any malformed or
    non-actionable output yields ``None`` (ignored), never an exception.
  - The model is prompted to emit JSON ONLY and the Ollama call requests
    ``format="json"`` with a low temperature and short output to keep latency
    down. A connection failure raises a typed ``ParserError`` that
    ``RequestParser.parse`` swallows into ``None`` + one stderr line -- Ollama
    being down (or unpulled) must never crash the app.
  - No network dependency is added: the client speaks Ollama's HTTP API over the
    stdlib ``urllib.request``. The ``ollama`` server + a pulled 1-3B model are an
    OPTIONAL runtime prerequisite (like Moonshine for speech); without them the
    parser simply degrades to "ignore every phrase".

THREADING: ``RequestParser.parse`` is called from the SPEECH thread's
``on_phrase`` callback (see ``src.launch.build_speech_on_phrase``), never the
render loop. The LLM call BLOCKS for a few hundred ms; that is fine on the speech
thread, and the bounded timeout here plus ``SpeechWorker.stop``'s bounded join
guarantee a wedged call can't hang shutdown.
"""

import json
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable, Optional, Protocol

# --------------------------------------------------------------------------- #
# Vocabulary: the ONLY values any field may take (validated against these).
# --------------------------------------------------------------------------- #

ACTION_INSERT = "insert"
ACTION_NONE = "none"

ROUTE_RETRIEVE = "retrieve"  # a real, nameable thing that can be looked up
ROUTE_GENERATE = "generate"  # an imagined/described thing that must be made

SOURCE_WIKIMEDIA = "wikimedia"
SOURCE_WEBSEARCH = "websearch"
SOURCE_GENERATOR = "generator"

# The two sources a "retrieve" request may resolve to (wikimedia is the default).
_RETRIEVE_SOURCES = (SOURCE_WIKIMEDIA, SOURCE_WEBSEARCH)

# Hard caps at the untrusted-model-output boundary. The model's reply should be a
# tiny JSON object; anything larger is noise (or an adversarial blob), so we cap
# the raw text we bother scanning and the resulting query length before it ever
# reaches Phase 14's fetch URL. Kept generous but bounded.
MAX_LLM_OUTPUT_CHARS = 20_000
MAX_QUERY_CHARS = 200

# Ollama defaults. The model must already be pulled (`ollama pull qwen2.5:1.5b`).
DEFAULT_MODEL = "qwen2.5:1.5b"
DEFAULT_HOST = "http://localhost:11434"
# Bounded so a slow/hung server can't wedge the speech thread past shutdown.
DEFAULT_TIMEOUT_SECONDS = 10.0
# Deterministic, low-latency decoding for a tiny structured output.
DEFAULT_TEMPERATURE = 0.0


class ParserError(RuntimeError):
    """Raised for an LLM/transport failure the caller degrades on (never crashes)."""


# --------------------------------------------------------------------------- #
# ParsedRequest: the immutable structured command (what the app consumes)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ParsedRequest:
    """Immutable, validated request parsed from one spoken phrase.

    ``action`` is "insert" or "none"; ``query`` is the cleaned search text;
    ``route`` is "retrieve" (a real thing) or "generate" (an imagined one);
    ``source`` is the resolved fetch source ("wikimedia"/"websearch"/"generator").
    ``raw_phrase`` keeps the original speech for debugging/logging.
    """

    action: str
    query: str
    route: str
    source: str
    raw_phrase: str = ""

    @property
    def is_actionable(self) -> bool:
        """True iff this is an insert with a non-empty query (worth acting on)."""
        return self.action == ACTION_INSERT and bool(self.query.strip())

    def __str__(self) -> str:
        # Matches the plan's done-when log shape (raw_phrase omitted as noise):
        # {action: insert, query: "CFTR protein structure", route: retrieve, source: wikimedia}
        return (
            f'{{action: {self.action}, query: "{self.query}", '
            f"route: {self.route}, source: {self.source}}}"
        )


# --------------------------------------------------------------------------- #
# Deterministic routing/source normalization (PURE -- the Phase 14/16 contract)
# --------------------------------------------------------------------------- #

def normalize_route(route: object) -> str:
    """Coerce any value to a valid route; unknown -> "retrieve" (the safe default)."""
    candidate = str(route).strip().lower()
    return candidate if candidate in (ROUTE_RETRIEVE, ROUTE_GENERATE) else ROUTE_RETRIEVE


def normalize_source(route: str, source: object) -> str:
    """Resolve the fetch source from the (already-normalized) route + model hint.

    The deterministic routing rule Phase 14/16 relies on:
      - ``generate`` -> always "generator" (the model's source hint is ignored).
      - ``retrieve`` -> the model's hint if it's a known retrieve source
        ("wikimedia"/"websearch"), else "wikimedia" (the default).
    """
    if route == ROUTE_GENERATE:
        return SOURCE_GENERATOR
    candidate = str(source).strip().lower()
    return candidate if candidate in _RETRIEVE_SOURCES else SOURCE_WIKIMEDIA


# --------------------------------------------------------------------------- #
# Safe JSON extraction + validation (THE security boundary -- trust nothing)
# --------------------------------------------------------------------------- #

def _extract_json_block(text: str) -> Optional[str]:
    """Return the first balanced ``{...}`` substring, or None if there isn't one.

    Robust to the usual model noise: leading prose, trailing prose, and Markdown
    code fences (```json ... ```) all fall out for free because we scan from the
    first ``{`` and match braces, skipping any inside JSON strings. Never raises.
    """
    if not isinstance(text, str):
        return None
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def parse_llm_json(text: str, raw_phrase: str = "") -> Optional[ParsedRequest]:
    """Safely parse the model's raw output into a validated ``ParsedRequest``.

    THE security boundary: model output is never trusted. Extract the first JSON
    object, ``json.loads`` it, and validate every field against the fixed enums,
    applying safe defaults (route -> "retrieve", source -> "wikimedia", derived
    from the route). Returns ``None`` -- never raises -- for anything malformed,
    a non-``insert`` action, or an empty query (i.e. non-actionable speech).
    """
    if not isinstance(text, str) or len(text) > MAX_LLM_OUTPUT_CHARS:
        return None  # oversized/adversarial blob -> ignore before scanning
    block = _extract_json_block(text)
    if block is None:
        return None
    try:
        data = json.loads(block)
    except (ValueError, RecursionError):
        # ValueError: malformed JSON. RecursionError: a deeply nested JSON bomb
        # (json.loads recurses) -- both are just untrusted noise we ignore.
        return None
    if not isinstance(data, dict):
        return None

    action = str(data.get("action", ACTION_NONE)).strip().lower()
    if action != ACTION_INSERT:
        return None  # non-actionable (chit-chat / explicit "none") -> ignore

    query = data.get("query", "")
    if not isinstance(query, str):
        return None
    query = query.strip()[:MAX_QUERY_CHARS]  # bound before it feeds a fetch URL
    if not query:
        return None  # insert with nothing to insert -> not actionable

    route = normalize_route(data.get("route", ROUTE_RETRIEVE))
    source = normalize_source(route, data.get("source", SOURCE_WIKIMEDIA))
    request = ParsedRequest(
        action=ACTION_INSERT,
        query=query,
        route=route,
        source=source,
        raw_phrase=raw_phrase,
    )
    return request if request.is_actionable else None


# --------------------------------------------------------------------------- #
# Prompt: strict JSON-only instruction + few-shots (kept short for latency)
# --------------------------------------------------------------------------- #

PROMPT_SYSTEM = (
    "You convert one short spoken phrase from a video call into a JSON command "
    "for an augmented-reality overlay tool. Decide whether the speaker wants to "
    "INSERT a visual (an image or object) into the video frame.\n"
    "Reply with ONE JSON object and NOTHING else -- no prose, no markdown, no "
    "code fences -- using EXACTLY these keys:\n"
    '  "action": "insert" if they want a visual brought into frame, else "none".\n'
    '  "query": a short clean search phrase for the thing to show (empty if none).\n'
    '  "route": "retrieve" for a real, nameable thing that can be found (a '
    "protein, a person, a place, a logo, a product); \"generate\" for an "
    "imagined or described thing (a friendly robot, a purple dragon).\n"
    '  "source": for retrieve use "wikimedia" (default) or "websearch"; for '
    'generate use "generator".\n'
    "Ignore chit-chat and filler: if they are not asking for a visual, action is "
    '"none".'
)

# (phrase, expected-json) few-shots: the CFTR retrieve case from the plan, a
# generate case, and a non-actionable chit-chat case. Kept to three for latency.
_FEW_SHOTS = (
    (
        "yeah we're looking at CFTR",
        {
            "action": "insert",
            "query": "CFTR protein structure",
            "route": "retrieve",
            "source": "wikimedia",
        },
    ),
    (
        "can you drop in a friendly robot mascot",
        {
            "action": "insert",
            "query": "friendly robot mascot",
            "route": "generate",
            "source": "generator",
        },
    ),
    (
        "so anyway how was your weekend",
        {"action": "none", "query": "", "route": "retrieve", "source": "wikimedia"},
    ),
)


def build_prompt(phrase: str) -> str:
    """Build the full JSON-only prompt (system rules + few-shots + the phrase)."""
    lines = [PROMPT_SYSTEM, "", "Examples:"]
    for example_phrase, example_json in _FEW_SHOTS:
        lines.append(f'Phrase: "{example_phrase}"')
        lines.append(f"JSON: {json.dumps(example_json)}")
    lines.append("")
    lines.append(f'Phrase: "{phrase}"')
    lines.append("JSON:")
    return "\n".join(lines)


PromptFn = Callable[[str], str]


# --------------------------------------------------------------------------- #
# LLM client: Protocol + an Ollama HTTP implementation (stdlib urllib, no dep)
# --------------------------------------------------------------------------- #

class LLMClient(Protocol):
    """A one-method text-completion client: prompt in, raw model text out."""

    def generate(self, prompt: str) -> str: ...


class OllamaClient:
    """Calls Ollama's ``/api/generate`` over HTTP (stdlib urllib -- no new dep).

    Non-streaming, ``format="json"`` to force a JSON body, low temperature, and a
    bounded timeout. Any transport/HTTP/decoding failure -- including Ollama not
    running or the model not pulled -- is turned into a ``ParserError`` so the
    caller can degrade to ignoring the phrase. Construction does no I/O (lazy):
    the first ``generate`` is the first network touch.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str = DEFAULT_HOST,
        temperature: float = DEFAULT_TEMPERATURE,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._model = model
        self._url = f"{host.rstrip('/')}/api/generate"
        self._temperature = temperature
        self._timeout = timeout

    def generate(self, prompt: str) -> str:
        body = json.dumps(
            {
                "model": self._model,
                "prompt": prompt,
                "stream": False,
                "format": "json",  # force a JSON object out of the endpoint
                "options": {"temperature": self._temperature},
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            self._url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, TimeoutError, ValueError) as exc:
            raise ParserError(f"Ollama request failed ({exc})") from exc
        text = payload.get("response")
        if not isinstance(text, str):
            raise ParserError("Ollama response missing a 'response' string")
        return text


# --------------------------------------------------------------------------- #
# RequestParser: prompt -> client.generate -> safe parse (the public entry)
# --------------------------------------------------------------------------- #

class RequestParser:
    """Tie a prompt + an ``LLMClient`` + safe parsing into one ``parse`` call.

    ``parse(phrase)`` returns a validated ``ParsedRequest`` for an actionable
    phrase, or ``None`` for non-actionable speech OR any failure (a ``ParserError``
    from the client, an unexpected client exception, or malformed model output),
    printing at most one stderr line on failure. It NEVER raises -- the speech
    thread that calls it must keep running.
    """

    def __init__(self, client: LLMClient, prompt_fn: PromptFn = build_prompt) -> None:
        self._client = client
        self._prompt_fn = prompt_fn

    def parse(self, phrase: str) -> Optional[ParsedRequest]:
        phrase = (phrase or "").strip()
        if not phrase:
            return None
        prompt = self._prompt_fn(phrase)
        try:
            text = self._client.generate(prompt)
            # parse_llm_json is inside the try too: it must never raise, but a
            # deep-nesting RecursionError is caught here as belt-and-suspenders.
            return parse_llm_json(text, raw_phrase=phrase)
        except ParserError as exc:
            print(f"Warning: LLM parse unavailable ({exc}); ignoring phrase.", file=sys.stderr)
            return None
        except Exception as exc:  # noqa: BLE001 - a bad client can't crash the app
            print(f"Warning: LLM parse error ({exc}); ignoring phrase.", file=sys.stderr)
            return None


def build_request_parser(model: str = DEFAULT_MODEL, host: str = DEFAULT_HOST) -> RequestParser:
    """Assemble a production ``RequestParser`` backed by a real ``OllamaClient``."""
    return RequestParser(OllamaClient(model=model, host=host))

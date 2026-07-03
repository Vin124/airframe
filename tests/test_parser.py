"""Unit tests for src.parser: safe JSON parsing, routing, and the RequestParser.

All pure/faked -- NO real Ollama server or network. ``parse_llm_json`` and the
normalization helpers are exercised as plain functions; ``RequestParser`` is
driven through a fake ``LLMClient`` (happy path / raises / garbage). This is the
security-boundary suite: it proves the parser never trusts model output and never
raises out to the speech thread that calls it.
"""

import dataclasses

import pytest

from src.parser import (
    ACTION_INSERT,
    MAX_LLM_OUTPUT_CHARS,
    MAX_QUERY_CHARS,
    ROUTE_GENERATE,
    ROUTE_RETRIEVE,
    SOURCE_GENERATOR,
    SOURCE_WEBSEARCH,
    SOURCE_WIKIMEDIA,
    OllamaClient,
    ParsedRequest,
    ParserError,
    RequestParser,
    build_prompt,
    build_request_parser,
    normalize_route,
    normalize_source,
    parse_llm_json,
)


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #

class _FakeClient:
    """Stand-in LLMClient: returns canned text, or raises the given exception."""

    def __init__(self, text: str = "", error: Exception = None) -> None:
        self._text = text
        self._error = error
        self.calls = 0

    def generate(self, prompt: str) -> str:
        self.calls += 1
        if self._error is not None:
            raise self._error
        return self._text


_CFTR_JSON = (
    '{"action": "insert", "query": "CFTR protein structure", '
    '"route": "retrieve", "source": "wikimedia"}'
)


# --------------------------------------------------------------------------- #
# ParsedRequest: immutability, is_actionable, log shape
# --------------------------------------------------------------------------- #

def test_parsed_request_is_frozen():
    req = ParsedRequest("insert", "CFTR", "retrieve", "wikimedia")
    with pytest.raises(dataclasses.FrozenInstanceError):
        req.query = "other"  # type: ignore[misc]


def test_is_actionable_true_for_insert_with_query():
    assert ParsedRequest("insert", "CFTR", "retrieve", "wikimedia").is_actionable is True


def test_is_actionable_false_for_none_action():
    assert ParsedRequest("none", "CFTR", "retrieve", "wikimedia").is_actionable is False


def test_is_actionable_false_for_blank_query():
    assert ParsedRequest("insert", "   ", "retrieve", "wikimedia").is_actionable is False


def test_str_matches_log_shape():
    req = ParsedRequest("insert", "CFTR protein structure", "retrieve", "wikimedia", raw_phrase="x")
    assert str(req) == (
        '{action: insert, query: "CFTR protein structure", '
        "route: retrieve, source: wikimedia}"
    )
    assert "raw_phrase" not in str(req)  # raw phrase is debug-only, not logged


# --------------------------------------------------------------------------- #
# normalize_route / normalize_source: the deterministic routing contract
# --------------------------------------------------------------------------- #

def test_normalize_route_passes_valid_values():
    assert normalize_route("retrieve") == ROUTE_RETRIEVE
    assert normalize_route("generate") == ROUTE_GENERATE


def test_normalize_route_is_case_insensitive_and_trims():
    assert normalize_route("  GENERATE ") == ROUTE_GENERATE


def test_normalize_route_defaults_unknown_to_retrieve():
    assert normalize_route("teleport") == ROUTE_RETRIEVE
    assert normalize_route(None) == ROUTE_RETRIEVE


def test_normalize_source_generate_is_always_generator():
    # The model's source hint is ignored for generate -> always the generator.
    assert normalize_source(ROUTE_GENERATE, "wikimedia") == SOURCE_GENERATOR
    assert normalize_source(ROUTE_GENERATE, "anything") == SOURCE_GENERATOR


def test_normalize_source_retrieve_keeps_known_source():
    assert normalize_source(ROUTE_RETRIEVE, "wikimedia") == SOURCE_WIKIMEDIA
    assert normalize_source(ROUTE_RETRIEVE, "websearch") == SOURCE_WEBSEARCH


def test_normalize_source_retrieve_defaults_unknown_to_wikimedia():
    assert normalize_source(ROUTE_RETRIEVE, "reddit") == SOURCE_WIKIMEDIA
    assert normalize_source(ROUTE_RETRIEVE, None) == SOURCE_WIKIMEDIA


# --------------------------------------------------------------------------- #
# parse_llm_json: the security boundary (trust nothing, never raise)
# --------------------------------------------------------------------------- #

def test_parse_clean_json():
    req = parse_llm_json(_CFTR_JSON, raw_phrase="yeah we're looking at CFTR")
    assert req is not None
    assert req.action == ACTION_INSERT
    assert req.query == "CFTR protein structure"
    assert req.route == ROUTE_RETRIEVE
    assert req.source == SOURCE_WIKIMEDIA
    assert req.raw_phrase == "yeah we're looking at CFTR"


def test_parse_json_in_code_fences():
    fenced = "```json\n" + _CFTR_JSON + "\n```"
    req = parse_llm_json(fenced)
    assert req is not None
    assert req.query == "CFTR protein structure"


def test_parse_json_with_leading_prose():
    noisy = "Sure! Here is the JSON you asked for:\n" + _CFTR_JSON + "\nHope that helps."
    req = parse_llm_json(noisy)
    assert req is not None
    assert req.route == ROUTE_RETRIEVE


def test_parse_missing_keys_applies_defaults():
    # Only action + query present: route/source default to retrieve/wikimedia.
    req = parse_llm_json('{"action": "insert", "query": "Eiffel Tower"}')
    assert req is not None
    assert req.route == ROUTE_RETRIEVE
    assert req.source == SOURCE_WIKIMEDIA


def test_parse_invalid_enums_are_normalized():
    req = parse_llm_json(
        '{"action": "insert", "query": "X", "route": "teleport", "source": "reddit"}'
    )
    assert req is not None
    assert req.route == ROUTE_RETRIEVE  # unknown route -> retrieve
    assert req.source == SOURCE_WIKIMEDIA  # unknown source -> wikimedia


def test_parse_generate_route_forces_generator_source():
    req = parse_llm_json(
        '{"action": "insert", "query": "friendly robot", '
        '"route": "generate", "source": "wikimedia"}'
    )
    assert req is not None
    assert req.route == ROUTE_GENERATE
    assert req.source == SOURCE_GENERATOR  # derived, model hint overridden


def test_parse_non_actionable_none_returns_none():
    assert parse_llm_json('{"action": "none", "query": "", "route": "retrieve"}') is None


def test_parse_insert_with_empty_query_returns_none():
    assert parse_llm_json('{"action": "insert", "query": "   "}') is None


def test_parse_malformed_text_returns_none():
    assert parse_llm_json("this is not json at all") is None
    assert parse_llm_json("") is None
    assert parse_llm_json("{ broken json ") is None


def test_parse_non_object_json_returns_none():
    # Valid JSON, but a list -> not a command object.
    assert parse_llm_json('["insert", "CFTR"]') is None


def test_parse_non_string_query_returns_none():
    assert parse_llm_json('{"action": "insert", "query": 42}') is None


def test_parse_action_is_case_insensitive():
    req = parse_llm_json('{"action": "INSERT", "query": "CFTR"}')
    assert req is not None and req.action == ACTION_INSERT


def test_parse_deep_nesting_json_bomb_returns_none_without_raising():
    # A RecursionError from json.loads on a deeply nested blob must be caught at
    # the boundary, not escape the "never raises" contract. Brace-wrapped (so the
    # extractor grabs it) and under MAX_LLM_OUTPUT_CHARS (so the length guard
    # doesn't short-circuit it first) -- this genuinely reaches json.loads.
    bomb = '{"a":' + "[" * 3000 + "]" * 3000 + "}"
    assert len(bomb) < MAX_LLM_OUTPUT_CHARS
    assert parse_llm_json(bomb) is None


def test_parse_oversized_output_returns_none():
    assert parse_llm_json("{" + "x" * (MAX_LLM_OUTPUT_CHARS + 1)) is None


def test_parse_clamps_overlong_query():
    long_query = "a" * (MAX_QUERY_CHARS + 500)
    req = parse_llm_json('{"action": "insert", "query": "' + long_query + '"}')
    assert req is not None
    assert len(req.query) == MAX_QUERY_CHARS


def test_request_parser_survives_json_bomb(capsys):
    # End-to-end: a client returning a JSON bomb yields None, no exception.
    bomb = '{"a":' + "[" * 3000 + "]" * 3000 + "}"
    parser = RequestParser(_FakeClient(text=bomb))
    assert parser.parse("show CFTR") is None


# --------------------------------------------------------------------------- #
# RequestParser: happy path / client raises / garbage -> None (never raises)
# --------------------------------------------------------------------------- #

def test_request_parser_happy_path():
    parser = RequestParser(_FakeClient(text=_CFTR_JSON))
    req = parser.parse("yeah we're looking at CFTR")
    assert req is not None
    assert req.query == "CFTR protein structure"
    assert req.raw_phrase == "yeah we're looking at CFTR"


def test_request_parser_client_error_returns_none_with_notice(capsys):
    parser = RequestParser(_FakeClient(error=ParserError("connection refused")))
    assert parser.parse("show CFTR") is None
    assert "LLM parse unavailable" in capsys.readouterr().err


def test_request_parser_unexpected_client_error_returns_none_with_notice(capsys):
    parser = RequestParser(_FakeClient(error=RuntimeError("boom")))
    assert parser.parse("show CFTR") is None
    assert "LLM parse error" in capsys.readouterr().err


def test_request_parser_garbage_output_returns_none():
    parser = RequestParser(_FakeClient(text="I cannot help with that."))
    assert parser.parse("show CFTR") is None


def test_request_parser_non_actionable_returns_none():
    chit_chat = '{"action": "none", "query": "", "route": "retrieve", "source": "wikimedia"}'
    parser = RequestParser(_FakeClient(text=chit_chat))
    assert parser.parse("how was your weekend") is None


def test_request_parser_blank_phrase_short_circuits():
    client = _FakeClient(text=_CFTR_JSON)
    parser = RequestParser(client)
    assert parser.parse("   ") is None
    assert client.calls == 0  # never bothered the model


def test_request_parser_uses_injected_prompt_fn():
    seen = {}

    def fake_prompt(phrase: str) -> str:
        seen["phrase"] = phrase
        return "PROMPT"

    client = _FakeClient(text=_CFTR_JSON)
    RequestParser(client, prompt_fn=fake_prompt).parse("hello CFTR")
    assert seen["phrase"] == "hello CFTR"


# --------------------------------------------------------------------------- #
# Prompt + factory smoke (no network)
# --------------------------------------------------------------------------- #

def test_build_prompt_contains_rules_examples_and_phrase():
    prompt = build_prompt("yeah we're looking at CFTR")
    assert "JSON" in prompt
    assert "CFTR protein structure" in prompt  # the CFTR few-shot
    assert "generator" in prompt  # the generate few-shot
    assert 'Phrase: "yeah we\'re looking at CFTR"' in prompt


def test_build_request_parser_returns_parser_over_ollama_client():
    parser = build_request_parser()
    assert isinstance(parser, RequestParser)
    assert isinstance(parser._client, OllamaClient)


# --------------------------------------------------------------------------- #
# OllamaClient.generate: transport + payload error mapping (monkeypatched urlopen)
# --------------------------------------------------------------------------- #

class _FakeResponse:
    """Context-manager stand-in for urlopen's response with a canned body."""

    def __init__(self, body: str) -> None:
        self._body = body.encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self._body


def test_ollama_client_maps_transport_error_to_parser_error(monkeypatch):
    import urllib.error

    def boom(*args, **kwargs):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    with pytest.raises(ParserError):
        OllamaClient().generate("prompt")


def test_ollama_client_maps_timeout_to_parser_error(monkeypatch):
    def boom(*args, **kwargs):
        raise TimeoutError("timed out")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    with pytest.raises(ParserError):
        OllamaClient().generate("prompt")


def test_ollama_client_missing_response_field_raises_parser_error(monkeypatch):
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *a, **k: _FakeResponse('{"done": true}')
    )
    with pytest.raises(ParserError):
        OllamaClient().generate("prompt")


def test_ollama_client_returns_response_text(monkeypatch):
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *a, **k: _FakeResponse('{"response": "' + _CFTR_JSON.replace('"', '\\"') + '"}'),
    )
    text = OllamaClient().generate("prompt")
    assert "CFTR protein structure" in text

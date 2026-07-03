"""Unit tests for src.retriever: Wikimedia parsing, URL/decode safety, composition.

All pure/faked -- NO real Wikimedia, DuckDuckGo, or network. The best-image
selector runs against synthetic MediaWiki JSON; the decode pipeline runs against
in-memory ``cv2.imencode`` bytes; the source composition runs against injected
fake sources; the network boundary (``_fetch_json`` / ``_download_bytes`` /
ddgs) is monkeypatched where unavoidable. This is the security-boundary suite for
Phase 14: it proves untrusted remote content is validated and that
``Retriever.retrieve`` never raises to the speech thread that calls it.
"""

import dataclasses
import urllib.error

import cv2
import numpy as np
import pytest

from src.parser import (
    ROUTE_GENERATE,
    ROUTE_RETRIEVE,
    SOURCE_GENERATOR,
    SOURCE_WEBSEARCH,
    SOURCE_WIKIMEDIA,
    ParsedRequest,
)
from src.retriever import (
    MAX_CACHE_FILES,
    MAX_DECODE_PIXELS,
    MAX_DOWNLOAD_BYTES,
    MAX_IMAGE_DIMENSION,
    LocalLibrarySource,
    Retriever,
    RetrievedAsset,
    RetrieverError,
    WebSearchSource,
    WikimediaImage,
    WikimediaSource,
    _download_bytes,
    _prune_cache,
    _slug,
    _ValidatingRedirectHandler,
    build_retriever,
    decode_to_bgra,
    is_allowed_url,
    match_filename,
    probe_dimensions,
    select_best_image,
)


# --------------------------------------------------------------------------- #
# Helpers: synthetic images, requests, fake sources
# --------------------------------------------------------------------------- #

def _png_bytes(height: int = 8, width: int = 8, channels: int = 3) -> bytes:
    """Encode a small solid image to PNG bytes (3 or 4 channel)."""
    arr = np.zeros((height, width, channels), dtype=np.uint8)
    arr[..., :3] = 127
    if channels == 4:
        arr[..., 3] = 255
    ok, buffer = cv2.imencode(".png", arr)
    assert ok
    return buffer.tobytes()


def _req(query: str = "CFTR", source: str = SOURCE_WIKIMEDIA, route: str = ROUTE_RETRIEVE):
    return ParsedRequest(action="insert", query=query, route=route, source=source)


def _asset(name: str) -> RetrievedAsset:
    return RetrievedAsset(bgra=np.zeros((2, 2, 4), np.uint8), source_name=name, query="q")


class _FakeRequest:
    """Minimal stand-in for a urllib Request the redirect handler can inspect."""

    def __init__(self, url: str = "https://example.com/start") -> None:
        self.full_url = url
        self.headers = {}
        self.unverifiable = False
        self.origin_req_host = "example.com"

    def get_full_url(self):
        return self.full_url

    def has_header(self, name):
        return False

    def get_method(self):
        return "GET"


class _FakeSource:
    """Injectable RetrievalSource: returns a canned asset/None or raises; counts calls."""

    def __init__(self, name: str, result=None, error: Exception = None) -> None:
        self.name = name
        self._result = result
        self._error = error
        self.calls = 0

    def fetch(self, query: str):
        self.calls += 1
        if self._error is not None:
            raise self._error
        return self._result


def _page(pageid, index, mime, url, title="File:X.png", extmeta=None, thumburl=None):
    info = {"mime": mime, "url": url, "width": 800, "height": 600}
    if thumburl is not None:
        info["thumburl"] = thumburl
    if extmeta is not None:
        info["extmetadata"] = extmeta
    return {"pageid": pageid, "ns": 6, "title": title, "index": index, "imageinfo": [info]}


def _payload(*pages):
    return {"batchcomplete": "", "query": {"pages": {str(p["pageid"]): p for p in pages}}}


# --------------------------------------------------------------------------- #
# RetrievedAsset: immutability + defaults
# --------------------------------------------------------------------------- #

def test_retrieved_asset_is_frozen():
    asset = _asset("wikimedia")
    with pytest.raises(dataclasses.FrozenInstanceError):
        asset.query = "other"  # type: ignore[misc]


def test_retrieved_asset_defaults():
    asset = RetrievedAsset(bgra=np.zeros((1, 1, 4), np.uint8), source_name="x", query="q")
    assert asset.attribution == ""
    assert asset.cache_path is None


def test_retriever_error_is_exception():
    assert issubclass(RetrieverError, RuntimeError)


# --------------------------------------------------------------------------- #
# is_allowed_url: only http/https with a host (the scheme gate)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("url", ["http://a.com/x.png", "https://commons.wikimedia.org/y.jpg"])
def test_is_allowed_url_accepts_web(url):
    assert is_allowed_url(url) is True


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "ftp://host/file",
        "data:image/png;base64,AAAA",
        "javascript:alert(1)",
        "http://",  # scheme ok but no host
        "",
        "not a url",
        None,
        42,
    ],
)
def test_is_allowed_url_rejects_non_web(url):
    assert is_allowed_url(url) is False


# --------------------------------------------------------------------------- #
# decode_to_bgra: bytes -> downscaled BGRA, alpha handling, garbage -> None
# --------------------------------------------------------------------------- #

def test_decode_bgr_gains_opaque_alpha():
    bgra = decode_to_bgra(_png_bytes(8, 8, channels=3))
    assert bgra is not None
    assert bgra.shape == (8, 8, 4)
    assert np.all(bgra[..., 3] == 255)  # opaque alpha added


def test_decode_grayscale_becomes_bgra():
    arr = np.full((6, 6), 100, dtype=np.uint8)  # 2-D grayscale
    ok, buffer = cv2.imencode(".png", arr)
    assert ok
    bgra = decode_to_bgra(buffer.tobytes())
    assert bgra is not None
    assert bgra.shape == (6, 6, 4)


def test_decode_preserves_existing_alpha():
    arr = np.zeros((4, 4, 4), dtype=np.uint8)
    arr[..., 3] = 10  # a distinctly non-opaque alpha
    arr[0, 0, 3] = 200
    ok, buffer = cv2.imencode(".png", arr)
    assert ok
    bgra = decode_to_bgra(buffer.tobytes())
    assert bgra is not None
    assert bgra.shape == (4, 4, 4)
    assert not np.all(bgra[..., 3] == 255)  # original alpha kept, not overwritten


def test_decode_downscales_over_max_dim():
    # 40x100 image with a tiny max_dim -> longest side clamped to max_dim, aspect kept.
    bgra = decode_to_bgra(_png_bytes(40, 100, channels=3), max_dim=20)
    assert bgra is not None
    assert max(bgra.shape[:2]) == 20
    assert bgra.shape == (8, 20, 4)  # 40*0.2=8, 100*0.2=20


def test_decode_no_downscale_under_max_dim():
    bgra = decode_to_bgra(_png_bytes(8, 8, channels=3), max_dim=MAX_IMAGE_DIMENSION)
    assert bgra is not None and bgra.shape == (8, 8, 4)


def test_decode_garbage_bytes_returns_none():
    assert decode_to_bgra(b"this is not an image at all") is None


def test_decode_empty_bytes_returns_none():
    assert decode_to_bgra(b"") is None


# --------------------------------------------------------------------------- #
# Security boundary: decode bomb, SSRF-via-redirect, byte cap, slug, cache prune
# --------------------------------------------------------------------------- #

def _png_with_forged_dims(width: int, height: int) -> bytes:
    """A valid-enough PNG header whose IHDR advertises (width, height)."""
    real = bytearray(_png_bytes(8, 8, 3))
    real[16:20] = int(width).to_bytes(4, "big")
    real[20:24] = int(height).to_bytes(4, "big")
    return bytes(real)


def test_probe_dimensions_reads_png():
    assert probe_dimensions(_png_bytes(12, 20, 3)) == (20, 12)


def test_probe_dimensions_reads_jpeg():
    arr = np.zeros((14, 22, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", arr)
    assert ok
    assert probe_dimensions(buf.tobytes()) == (22, 14)


def test_probe_dimensions_unknown_returns_none():
    assert probe_dimensions(b"not an image") is None


def test_decode_rejects_pixel_bomb_before_decoding(capsys):
    # A header advertising 30000x30000 (900 MP) must be rejected BEFORE imdecode
    # so it can't allocate gigabytes -- even though the byte blob is tiny.
    bomb = _png_with_forged_dims(30000, 30000)
    assert 30000 * 30000 > MAX_DECODE_PIXELS
    assert decode_to_bgra(bomb) is None
    assert "exceed the decode cap" in capsys.readouterr().err


def test_redirect_to_disallowed_scheme_is_blocked():
    handler = _ValidatingRedirectHandler()
    with pytest.raises(urllib.error.URLError):
        handler.redirect_request(
            _FakeRequest(), None, 302, "Found", {}, "ftp://internal-host/secret"
        )


def test_redirect_to_allowed_scheme_is_permitted():
    handler = _ValidatingRedirectHandler()
    # Delegates to the stdlib parent, which returns a Request for an http(s) target.
    result = handler.redirect_request(
        _FakeRequest(), None, 302, "Found", {}, "https://example.com/image.png"
    )
    assert result is not None


def test_download_bytes_streams_over_cap_without_content_length(monkeypatch):
    # A server that omits Content-Length and streams past the cap must still be
    # bounded by the post-read length check.
    oversized = b"x" * (MAX_DOWNLOAD_BYTES + 10)

    class _NoLenResponse:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def getheader(self, name):
            return None  # no Content-Length advertised

        def read(self, n):
            return oversized[:n]

    monkeypatch.setattr("src.retriever._open", lambda url, timeout: _NoLenResponse())
    assert _download_bytes("https://example.com/huge.png") is None


@pytest.mark.parametrize(
    "malicious",
    ["../../etc/passwd", r"C:\Windows\evil", "x\x00y", "..\\..\\secret"],
)
def test_slug_neutralizes_path_traversal(malicious):
    slug = _slug(malicious)
    for bad in ("/", "\\", "..", "\x00", ":"):
        assert bad not in slug


def test_prune_cache_keeps_only_newest(tmp_path):
    import os

    for i in range(MAX_CACHE_FILES + 10):
        p = tmp_path / f"f{i:03d}.img"
        p.write_bytes(b"x")
        os.utime(p, (i, i))  # ascending mtime so ordering is deterministic
    _prune_cache(str(tmp_path), keep=MAX_CACHE_FILES)
    remaining = list(tmp_path.iterdir())
    assert len(remaining) == MAX_CACHE_FILES
    # The oldest (f000..) were evicted; the newest survive.
    assert (tmp_path / "f000.img") not in remaining


# --------------------------------------------------------------------------- #
# select_best_image: the pure Wikimedia best-candidate selector
# --------------------------------------------------------------------------- #

def test_select_prefers_png_over_svg():
    # SVG is higher search relevance (lower index) but must be filtered out.
    payload = _payload(
        _page(1, index=1, mime="image/svg+xml", url="https://c/x.svg"),
        _page(2, index=2, mime="image/png", url="https://c/x.png"),
    )
    best = select_best_image(payload)
    assert best is not None
    assert best.mime == "image/png"
    assert best.url == "https://c/x.png"


def test_select_picks_lowest_search_index_among_same_mime():
    payload = _payload(
        _page(1, index=3, mime="image/jpeg", url="https://c/three.jpg"),
        _page(2, index=1, mime="image/jpeg", url="https://c/one.jpg"),
    )
    best = select_best_image(payload)
    assert best is not None and best.url == "https://c/one.jpg"  # index 1 wins


def test_select_prefers_preferred_mime_over_gif_even_if_worse_index():
    payload = _payload(
        _page(1, index=1, mime="image/gif", url="https://c/x.gif"),
        _page(2, index=5, mime="image/png", url="https://c/x.png"),
    )
    best = select_best_image(payload)
    assert best is not None and best.mime == "image/png"  # png rank beats gif's index


def test_select_prefers_thumburl_over_full_url():
    payload = _payload(
        _page(
            1, index=1, mime="image/jpeg",
            url="https://c/full.jpg", thumburl="https://c/thumb.jpg",
        )
    )
    best = select_best_image(payload)
    assert best is not None and best.url == "https://c/thumb.jpg"


def test_select_extracts_attribution_and_strips_html():
    extmeta = {
        "LicenseShortName": {"value": "CC BY-SA 4.0"},
        "Artist": {"value": '<a href="/wiki/User:Jane">Jane Doe</a>'},
    }
    payload = _payload(
        _page(1, index=1, mime="image/png", url="https://c/x.png",
              title="File:CFTR protein.png", extmeta=extmeta)
    )
    best = select_best_image(payload)
    assert best is not None
    assert "CFTR protein" in best.attribution  # title, extension + File: stripped
    assert "CC BY-SA 4.0" in best.attribution
    assert "Jane Doe" in best.attribution
    assert "<a" not in best.attribution  # HTML tags stripped


def test_select_empty_results_returns_none():
    assert select_best_image({"batchcomplete": "", "query": {"pages": {}}}) is None


def test_select_missing_query_returns_none():
    assert select_best_image({"batchcomplete": ""}) is None
    assert select_best_image(None) is None
    assert select_best_image("garbage") is None


def test_select_skips_pages_without_imageinfo():
    good = _page(2, index=2, mime="image/png", url="https://c/ok.png")
    no_info = {"pageid": 1, "ns": 6, "title": "File:x", "index": 1}  # no imageinfo
    payload = {"query": {"pages": {"1": no_info, "2": good}}}
    best = select_best_image(payload)
    assert best is not None and best.url == "https://c/ok.png"


def test_select_all_unsupported_types_returns_none():
    payload = _payload(
        _page(1, index=1, mime="image/svg+xml", url="https://c/x.svg"),
        _page(2, index=2, mime="application/pdf", url="https://c/x.pdf"),
    )
    assert select_best_image(payload) is None


def test_select_skips_non_web_url():
    payload = _payload(_page(1, index=1, mime="image/png", url="file:///x.png"))
    assert select_best_image(payload) is None


# --------------------------------------------------------------------------- #
# match_filename: simple token-substring matcher (CLIP deferred)
# --------------------------------------------------------------------------- #

def test_match_filename_substring_hit():
    files = ["cftr_protein.png", "dog.jpg", "sunset.png"]
    assert match_filename(files, "CFTR structure") == "cftr_protein.png"


def test_match_filename_prefers_most_tokens():
    files = ["cftr.png", "cftr_protein_structure.jpg"]
    assert match_filename(files, "cftr protein structure") == "cftr_protein_structure.jpg"


def test_match_filename_no_match_returns_none():
    assert match_filename(["dog.jpg", "cat.png"], "elephant") is None


def test_match_filename_empty_query_returns_none():
    assert match_filename(["cftr.png"], "   ") is None


# --------------------------------------------------------------------------- #
# LocalLibrarySource: filesystem substring matching against real files
# --------------------------------------------------------------------------- #

def test_local_library_returns_matching_asset(tmp_path):
    path = tmp_path / "cftr_map.png"
    cv2.imwrite(str(path), np.full((5, 5, 3), 200, np.uint8))
    source = LocalLibrarySource(str(tmp_path))
    asset = source.fetch("CFTR structure")
    assert asset is not None
    assert asset.source_name == "local"
    assert asset.bgra.shape[2] == 4
    assert asset.cache_path == str(path)
    assert "cftr_map.png" in asset.attribution


def test_local_library_no_filename_match_returns_none(tmp_path):
    cv2.imwrite(str(tmp_path / "dog.png"), np.zeros((4, 4, 3), np.uint8))
    assert LocalLibrarySource(str(tmp_path)).fetch("elephant") is None


def test_local_library_ignores_non_image_files(tmp_path):
    (tmp_path / "cftr_notes.txt").write_text("not an image")
    assert LocalLibrarySource(str(tmp_path)).fetch("cftr") is None


def test_local_library_none_folder_returns_none():
    assert LocalLibrarySource(None).fetch("cftr") is None


def test_local_library_missing_folder_returns_none(tmp_path):
    assert LocalLibrarySource(str(tmp_path / "nope")).fetch("cftr") is None


# --------------------------------------------------------------------------- #
# Retriever: source composition + fallback (injected fakes, never raises)
# --------------------------------------------------------------------------- #

def test_retrieve_primary_none_falls_back_to_websearch():
    wiki = _FakeSource("wikimedia", result=None)
    web = _FakeSource("websearch", result=_asset("websearch"))
    result = Retriever(wikimedia=wiki, websearch=web).retrieve(_req(source=SOURCE_WIKIMEDIA))
    assert result is not None and result.source_name == "websearch"
    assert wiki.calls == 1 and web.calls == 1  # primary tried, then fallback


def test_retrieve_primary_asset_skips_fallback():
    wiki = _FakeSource("wikimedia", result=_asset("wikimedia"))
    web = _FakeSource("websearch", result=_asset("websearch"))
    result = Retriever(wikimedia=wiki, websearch=web).retrieve(_req(source=SOURCE_WIKIMEDIA))
    assert result is not None and result.source_name == "wikimedia"
    assert web.calls == 0  # fallback never touched


def test_retrieve_websearch_source_tries_websearch_first():
    wiki = _FakeSource("wikimedia", result=_asset("wikimedia"))
    web = _FakeSource("websearch", result=_asset("websearch"))
    result = Retriever(wikimedia=wiki, websearch=web).retrieve(_req(source=SOURCE_WEBSEARCH))
    assert result is not None and result.source_name == "websearch"
    assert wiki.calls == 0  # websearch preferred, wiki not tried


def test_retrieve_local_tried_first_when_configured():
    local = _FakeSource("local", result=_asset("local"))
    wiki = _FakeSource("wikimedia", result=_asset("wikimedia"))
    web = _FakeSource("websearch", result=_asset("websearch"))
    result = Retriever(wikimedia=wiki, websearch=web, local=local).retrieve(_req())
    assert result is not None and result.source_name == "local"
    assert wiki.calls == 0 and web.calls == 0  # local hit short-circuits


def test_retrieve_all_sources_fail_returns_none():
    wiki = _FakeSource("wikimedia", result=None)
    web = _FakeSource("websearch", result=None)
    result = Retriever(wikimedia=wiki, websearch=web).retrieve(_req())
    assert result is None
    assert wiki.calls == 1 and web.calls == 1  # both attempted


def test_retrieve_non_retrieve_route_returns_none_without_fetching():
    wiki = _FakeSource("wikimedia", result=_asset("wikimedia"))
    web = _FakeSource("websearch", result=_asset("websearch"))
    request = ParsedRequest(action="insert", query="robot", route=ROUTE_GENERATE, source=SOURCE_GENERATOR)
    result = Retriever(wikimedia=wiki, websearch=web).retrieve(request)
    assert result is None
    assert wiki.calls == 0 and web.calls == 0  # generate isn't the retriever's job


def test_retrieve_none_request_returns_none():
    assert Retriever(wikimedia=_FakeSource("wikimedia"), websearch=_FakeSource("websearch")).retrieve(None) is None


def test_retrieve_blank_query_returns_none_without_fetching():
    wiki = _FakeSource("wikimedia", result=_asset("wikimedia"))
    result = Retriever(wikimedia=wiki, websearch=_FakeSource("websearch")).retrieve(_req(query="   "))
    assert result is None and wiki.calls == 0


def test_retrieve_source_error_is_caught_and_next_tried(capsys):
    wiki = _FakeSource("wikimedia", error=RuntimeError("boom"))
    web = _FakeSource("websearch", result=_asset("websearch"))
    result = Retriever(wikimedia=wiki, websearch=web).retrieve(_req(source=SOURCE_WIKIMEDIA))
    assert result is not None and result.source_name == "websearch"
    assert "errored" in capsys.readouterr().err


def test_retrieve_never_raises_on_bad_request(capsys):
    class _BoomRequest:
        @property
        def route(self):
            raise RuntimeError("attribute boom")

    result = Retriever(wikimedia=_FakeSource("w"), websearch=_FakeSource("s")).retrieve(_BoomRequest())
    assert result is None
    assert "retriever failed" in capsys.readouterr().err


def test_build_retriever_defaults_and_local():
    assert build_retriever().local is None
    with_local = build_retriever(local_folder="some/folder")
    assert isinstance(with_local.local, LocalLibrarySource)
    assert isinstance(with_local.wikimedia, WikimediaSource)
    assert isinstance(with_local.websearch, WebSearchSource)


# --------------------------------------------------------------------------- #
# WikimediaSource.fetch: end-to-end assembly (monkeypatched network boundary)
# --------------------------------------------------------------------------- #

def test_wikimedia_fetch_assembles_asset(monkeypatch):
    extmeta = {"LicenseShortName": {"value": "CC0"}}
    payload = _payload(
        _page(1, index=1, mime="image/png", url="https://c/x.png",
              title="File:CFTR.png", extmeta=extmeta)
    )
    monkeypatch.setattr("src.retriever._fetch_json", lambda *a, **k: payload)
    monkeypatch.setattr("src.retriever._download_bytes", lambda *a, **k: _png_bytes(10, 12))
    monkeypatch.setattr("src.retriever._save_to_cache", lambda *a, **k: "/tmp/shacam_cache/x.img")

    asset = WikimediaSource().fetch("CFTR")
    assert asset is not None
    assert asset.source_name == "wikimedia"
    assert asset.bgra.shape == (10, 12, 4)
    assert asset.cache_path == "/tmp/shacam_cache/x.img"
    assert "CC0" in asset.attribution


def test_wikimedia_fetch_none_when_no_image(monkeypatch):
    monkeypatch.setattr("src.retriever._fetch_json", lambda *a, **k: {"query": {"pages": {}}})
    called = []
    monkeypatch.setattr("src.retriever._download_bytes", lambda *a, **k: called.append(1))
    assert WikimediaSource().fetch("nothing here") is None
    assert called == []  # never tried to download


def test_wikimedia_fetch_none_when_download_fails(monkeypatch):
    payload = _payload(_page(1, index=1, mime="image/png", url="https://c/x.png"))
    monkeypatch.setattr("src.retriever._fetch_json", lambda *a, **k: payload)
    monkeypatch.setattr("src.retriever._download_bytes", lambda *a, **k: None)
    assert WikimediaSource().fetch("CFTR") is None


def test_wikimedia_query_url_encodes_query():
    url = WikimediaSource()._query_url("CFTR protein & stuff")
    assert "commons.wikimedia.org/w/api.php" in url
    assert "gsrsearch=CFTR+protein+%26+stuff" in url  # url-encoded
    assert "gsrnamespace=6" in url and "iiurlwidth=" in url


# --------------------------------------------------------------------------- #
# WebSearchSource.fetch: first decodable hit wins (monkeypatched download)
# --------------------------------------------------------------------------- #

def test_websearch_fetch_uses_first_decodable_url(monkeypatch):
    source = WebSearchSource()
    monkeypatch.setattr(source, "_search_urls", lambda q: ["https://a/img.png"])
    monkeypatch.setattr("src.retriever._download_bytes", lambda *a, **k: _png_bytes(6, 6))
    asset = source.fetch("CFTR")
    assert asset is not None and asset.source_name == "websearch"
    assert asset.bgra.shape == (6, 6, 4)


def test_websearch_fetch_skips_non_web_and_undecodable(monkeypatch):
    source = WebSearchSource()
    monkeypatch.setattr(source, "_search_urls", lambda q: ["file:///x.png", "https://a/img.png"])
    good = _png_bytes(5, 5)

    def fake_download(url, *a, **k):
        return good  # only reached for the allowed https url; file:// is filtered first

    monkeypatch.setattr("src.retriever._download_bytes", fake_download)
    asset = source.fetch("CFTR")
    assert asset is not None and asset.bgra.shape == (5, 5, 4)


def test_websearch_fetch_empty_search_returns_none(monkeypatch):
    source = WebSearchSource()
    monkeypatch.setattr(source, "_search_urls", lambda q: [])
    assert source.fetch("CFTR") is None

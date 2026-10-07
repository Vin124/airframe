"""Image retriever: turn a parsed ``retrieve`` request into an image IN HAND.

Phase 14. The parser (``src.parser``) classifies a spoken phrase into a
``ParsedRequest`` with ``route="retrieve"`` and a ``source`` hint
("wikimedia"/"websearch"). This module fetches the top matching image, downloads
it to a temp cache, decodes it to a BGRA array, and hands it back as a ready
``RetrievedAsset`` -- the same BGRA-overlay contract ``ArraySource`` /
``ImageSource`` expose, so Phase 15 can wrap it into a scene ``Layer`` unchanged.
This phase does NOT touch the scene or the render loop.

Sources behind one interface (``RetrievalSource.fetch``), swappable + composable:
  - ``WikimediaSource`` (default): the MediaWiki Commons API. Pure JSON parsing
    (``select_best_image``) picks the best licensed raster image and extracts an
    attribution string. stdlib ``urllib`` only -- NO new dependency.
  - ``WebSearchSource`` (fallback for misses): DuckDuckGo image search via the
    ``ddgs`` package, imported LAZILY so its absence only disables this fallback.
  - ``LocalLibrarySource`` (optional): a preloaded folder matched to the query by
    case-insensitive filename substring. (CLIP-embedding matching is deferred --
    see ``match_filename``.)

SECURITY -- remote URLs here come from SEARCH RESULTS and are UNTRUSTED. They are
only LIGHTLY validated for this debug harness: scheme is restricted to http/https
(``is_allowed_url`` rejects file://, ftp://, data:, ...), downloads are size- and
time-capped, and the decoded result must actually be an image (``cv2.imdecode``
returning ``None`` on garbage is treated as a no-result). No further SSRF
hardening (IP/host allow-listing) is done. The query is already length-bounded by
the parser (``MAX_QUERY_CHARS``); it is url-encoded before use.

FAILURE POLICY (the app's "degrade, never die" rule): every source returns
``None`` on a no-result/network error (plus one stderr notice) rather than
raising; ``Retriever.retrieve`` catches everything and NEVER raises to its caller.
A typed ``RetrieverError`` exists for genuinely unexpected failures but is not
raised on the normal network/no-result paths.

THREADING: ``Retriever.retrieve`` is called from the SPEECH thread's ``on_phrase``
callback (see ``src.launch.build_speech_on_phrase``), never the render loop. Its
HTTP calls BLOCK for up to ``REQUEST_TIMEOUT_SECONDS``; that is fine off the
render thread and can never stall render/tracking FPS.
"""

import html
import json
import os
import re
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence

import cv2
import numpy as np

from src.parser import ROUTE_RETRIEVE, ParsedRequest
from src.sources.image import ensure_bgra

# --------------------------------------------------------------------------- #
# Security / resource caps (constants -- the untrusted-remote-content boundary)
# --------------------------------------------------------------------------- #

# Only these URL schemes are ever fetched (rejects file://, ftp://, data:, ...).
ALLOWED_SCHEMES = ("http", "https")
# Hard cap on any single download; a search result claiming/streaming more is
# refused before it can exhaust memory/disk.
MAX_DOWNLOAD_BYTES = 20 * 1024 * 1024  # 20 MB
# Bounded so a slow/hung host can't wedge the speech thread.
REQUEST_TIMEOUT_SECONDS = 10.0
# Decoded images are downscaled so their longest side is at most this many
# pixels (also requested server-side from Wikimedia via iiurlwidth).
MAX_IMAGE_DIMENSION = 1024
# Hard cap on the DECODED pixel count, checked from the image header BEFORE
# cv2.imdecode allocates anything. The 20 MB compressed cap does NOT bound
# decode memory -- a small highly-compressed PNG/JPEG can decode to gigabytes
# (a "decompression bomb"). 40 MP (~160 MB BGRA) admits big real photos while
# refusing the 30000x30000-style bomb.
MAX_DECODE_PIXELS = 40_000_000
# Max HTTP redirects followed; each hop is re-validated (see the redirect handler).
MAX_REDIRECTS = 5
# Wikimedia requires a descriptive User-Agent on API + upload requests.
USER_AGENT = "airframe/0.1 (hand-tracking debug harness)"

# Wikimedia Commons MediaWiki API endpoint.
WIKIMEDIA_API_URL = "https://commons.wikimedia.org/w/api.php"
# How many search hits to ask each backend for (we pick the best of them).
DEFAULT_MAX_RESULTS = 8

# Raster mimetypes we accept; SVG/PDF/TIFF/DjVu/audio/video fall out. jpeg/png
# are preferred over the rest when ranking candidates.
_PREFERRED_MIMES = ("image/jpeg", "image/png")
_ALLOWED_MIMES = ("image/jpeg", "image/png", "image/gif", "image/webp")

# Temp cache directory name (under the OS temp dir) for downloaded originals.
CACHE_DIR_NAME = "airframe_cache"
# Image file extensions the local library will consider.
_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp")
# Attribution strings are bounded so a hostile extmetadata blob stays small.
_MAX_ATTRIBUTION_CHARS = 240
# Readable-name length cap for a cache filename slug.
_MAX_SLUG_CHARS = 40
# Keep at most this many downloaded originals in the temp cache (LRU by mtime).
MAX_CACHE_FILES = 64


class RetrieverError(RuntimeError):
    """Raised ONLY for genuinely unexpected failures.

    The normal network/no-result paths return ``None`` (plus a stderr notice)
    instead, so callers degrade gracefully; this exists for completeness and is
    swallowed by ``Retriever.retrieve`` if it ever surfaces.
    """


# --------------------------------------------------------------------------- #
# RetrievedAsset: the immutable, ready-to-show result (BGRA + provenance)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class RetrievedAsset:
    """An image fetched + decoded and ready to become a scene ``Layer``.

    ``bgra`` is an ``HxWx4`` uint8 BGRA array (the overlay-layer contract).
    ``source_name`` records which backend produced it ("wikimedia"/"websearch"/
    "local"); ``query`` is the search text used; ``attribution`` carries a
    title/license/author string when the backend supplies one (Wikimedia does);
    ``cache_path`` is the temp file the original bytes were saved to (or ``None``
    if the save failed -- the in-memory ``bgra`` is still valid). Immutable.
    """

    bgra: np.ndarray
    source_name: str
    query: str
    attribution: str = ""
    cache_path: Optional[str] = None


# --------------------------------------------------------------------------- #
# RetrievalSource: the one interface every backend implements
# --------------------------------------------------------------------------- #

class RetrievalSource(Protocol):
    """A swappable image backend: ``fetch(query)`` -> asset or ``None``.

    ``name`` labels the backend (stamped onto the ``RetrievedAsset``). ``fetch``
    returns ``None`` on a no-result or a handled network error -- it should
    PREFER that over raising so ``Retriever`` can fall through to the next source.
    """

    name: str

    def fetch(self, query: str) -> Optional[RetrievedAsset]: ...


# --------------------------------------------------------------------------- #
# URL validation + bounded HTTP helpers (the untrusted-fetch boundary)
# --------------------------------------------------------------------------- #

def is_allowed_url(url: object) -> bool:
    """True iff ``url`` is an http/https URL with a host (rejects file/ftp/data).

    The first gate on every remote fetch: search results are untrusted, so only
    plain web URLs are ever opened. Never raises.
    """
    if not isinstance(url, str) or not url:
        return False
    try:
        parsed = urllib.parse.urlparse(url)
    except (ValueError, TypeError):
        return False
    return parsed.scheme in ALLOWED_SCHEMES and bool(parsed.netloc)


class _ValidatingRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Re-validate every redirect hop so the scheme gate can't be bypassed.

    ``urllib`` follows 30x redirects automatically, and the stdlib handler
    permits http/https/ftp targets -- so a permitted https URL could redirect to
    ``ftp://`` or an internal host, defeating ``is_allowed_url`` (which is only
    applied to the FIRST URL). This subclass rejects any redirect target that
    ``is_allowed_url`` disallows before the hop is followed.
    """

    max_redirections = MAX_REDIRECTS

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        if not is_allowed_url(newurl):
            raise urllib.error.URLError(f"blocked redirect to disallowed URL ({newurl!r})")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


# Module opener with per-hop redirect validation + the redirect cap. Built once.
_OPENER = urllib.request.build_opener(_ValidatingRedirectHandler())
_OPENER.addheaders = []  # we set User-Agent explicitly per request


def _open(url: str, timeout: float) -> Any:
    """Open ``url`` (per-hop-validated redirects) with the required User-Agent."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    return _OPENER.open(request, timeout=timeout)


def _fetch_json(
    url: str, timeout: float = REQUEST_TIMEOUT_SECONDS, max_bytes: int = MAX_DOWNLOAD_BYTES
) -> Optional[Dict[str, Any]]:
    """GET ``url`` and parse a JSON object, or ``None`` on any failure.

    Bounded by ``timeout`` + ``max_bytes``. Returns ``None`` (with one stderr
    notice) on a bad scheme, transport/HTTP error, oversized body, or non-JSON.
    """
    if not is_allowed_url(url):
        return None
    try:
        with _open(url, timeout) as response:
            data = response.read(max_bytes + 1)
        if len(data) > max_bytes:
            print("Warning: retriever API response too large; ignoring.", file=sys.stderr)
            return None
        payload = json.loads(data.decode("utf-8"))
    except (urllib.error.URLError, OSError, TimeoutError, ValueError, RecursionError) as exc:
        # RecursionError: a deeply nested JSON body (json.loads recurses).
        print(f"Warning: retriever API request failed ({exc}); ignoring.", file=sys.stderr)
        return None
    return payload if isinstance(payload, dict) else None


def _download_bytes(
    url: str, timeout: float = REQUEST_TIMEOUT_SECONDS, max_bytes: int = MAX_DOWNLOAD_BYTES
) -> Optional[bytes]:
    """Download raw bytes from ``url`` within the size/time caps, or ``None``.

    Refuses non-http(s) URLs, an advertised ``Content-Length`` over the cap, and
    a body that streams past the cap. Returns ``None`` (with one stderr notice)
    on any transport failure. The bytes are NOT decoded here.
    """
    if not is_allowed_url(url):
        print(f"Warning: retriever refusing non-web URL ({url!r}).", file=sys.stderr)
        return None
    try:
        with _open(url, timeout) as response:
            declared = response.getheader("Content-Length")
            if declared is not None and declared.isdigit() and int(declared) > max_bytes:
                print("Warning: retriever image exceeds size cap; skipping.", file=sys.stderr)
                return None
            data = response.read(max_bytes + 1)
    except (urllib.error.URLError, OSError, TimeoutError, ValueError) as exc:
        print(f"Warning: retriever image download failed ({exc}); skipping.", file=sys.stderr)
        return None
    if len(data) > max_bytes:
        print("Warning: retriever image exceeds size cap; skipping.", file=sys.stderr)
        return None
    return data


# --------------------------------------------------------------------------- #
# Decode + downscale to BGRA (pure -- the in-memory image pipeline)
# --------------------------------------------------------------------------- #

def _downscale_to_max_dim(image: np.ndarray, max_dim: int) -> np.ndarray:
    """Downscale ``image`` so its longest side is at most ``max_dim`` (aspect kept)."""
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_dim:
        return image
    scale = max_dim / float(longest)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def probe_dimensions(data: bytes) -> Optional[tuple]:
    """Read (width, height) from an image header WITHOUT decoding pixels, or None.

    Covers the formats the retriever accepts (PNG/JPEG/GIF/BMP); returns ``None``
    for anything it can't cheaply parse (truncated/unknown headers). Used to
    reject a decompression bomb before ``cv2.imdecode`` allocates the full image.
    Never raises.
    """
    try:
        if data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) >= 24:
            w = int.from_bytes(data[16:20], "big")
            h = int.from_bytes(data[20:24], "big")
            return (w, h)
        if data[:3] == b"GIF" and len(data) >= 10:
            w = int.from_bytes(data[6:8], "little")
            h = int.from_bytes(data[8:10], "little")
            return (w, h)
        if data[:2] == b"BM" and len(data) >= 26:
            w = int.from_bytes(data[18:22], "little", signed=True)
            h = int.from_bytes(data[22:26], "little", signed=True)
            return (abs(w), abs(h))
        if data[:2] == b"\xff\xd8":  # JPEG: scan segments for an SOF marker
            i = 2
            n = len(data)
            while i + 9 < n:
                if data[i] != 0xFF:
                    i += 1
                    continue
                marker = data[i + 1]
                # SOF0..SOF15 carry the frame dimensions (skip DHT/DAC/RST/SOS).
                if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
                    h = int.from_bytes(data[i + 5:i + 7], "big")
                    w = int.from_bytes(data[i + 7:i + 9], "big")
                    return (w, h)
                if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
                    i += 2  # standalone markers carry no length
                    continue
                seg_len = int.from_bytes(data[i + 2:i + 4], "big")
                if seg_len < 2:
                    return None
                i += 2 + seg_len
    except (IndexError, ValueError):
        return None
    return None


def decode_to_bgra(data: bytes, max_dim: int = MAX_IMAGE_DIMENSION) -> Optional[np.ndarray]:
    """Decode downloaded bytes into a downscaled ``HxWx4`` BGRA array, or ``None``.

    Validates that the bytes are actually a decodable image (``cv2.imdecode``
    yields ``None`` on garbage -> ``None`` here, so non-image content can't slip
    through), normalizes to BGRA (grayscale/BGR gain an opaque alpha, existing
    alpha is kept via ``ensure_bgra``), then enforces ``max_dim``. A header-probed
    pixel count over ``MAX_DECODE_PIXELS`` is rejected BEFORE decode so a
    decompression bomb can't allocate gigabytes. Never raises.
    """
    if not data:
        return None
    dims = probe_dimensions(data)
    if dims is not None and dims[0] * dims[1] > MAX_DECODE_PIXELS:
        print("Warning: retriever image dimensions exceed the decode cap; skipping.",
              file=sys.stderr)
        return None
    try:
        buffer = np.frombuffer(data, dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
    except (cv2.error, ValueError):
        return None
    if image is None:
        return None
    if image.shape[0] * image.shape[1] > MAX_DECODE_PIXELS:
        return None  # backstop for a format probe_dimensions couldn't read
    image = ensure_bgra(image)
    if image.ndim != 3 or image.shape[2] != 4:
        # ensure_bgra only handles gray/BGR/BGRA; anything else (e.g. a rare
        # 2-channel gray+alpha) isn't a usable overlay -> treat as no-result.
        return None
    return _downscale_to_max_dim(image, max_dim)


# --------------------------------------------------------------------------- #
# Temp cache (download originals so an asset carries a real cache_path)
# --------------------------------------------------------------------------- #

def _cache_dir() -> str:
    """Return (creating if needed) the temp cache directory for downloads."""
    path = os.path.join(tempfile.gettempdir(), CACHE_DIR_NAME)
    os.makedirs(path, exist_ok=True)
    return path


def _slug(query: str) -> str:
    """A short filesystem-safe slug from the query (for a readable cache name).

    Reduces the (untrusted) query to ``[a-z0-9_]`` only, so path separators, ``..``,
    drive letters and NULs can't escape the cache dir; ``mkstemp`` adds randomness.
    """
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", query).strip("_").lower()
    return (cleaned[:_MAX_SLUG_CHARS] or "image")


def _prune_cache(directory: str, keep: int = MAX_CACHE_FILES) -> None:
    """Evict all but the ``keep`` most-recent files in the cache dir (LRU by mtime).

    Bounds unbounded temp growth over a long retrieving session. Best-effort: any
    OS error is ignored (the cache is disposable). Never raises.
    """
    try:
        entries = [os.path.join(directory, name) for name in os.listdir(directory)]
        files = [p for p in entries if os.path.isfile(p)]
        if len(files) <= keep:
            return
        files.sort(key=lambda p: os.path.getmtime(p))
        for path in files[:len(files) - keep]:
            try:
                os.remove(path)
            except OSError:
                pass
    except OSError:
        pass


def _save_to_cache(data: bytes, query: str) -> Optional[str]:
    """Write ``data`` to a unique temp file and return its path, or ``None``.

    A failed save is non-fatal: the asset's in-memory BGRA is still valid, so the
    caller keeps ``cache_path=None`` and carries on. The cache is pruned to
    ``MAX_CACHE_FILES`` after each write so it can't grow without bound.
    """
    try:
        directory = _cache_dir()
        fd, path = tempfile.mkstemp(prefix=f"{_slug(query)}_", suffix=".img", dir=directory)
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        _prune_cache(directory)
        return path
    except OSError as exc:
        print(f"Warning: retriever cache write failed ({exc}).", file=sys.stderr)
        return None


def _assemble(
    data: bytes, query: str, source_name: str, attribution: str, max_dim: int
) -> Optional[RetrievedAsset]:
    """Decode ``data`` to BGRA + cache it into a ``RetrievedAsset``, or ``None``.

    Shared tail of every source: garbage bytes (undecodable) yield ``None``.
    """
    bgra = decode_to_bgra(data, max_dim)
    if bgra is None:
        return None
    cache_path = _save_to_cache(data, query)
    return RetrievedAsset(
        bgra=bgra,
        source_name=source_name,
        query=query,
        attribution=attribution,
        cache_path=cache_path,
    )


# --------------------------------------------------------------------------- #
# Wikimedia response parsing (PURE -- the best-image selection, unit-tested)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class WikimediaImage:
    """One candidate raster image parsed from a Commons API response."""

    url: str
    mime: str
    attribution: str
    search_index: int


def _extmeta_value(extmetadata: object, key: str) -> str:
    """Return ``extmetadata[key]['value']`` as a clean string, or ""."""
    if not isinstance(extmetadata, dict):
        return ""
    entry = extmetadata.get(key)
    if not isinstance(entry, dict):
        return ""
    value = entry.get("value")
    return _clean_html(value) if isinstance(value, str) else ""


def _clean_html(text: str) -> str:
    """Strip HTML tags/entities and collapse whitespace (extmetadata is HTML)."""
    without_tags = re.sub(r"<[^>]+>", " ", text)
    unescaped = html.unescape(without_tags)
    return re.sub(r"\s+", " ", unescaped).strip()


def _title_label(title: object) -> str:
    """A human title from a page ``title`` ("File:CFTR map.png" -> "CFTR map")."""
    if not isinstance(title, str):
        return ""
    label = title.split(":", 1)[1] if ":" in title else title
    label = os.path.splitext(label)[0]
    return label.strip()


def _build_attribution(title: object, extmetadata: object) -> str:
    """Compose a short "<title> (<license>) by <artist>" attribution string."""
    parts: List[str] = []
    label = _title_label(title)
    if label:
        parts.append(label)
    license_name = _extmeta_value(extmetadata, "LicenseShortName")
    if license_name:
        parts.append(f"({license_name})")
    artist = _extmeta_value(extmetadata, "Artist")
    if artist:
        parts.append(f"by {artist}")
    return " ".join(parts)[:_MAX_ATTRIBUTION_CHARS]


def _pages_of(payload: object) -> Sequence[Dict[str, Any]]:
    """Yield the page dicts from a MediaWiki response (dict- or list-keyed)."""
    if not isinstance(payload, dict):
        return ()
    query = payload.get("query")
    if not isinstance(query, dict):
        return ()
    pages = query.get("pages")
    if isinstance(pages, dict):
        values = pages.values()
    elif isinstance(pages, list):
        values = pages
    else:
        return ()
    return [page for page in values if isinstance(page, dict)]


def _extract_candidates(payload: object) -> List[WikimediaImage]:
    """Parse all usable raster-image candidates from a Commons API response."""
    candidates: List[WikimediaImage] = []
    for page in _pages_of(payload):
        infos = page.get("imageinfo")
        if not isinstance(infos, list) or not infos:
            continue  # a page with no imageinfo (e.g. non-file) is skipped
        info = infos[0]
        if not isinstance(info, dict):
            continue
        mime = str(info.get("mime", "")).lower()
        if mime not in _ALLOWED_MIMES:
            continue  # skip SVG/PDF/TIFF/audio/video
        # Prefer the server-downscaled thumbnail (respects iiurlwidth) over the
        # full-resolution original.
        url = info.get("thumburl") or info.get("url")
        if not isinstance(url, str) or not is_allowed_url(url):
            continue
        raw_index = page.get("index", 0)
        search_index = raw_index if isinstance(raw_index, int) else 0
        attribution = _build_attribution(page.get("title", ""), info.get("extmetadata"))
        candidates.append(WikimediaImage(url, mime, attribution, search_index))
    return candidates


def select_best_image(payload: object) -> Optional[WikimediaImage]:
    """Pick the best raster image from a Commons API response, or ``None``.

    Selection criteria (deterministic): keep only allowed raster mimetypes
    (drops SVG/PDF/audio), prefer jpeg/png over gif/webp, then take the highest
    search relevance (lowest ``index``). Returns ``None`` when the search had no
    usable image (empty results, or every hit an unsupported type / no imageinfo).
    """
    candidates = _extract_candidates(payload)
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda c: (0 if c.mime in _PREFERRED_MIMES else 1, c.search_index),
    )


# --------------------------------------------------------------------------- #
# WikimediaSource (default backend, stdlib-only)
# --------------------------------------------------------------------------- #

class WikimediaSource:
    """Fetch the best licensed image from Wikimedia Commons (stdlib urllib)."""

    name = "wikimedia"

    def __init__(
        self,
        max_dim: int = MAX_IMAGE_DIMENSION,
        max_results: int = DEFAULT_MAX_RESULTS,
        timeout: float = REQUEST_TIMEOUT_SECONDS,
    ) -> None:
        self._max_dim = max_dim
        self._max_results = max_results
        self._timeout = timeout

    def _query_url(self, query: str) -> str:
        """Build the MediaWiki ``generator=search`` imageinfo query URL."""
        params = {
            "action": "query",
            "generator": "search",
            "gsrsearch": query,  # urlencode escapes it (query is length-bounded)
            "gsrnamespace": "6",  # File: namespace only
            "gsrlimit": str(self._max_results),
            "prop": "imageinfo",
            "iiprop": "url|extmetadata|mime|size",
            "iiurlwidth": str(self._max_dim),  # ask the server to downscale
            "format": "json",
        }
        return f"{WIKIMEDIA_API_URL}?{urllib.parse.urlencode(params)}"

    def fetch(self, query: str) -> Optional[RetrievedAsset]:
        payload = _fetch_json(self._query_url(query), self._timeout)
        best = select_best_image(payload)
        if best is None:
            return None
        data = _download_bytes(best.url, self._timeout)
        if data is None:
            return None
        return _assemble(data, query, self.name, best.attribution, self._max_dim)


# --------------------------------------------------------------------------- #
# WebSearchSource (fallback backend, DuckDuckGo via lazily-imported ddgs)
# --------------------------------------------------------------------------- #

class WebSearchSource:
    """Fallback backend: DuckDuckGo image search via the ``ddgs`` package.

    ``ddgs`` is imported LAZILY inside ``_search_urls`` so the base pipeline runs
    without it -- its absence simply disables this fallback (one stderr notice).
    """

    name = "websearch"

    def __init__(
        self,
        max_dim: int = MAX_IMAGE_DIMENSION,
        max_results: int = DEFAULT_MAX_RESULTS,
        timeout: float = REQUEST_TIMEOUT_SECONDS,
    ) -> None:
        self._max_dim = max_dim
        self._max_results = max_results
        self._timeout = timeout

    def _search_urls(self, query: str) -> List[str]:
        """Return candidate image URLs from ddgs, or [] if unavailable/empty."""
        try:
            from ddgs import DDGS  # lazy: optional dep
        except ImportError:
            print(
                "Warning: 'ddgs' not installed; web-search fallback disabled "
                "(pip install ddgs).",
                file=sys.stderr,
            )
            return []
        try:
            with DDGS(timeout=int(self._timeout)) as client:
                hits = client.images(query, max_results=self._max_results)
        except Exception as exc:  # noqa: BLE001 - a search backend can't crash us
            print(f"Warning: web image search failed ({exc}); ignoring.", file=sys.stderr)
            return []
        urls: List[str] = []
        for hit in hits or ():
            if isinstance(hit, dict):
                url = hit.get("image") or hit.get("url")
                if isinstance(url, str):
                    urls.append(url)
        return urls

    def fetch(self, query: str) -> Optional[RetrievedAsset]:
        for url in self._search_urls(query):
            if not is_allowed_url(url):
                continue
            data = _download_bytes(url, self._timeout)
            if data is None:
                continue  # try the next hit
            asset = _assemble(data, query, self.name, f"web image search: {url}", self._max_dim)
            if asset is not None:
                return asset  # first hit that downloads AND decodes wins
        return None


# --------------------------------------------------------------------------- #
# LocalLibrarySource (optional, filename-substring matching -- CLIP deferred)
# --------------------------------------------------------------------------- #

def match_filename(filenames: Sequence[str], query: str) -> Optional[str]:
    """Pick the filename matching the most query tokens (case-insensitive), else None.

    A deliberately SIMPLE matcher for this pass: tokenize the query on
    non-alphanumerics and score each filename by how many tokens appear as
    substrings of it; the highest-scoring name (score > 0) wins, ties broken by
    input order. Semantic CLIP-embedding matching is explicitly DEFERRED.
    """
    tokens = [token for token in re.split(r"\W+", query.lower()) if token]
    if not tokens:
        return None
    best_name: Optional[str] = None
    best_score = 0
    for name in filenames:
        lowered = name.lower()
        score = sum(1 for token in tokens if token in lowered)
        if score > best_score:
            best_score, best_name = score, name
    return best_name if best_score > 0 else None


class LocalLibrarySource:
    """Serve a preloaded folder, matched to the query by filename substring."""

    name = "local"

    def __init__(self, folder: Optional[str], max_dim: int = MAX_IMAGE_DIMENSION) -> None:
        self._folder = folder
        self._max_dim = max_dim

    def _image_files(self) -> List[str]:
        """Sorted image filenames in the folder ([] if no/invalid folder)."""
        if not self._folder or not os.path.isdir(self._folder):
            return []
        try:
            entries = sorted(os.listdir(self._folder))
        except OSError:
            return []
        return [
            name
            for name in entries
            if os.path.splitext(name)[1].lower() in _IMAGE_EXTENSIONS
            and os.path.isfile(os.path.join(self._folder, name))
        ]

    def fetch(self, query: str) -> Optional[RetrievedAsset]:
        match = match_filename(self._image_files(), query)
        if match is None:
            return None
        path = os.path.join(self._folder or "", match)
        try:
            with open(path, "rb") as handle:
                data = handle.read()
        except OSError:
            return None
        bgra = decode_to_bgra(data, self._max_dim)
        if bgra is None:
            return None
        # cache_path points at the source file itself (no copy needed).
        return RetrievedAsset(
            bgra=bgra,
            source_name=self.name,
            query=query,
            attribution=f"local library: {match}",
            cache_path=path,
        )


# --------------------------------------------------------------------------- #
# Retriever: compose sources by the request's source hint, with fallback
# --------------------------------------------------------------------------- #

@dataclass
class Retriever:
    """Compose the backends and fetch an image for a parsed ``retrieve`` request.

    Ordering: a configured ``local`` library is tried FIRST (offline hit),
    then the request's preferred source, then the other web source as a fallback
    (source "wikimedia" -> [wikimedia, websearch]; "websearch" -> [websearch,
    wikimedia]). ``retrieve`` acts only on ``route="retrieve"`` requests, catches
    everything, and NEVER raises -- a total failure yields ``None`` + one notice.
    """

    wikimedia: WikimediaSource = field(default_factory=WikimediaSource)
    websearch: WebSearchSource = field(default_factory=WebSearchSource)
    local: Optional[LocalLibrarySource] = None

    def _sources_for(self, request: ParsedRequest) -> List[RetrievalSource]:
        ordered: List[RetrievalSource] = []
        if self.local is not None:
            ordered.append(self.local)
        from src.parser import SOURCE_WEBSEARCH  # local import: avoid a top-line

        if request.source == SOURCE_WEBSEARCH:
            ordered += [self.websearch, self.wikimedia]
        else:
            ordered += [self.wikimedia, self.websearch]
        return ordered

    def retrieve(self, request: Optional[ParsedRequest]) -> Optional[RetrievedAsset]:
        """Fetch an image for ``request``; ``None`` if it can't (never raises)."""
        try:
            if request is None or request.route != ROUTE_RETRIEVE:
                return None  # only retrieve-routed requests are ours (generate = Ph16)
            query = (request.query or "").strip()
            if not query:
                return None
            for source in self._sources_for(request):
                try:
                    asset = source.fetch(query)
                except Exception as exc:  # noqa: BLE001 - one bad source can't stop us
                    print(
                        f"Warning: retriever source {getattr(source, 'name', '?')!r} "
                        f"errored ({exc}); trying the next.",
                        file=sys.stderr,
                    )
                    asset = None
                if asset is not None:
                    return asset
            return None  # no source produced an image (caller logs "no image")
        except Exception as exc:  # noqa: BLE001 - retrieve must never raise upstream
            print(f"Warning: retriever failed ({exc}); no image.", file=sys.stderr)
            return None


def build_retriever(local_folder: Optional[str] = None) -> Retriever:
    """Assemble a production ``Retriever`` (Wikimedia default + web fallback).

    Pass ``local_folder`` to also try a preloaded local library FIRST.
    """
    local = LocalLibrarySource(local_folder) if local_folder else None
    return Retriever(local=local)

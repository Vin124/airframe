"""Enumerate currently-open top-level window titles for the panel's picker.

Pure ctypes over ``user32`` (``EnumWindows`` + ``IsWindowVisible`` +
``GetWindowTextLengthW`` / ``GetWindowTextW``) -- no new pip dependency. The
Windows enumeration is injectable (the ``enumerator`` seam) so the filtering /
dedup / sort logic is unit-testable with a fake, and the real path is wrapped so
ANY ctypes failure yields ``[]`` rather than propagating -- enumerating windows
must never crash the control panel.
"""

from typing import Callable, Iterable, List

# The app's own windows, hidden from the picker (matched case-insensitively):
# the OpenCV preview ("handy") and the Tk control panel ("handy control").
_OWN_TITLES = frozenset({"handy", "handy control"})


def _win32_titles() -> List[str]:
    """Collect visible top-level window titles via ctypes ``user32`` (real path).

    Imported lazily and used only on Windows; any failure here is caught by
    ``list_window_titles`` and turned into an empty list.
    """
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    titles: List[str] = []

    # BOOL EnumWindowsProc(HWND, LPARAM); returning TRUE keeps the enumeration going.
    proc_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    # Explicit prototypes: without argtypes, ctypes converts HWNDs through the
    # default signed 32-bit c_int, so a handle with the high bit set raises
    # mid-enumeration. HWND keeps the full pointer width.
    user32.IsWindowVisible.argtypes = [wintypes.HWND]
    user32.IsWindowVisible.restype = wintypes.BOOL
    user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
    user32.GetWindowTextLengthW.restype = ctypes.c_int
    user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user32.GetWindowTextW.restype = ctypes.c_int
    user32.EnumWindows.argtypes = [proc_type, wintypes.LPARAM]
    user32.EnumWindows.restype = wintypes.BOOL

    def _callback(hwnd, _lparam):
        # A Python exception can't propagate through the C callback boundary --
        # ctypes would print it to stderr and abort the enumeration mid-way,
        # silently yielding a partial list. Guarding here keeps the module's
        # "any failure -> []" contract meaningful: per-window failures are
        # skipped, the enumeration completes.
        try:
            if not user32.IsWindowVisible(hwnd):
                return True
            length = user32.GetWindowTextLengthW(hwnd)
            if length <= 0:
                return True
            buffer = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buffer, length + 1)
            titles.append(buffer.value)
        except Exception:  # noqa: BLE001 - must not cross the C boundary
            pass
        return True

    user32.EnumWindows(proc_type(_callback), 0)
    return titles


def list_window_titles(
    enumerator: Callable[[], Iterable[str]] = _win32_titles,
) -> List[str]:
    """Return open top-level window titles: filtered, deduped, sorted (case-insensitive).

    ``enumerator`` yields raw titles (defaults to the real ctypes path; a fake is
    injected in tests). Empty / whitespace-only titles and the app's own windows
    are dropped, duplicates are collapsed case-insensitively (first-seen casing
    wins), and the result is sorted case-insensitively. ANY exception from the
    enumerator returns ``[]`` -- the picker must never crash the panel.
    """
    try:
        raw = list(enumerator())
    except Exception:  # noqa: BLE001 - enumeration must never crash the panel
        return []

    deduped: dict = {}
    for title in raw:
        if not isinstance(title, str):
            continue
        stripped = title.strip()
        if not stripped:
            continue
        key = stripped.lower()
        if key in _OWN_TITLES or key in deduped:
            continue
        deduped[key] = stripped
    return sorted(deduped.values(), key=str.lower)

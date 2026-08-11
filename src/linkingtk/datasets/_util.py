"""Helpers shared across dataset loader modules."""

from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.request import urlopen

_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "linkingtk" / "downloads"


def local_name(uri: str) -> str:
    """The local (fragment/path-final) part of a URI, e.g. ``.../Big_Cat`` -> ``Big_Cat``."""
    return uri.rsplit("#", 1)[-1].rsplit("/", 1)[-1]


def label_from_raw(raw: str) -> str:
    """A human-readable label from a raw dataset value.

    URIs are reduced to their local name with underscores humanized (e.g.
    ``http://dbpedia.org/resource/Big_Cat`` -> ``Big Cat``); anything else
    (already-plain-text entries, as some datasets have) passes through
    unchanged.
    """
    if raw.startswith(("http://", "https://")):
        return local_name(raw).replace("_", " ")
    return raw


def fetch_cached(url: str, cache_dir: Path | None = None) -> bytes:
    """Fetch ``url``'s bytes, caching to disk under ``cache_dir``.

    ``file://`` URLs are read directly and never cached -- they're already
    local, used to point tests at small fixture files/archives instead of
    the real (multi-MB) hosted datasets.

    Args:
        url: The URL to fetch.
        cache_dir: Directory to cache downloads in. Defaults to
            ``~/.cache/linkingtk/downloads``.

    Returns:
        The response body.
    """
    if url.startswith("file://"):
        with urlopen(url) as response:
            data: bytes = response.read()
        return data

    cache_dir = cache_dir if cache_dir is not None else _DEFAULT_CACHE_DIR
    cache_path = cache_dir / hashlib.sha256(url.encode()).hexdigest()
    if cache_path.exists():
        return cache_path.read_bytes()

    with urlopen(url) as response:
        fetched: bytes = response.read()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(fetched)
    return fetched

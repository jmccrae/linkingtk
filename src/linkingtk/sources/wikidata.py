"""EntitySource wrapper for Wikidata, via the public action API.

Wraps [Wikidata's action API](https://www.wikidata.org/w/api.php) as a
query-driven [EntitySource][linkingtk.core.source.EntitySource], the same way
[WikipediaEntitySource][linkingtk.sources.wikipedia.WikipediaEntitySource]
targets live Wikipedia. Live per-mention Wikidata queries don't scale well
(see [linkingtk.sources.vector_index][]'s module docstring) -- pass a
prebuilt [VectorIndexEntitySource][linkingtk.sources.vector_index.VectorIndexEntitySource]
as `vector_index` to search/get entirely locally instead. `WikidataDumpEntities`
in this module streams `Entity` objects straight from a downloaded Wikidata
dump to build one.
"""

from __future__ import annotations

import gzip
import io
import json
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, cast

from tqdm import tqdm

from linkingtk.core.entity import Entity
from linkingtk.core.source import EntitySource
from linkingtk.exceptions import OptionalDependencyError
from linkingtk.sources.vector_index import VectorIndexEntitySource

if TYPE_CHECKING:
    import requests

_USER_AGENT = "linkingtk/0.1 (https://github.com/jmccrae/linkingtk)"


def _is_url(path: str) -> bool:
    return path.startswith(("http://", "https://"))


def _instance_of_qids(claims: dict[str, Any]) -> list[str]:
    """QIDs from a Wikidata entity's ``P31`` ("instance of") claims."""
    qids = []
    for claim in claims.get("P31", []):
        value = claim.get("mainsnak", {}).get("datavalue", {}).get("value")
        if isinstance(value, dict) and "id" in value:
            qids.append(value["id"])
    return qids


def _entity_from_wikidata_json(data: dict[str, Any], lang: str) -> Entity | None:
    """Build an `Entity` from one Wikidata item's own JSON representation.

    ``wbgetentities``/``wbsearchentities`` responses and the official bulk
    JSON dump (see `WikidataDumpEntities`) both represent an item with this
    identical schema, so both go through this one extraction. `None` if
    `data` has no `lang` label -- unusable as a search/blocking target.
    """
    label = data.get("labels", {}).get(lang, {}).get("value")
    if label is None:
        return None
    aliases = [alias["value"] for alias in data.get("aliases", {}).get(lang, [])]
    labels: list[str | tuple[str, str]] = [label] + [a for a in aliases if a != label]
    description = data.get("descriptions", {}).get(lang, {}).get("value")
    instance_of = _instance_of_qids(data.get("claims", {}))
    properties = {"instance_of": " ".join(instance_of)} if instance_of else {}
    return Entity(id=data["id"], labels=labels, description=description, properties=properties)


class WikidataDumpEntities:
    """Streams `Entity` objects from an official Wikidata JSON dump.

    Wikidata publishes its full dump
    (https://www.wikidata.org/wiki/Wikidata:Database_download) as
    ``latest-all.json.gz``: a JSON array of entity objects, one compact
    object per line (plus bracket-only first/last lines) -- written that
    way specifically so it can be streamed line by line rather than parsed
    as one huge array, which is exactly what this class does. A truncated
    or filtered dump in the same line-delimited shape works too.

    Generalizes (reads the dump directly, no intermediate file needed) the
    dump-processing step of
    [`wn-wd-entity-align`](https://github.com/jmccrae/wn-wd-entity-align)'s
    own FAISS-index pipeline
    (``wikidata_faiss/extract_dump_labels.py``, which instead expects an
    already-filtered, already-sorted N-Triples export).

    Re-iterable -- `__iter__` reopens (re-requests the URL, or re-opens the
    local file) and re-reads from the start every call -- so it can be
    passed straight to
    [VectorIndexEntitySource.build][linkingtk.sources.vector_index.VectorIndexEntitySource.build]
    even with `reduced_dim` set (which needs two passes over its input); a
    plain generator can't do that, since it's exhausted after one pass.

    Args:
        path: Path to the dump file, or an ``http(s)://`` URL to stream it
            from directly (e.g. Wikidata's own dump mirrors) -- fetched
            fresh, never cached to disk, since the whole point is not
            having to hold a multi-hundred-GB file locally. Either way,
            gzip-decompressed on the fly if the name ends in ``.gz`` (as
            the official dump does).
        lang: Which language's label/description/aliases to read -- see
            `WikidataEntitySource`.
        limit: Stop after yielding this many entities, e.g. for a quick
            local test against a full multi-hundred-GB dump.
        progress: Show a `tqdm` progress bar (lines read, entities
            yielded) while iterating -- this can run for hours over a full
            dump, so some feedback matters. Pass `False` to disable.
    """

    def __init__(
        self,
        path: str | Path,
        lang: str = "en",
        limit: int | None = None,
        progress: bool = True,
    ) -> None:
        self.path = path if isinstance(path, str) and _is_url(path) else Path(path)
        self.lang = lang
        self.limit = limit
        self.progress = progress

    def _open_binary(self) -> IO[bytes]:
        if isinstance(self.path, str):
            request = urllib.request.Request(self.path, headers={"User-Agent": _USER_AGENT})
            raw: IO[bytes] = urllib.request.urlopen(request)  # noqa: S310
            is_gzip = self.path.endswith(".gz")
        else:
            raw = self.path.open("rb")
            is_gzip = self.path.name.endswith(".gz")
        return cast("IO[bytes]", gzip.GzipFile(fileobj=raw)) if is_gzip else raw

    def __iter__(self) -> Iterator[Entity]:
        count = 0
        with io.TextIOWrapper(self._open_binary(), encoding="utf-8") as f:
            lines = tqdm(f, desc=str(self.path), unit=" lines", disable=not self.progress)
            for line in lines:
                stripped = line.strip().rstrip(",")
                if stripped in ("", "[", "]"):
                    continue
                try:
                    data = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                if data.get("type") != "item":
                    continue
                entity = _entity_from_wikidata_json(data, self.lang)
                if entity is None:
                    continue
                count += 1
                lines.set_postfix(entities=count)
                yield entity
                if self.limit is not None and count >= self.limit:
                    return


class WikidataEntitySource(EntitySource):
    """Queries live Wikidata, via its action API, as an
    [EntitySource][linkingtk.core.source.EntitySource].

    Entity ids are QIDs (e.g. ``"Q42"``). A ``P31`` ("instance of") claim,
    when present, is surfaced into ``Entity.properties["instance_of"]`` as
    space-joined QIDs.

    Args:
        lang: The Wikidata language to request labels/descriptions in
            (e.g. ``"en"``).
        session: A `requests.Session` to issue requests through. Defaults to
            a fresh one carrying a descriptive `User-Agent` -- see
            [WikipediaEntitySource][linkingtk.sources.wikipedia.WikipediaEntitySource]
            for why that matters. A caller-supplied session is used exactly
            as given, headers untouched.
        vector_index: An optional prebuilt local index (see
            [linkingtk.sources.vector_index][]) to search/get against
            instead of the live API. `search` delegates to it entirely;
            `get` tries it first and only falls back to a live
            ``wbgetentities`` call on a miss (the index may be a partial or
            sampled build).

    Raises:
        OptionalDependencyError: If `requests` isn't installed.
    """

    def __init__(
        self,
        lang: str = "en",
        session: requests.Session | None = None,
        vector_index: VectorIndexEntitySource | None = None,
    ) -> None:
        try:
            import requests
        except ImportError as exc:
            raise OptionalDependencyError("WikidataEntitySource", "wikipedia") from exc
        self.lang = lang
        self.vector_index = vector_index
        self._api_url = "https://www.wikidata.org/w/api.php"
        if session is not None:
            self._session = session
        else:
            self._session = requests.Session()
            self._session.headers["User-Agent"] = _USER_AGENT

    def search(self, query: str, top_k: int = 10) -> list[Entity]:
        """Nearest entities to `query`, best match first.

        Delegates to `vector_index` (no network) if one was given at
        construction; otherwise uses ``action=wbsearchentities``.
        """
        if self.vector_index is not None:
            return self.vector_index.search(query, top_k)

        params: dict[str, str | int] = {
            "action": "wbsearchentities",
            "search": query,
            "language": self.lang,
            "limit": top_k,
            "format": "json",
        }
        response = self._session.get(self._api_url, params=params, timeout=10)
        response.raise_for_status()
        hits = response.json()["search"]
        return [
            Entity(id=hit["id"], labels=[hit["label"]], description=hit.get("description"))
            for hit in hits
            if "label" in hit
        ]

    def get(self, entity_id: str) -> Entity | None:
        """Look up a QID, or ``None`` if it doesn't resolve.

        Tries `vector_index` first (if given); falls back to a live
        ``action=wbgetentities`` call on a miss.
        """
        if self.vector_index is not None:
            cached = self.vector_index.get(entity_id)
            if cached is not None:
                return cached

        params: dict[str, str | int] = {
            "action": "wbgetentities",
            "ids": entity_id,
            "languages": self.lang,
            "format": "json",
        }
        response = self._session.get(self._api_url, params=params, timeout=10)
        response.raise_for_status()
        payload = response.json()
        # Wikidata reports a not-found entity two different ways: a
        # syntactically plausible but nonexistent QID comes back inside
        # "entities" with a "missing" marker, while a QID outside any
        # plausible id range comes back as a top-level API error instead --
        # confirmed directly against the live API for "Q999999999" (missing
        # marker) vs. "Q999999999999" (top-level error).
        if "error" in payload:
            return None
        entity = next(iter(payload["entities"].values()))
        if "missing" in entity:
            return None
        return _entity_from_wikidata_json(entity, self.lang)

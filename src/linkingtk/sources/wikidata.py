"""EntitySource wrapper for Wikidata, via the public action API.

Wraps [Wikidata's action API](https://www.wikidata.org/w/api.php) as a
query-driven [EntitySource][linkingtk.core.source.EntitySource], the same way
[WikipediaEntitySource][linkingtk.sources.wikipedia.WikipediaEntitySource]
targets live Wikipedia. Live per-mention Wikidata queries don't scale well
(see [linkingtk.sources.vector_index][]'s module docstring) -- pass a
prebuilt [VectorIndexEntitySource][linkingtk.sources.vector_index.VectorIndexEntitySource]
as `vector_index` to search/get entirely locally instead.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from linkingtk.core.entity import Entity
from linkingtk.core.source import EntitySource
from linkingtk.exceptions import OptionalDependencyError
from linkingtk.sources.vector_index import VectorIndexEntitySource

if TYPE_CHECKING:
    import requests

_USER_AGENT = "linkingtk/0.1 (https://github.com/jmccrae/linkingtk)"


def _instance_of_qids(claims: dict[str, Any]) -> list[str]:
    """QIDs from a ``wbgetentities`` entity's ``P31`` ("instance of") claims."""
    qids = []
    for claim in claims.get("P31", []):
        value = claim.get("mainsnak", {}).get("datavalue", {}).get("value")
        if isinstance(value, dict) and "id" in value:
            qids.append(value["id"])
    return qids


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

        label = entity.get("labels", {}).get(self.lang, {}).get("value")
        if label is None:
            return None
        description = entity.get("descriptions", {}).get(self.lang, {}).get("value")
        instance_of = _instance_of_qids(entity.get("claims", {}))
        properties = {"instance_of": " ".join(instance_of)} if instance_of else {}
        return Entity(
            id=entity["id"], labels=[label], description=description, properties=properties
        )

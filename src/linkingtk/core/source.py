"""Query-driven target datasets for blocking/linking against huge or external entity sets."""

from __future__ import annotations

from abc import ABC, abstractmethod

from linkingtk.core.entity import Entity


class EntitySource(ABC):
    """A query-driven stand-in for a fully materialized ``list[Entity]`` target dataset.

    Lets [BlockingStrategy][linkingtk.blocking.base.BlockingStrategy] and
    [BaseLinker][linkingtk.algorithms.base.BaseLinker] target datasets too
    large, or already indexed elsewhere, to enumerate into memory -- e.g.
    WordNet (via ``wn``), Wikipedia, Wikidata -- by querying per-mention
    instead.
    """

    @abstractmethod
    def search(self, query: str, top_k: int = 10) -> list[Entity]:
        """Return up to ``top_k`` entities matching ``query``, best match first."""

    @abstractmethod
    def get(self, entity_id: str) -> Entity | None:
        """Look up a single entity by id, or ``None`` if it doesn't exist."""

    def search_batch(self, queries: list[str], top_k: int = 10) -> list[list[Entity]]:
        """Return ``search(query, top_k)`` results for each of ``queries``, in order.

        Defaults to looping over [search][linkingtk.core.source.EntitySource.search];
        override this for sources with a real batch API (e.g. a local ANN
        index, Elasticsearch ``_msearch``).
        """
        return [self.search(query, top_k) for query in queries]


class CachingEntitySource(EntitySource):
    """Wraps an ``EntitySource``, memoizing ``search``/``get`` calls.

    HTTP-backed or otherwise slow/rate-limited sources benefit from this
    since the same query (e.g. a repeated mention text) tends to recur
    within a session.
    """

    def __init__(self, source: EntitySource) -> None:
        self._source = source
        self._search_cache: dict[tuple[str, int], list[Entity]] = {}
        self._get_cache: dict[str, Entity | None] = {}

    def search(self, query: str, top_k: int = 10) -> list[Entity]:
        key = (query, top_k)
        if key not in self._search_cache:
            self._search_cache[key] = self._source.search(query, top_k)
        return self._search_cache[key]

    def get(self, entity_id: str) -> Entity | None:
        if entity_id not in self._get_cache:
            self._get_cache[entity_id] = self._source.get(entity_id)
        return self._get_cache[entity_id]

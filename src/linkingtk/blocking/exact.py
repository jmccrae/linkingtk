"""Exact-label blocking strategy."""

from __future__ import annotations

from collections import defaultdict

from linkingtk.blocking.base import BlockingStrategy
from linkingtk.core.entity import Entity, label_texts
from linkingtk.core.source import EntitySource


def _normalize(text: str) -> str:
    """Lowercase, underscore/space-insensitive form used only for the
    ``EntitySource`` post-filter comparison (never for the query sent to
    ``search()`` itself). Multi-word lemmas are joined with underscores in
    the classic WordNet distribution format but indexed space-separated by
    some `wn` lexicons (e.g. OMW's ``omw-en:1.4``/``omw-en:2.0`` -- see
    [sources.wn][linkingtk.sources.wn]'s own underscore/space fallback for
    the query side of this same mismatch).
    """
    return text.lower().replace("_", " ")


class ExactMatch(BlockingStrategy):
    """Blocks entities that share at least one identical label.

    This is the default blocking strategy: two entities are considered
    candidates if any of their labels are exactly equal (language tags are
    ignored for comparison purposes). Also supports an
    [EntitySource][linkingtk.core.source.EntitySource] for ``dataset2``: it
    queries ``dataset2.search(label)`` per label instead of enumerating the
    whole target set, then keeps only the results that actually have a
    matching label -- same semantics, no materialization required.

    Args:
        top_k: Forwarded to ``dataset2.search(label, top_k=...)`` when
            ``dataset2`` is an ``EntitySource`` -- ignored for a plain
            ``list[Entity]`` ``dataset2``, which has no such cap. The
            default (``EntitySource.search``'s own default) is too narrow
            for a highly polysemous word once every one of its senses is a
            valid candidate (e.g. WSD against a full WordNet) -- raise
            this in that case.
    """

    def __init__(self, top_k: int = 10) -> None:
        self.top_k = top_k

    def candidate_pairs(
        self, dataset1: list[Entity], dataset2: list[Entity] | EntitySource
    ) -> list[tuple[Entity, Entity]]:
        if isinstance(dataset2, EntitySource):
            return self._candidate_pairs_from_source(dataset1, dataset2)
        return self._candidate_pairs_from_list(dataset1, dataset2)

    def _candidate_pairs_from_list(
        self, dataset1: list[Entity], dataset2: list[Entity]
    ) -> list[tuple[Entity, Entity]]:
        index: dict[str, list[Entity]] = defaultdict(list)
        for entity in dataset2:
            for text in set(label_texts(entity)):
                index[text].append(entity)

        pairs: list[tuple[Entity, Entity]] = []
        for entity1 in dataset1:
            seen: set[str] = set()
            for text in set(label_texts(entity1)):
                for entity2 in index.get(text, []):
                    if entity2.id not in seen:
                        pairs.append((entity1, entity2))
                        seen.add(entity2.id)
        return pairs

    def _candidate_pairs_from_source(
        self, dataset1: list[Entity], dataset2: EntitySource
    ) -> list[tuple[Entity, Entity]]:
        pairs: list[tuple[Entity, Entity]] = []
        for entity1 in dataset1:
            seen: set[str] = set()
            for text in set(label_texts(entity1)):
                # Normalized (case- and underscore/space-insensitive) on this
                # side only: an EntitySource's own search() may itself be
                # looser than exact string equality (e.g. `wn`'s -- querying
                # "friday" finds the synset WordNet lemmatizes as "Friday",
                # and "point_out" finds "point out"), so a stricter
                # post-filter here would silently drop real matches
                # search() already found.
                for entity2 in dataset2.search(text, top_k=self.top_k):
                    if entity2.id in seen:
                        continue
                    if _normalize(text) in {_normalize(t) for t in label_texts(entity2)}:
                        pairs.append((entity1, entity2))
                        seen.add(entity2.id)
        return pairs

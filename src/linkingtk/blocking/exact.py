"""Exact-label blocking strategy."""

from __future__ import annotations

from collections import defaultdict

from linkingtk.blocking.base import BlockingStrategy
from linkingtk.core.entity import Entity, label_texts
from linkingtk.core.source import EntitySource


class ExactMatch(BlockingStrategy):
    """Blocks entities that share at least one identical label.

    This is the default blocking strategy: two entities are considered
    candidates if any of their labels are exactly equal (language tags are
    ignored for comparison purposes). Also supports an
    [EntitySource][linkingtk.core.source.EntitySource] for ``dataset2``: it
    queries ``dataset2.search(label)`` per label instead of enumerating the
    whole target set, then keeps only the results that actually have a
    matching label -- same semantics, no materialization required.
    """

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
                for entity2 in dataset2.search(text):
                    if entity2.id in seen:
                        continue
                    if text in set(label_texts(entity2)):
                        pairs.append((entity1, entity2))
                        seen.add(entity2.id)
        return pairs

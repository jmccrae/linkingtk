"""Exact-label blocking strategy."""

from __future__ import annotations

from collections import defaultdict

from linkingtk.blocking.base import BlockingStrategy
from linkingtk.core.entity import Entity


def _label_texts(entity: Entity) -> set[str]:
    """Extract the plain text of each label, ignoring language tags."""
    texts = set()
    for label in entity.labels:
        texts.add(label[0] if isinstance(label, tuple) else label)
    return texts


class ExactMatch(BlockingStrategy):
    """Blocks entities that share at least one identical label.

    This is the default blocking strategy: two entities are considered
    candidates if any of their labels are exactly equal (language tags are
    ignored for comparison purposes).
    """

    def candidate_pairs(
        self, dataset1: list[Entity], dataset2: list[Entity]
    ) -> list[tuple[Entity, Entity]]:
        index: dict[str, list[Entity]] = defaultdict(list)
        for entity in dataset2:
            for text in _label_texts(entity):
                index[text].append(entity)

        pairs: list[tuple[Entity, Entity]] = []
        for entity1 in dataset1:
            seen: set[str] = set()
            for text in _label_texts(entity1):
                for entity2 in index.get(text, []):
                    if entity2.id not in seen:
                        pairs.append((entity1, entity2))
                        seen.add(entity2.id)
        return pairs

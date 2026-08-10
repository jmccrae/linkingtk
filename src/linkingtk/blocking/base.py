"""Abstract interface for blocking strategies."""

from __future__ import annotations

import heapq
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable

from linkingtk.core.entity import Entity


class BlockingStrategy(ABC):
    """Generates candidate entity pairs prior to full linking.

    Blocking reduces the O(n*m) comparison space between two datasets down
    to a smaller set of plausible candidate pairs.
    """

    @abstractmethod
    def candidate_pairs(
        self, dataset1: list[Entity], dataset2: list[Entity]
    ) -> list[tuple[Entity, Entity]]:
        """Return candidate entity pairs to be scored by a linker.

        Args:
            dataset1: Entities from the first dataset.
            dataset2: Entities from the second dataset.

        Returns:
            A list of candidate ``(entity1, entity2)`` pairs.
        """


def rank_top_matches(
    scores: dict[str, float],
    entities_by_id: dict[str, Entity],
    k: int,
    *,
    descending: bool,
    keep: Callable[[float], bool] | None = None,
) -> list[Entity]:
    """Rank ``scores`` (entity id -> score) and return the top ``k`` matching entities.

    Shared by blocking strategies that score every candidate against a
    source entity and keep only the best few
    (`~linkingtk.blocking.label_overlap.LabelOverlap`,
    `~linkingtk.blocking.embedding.EmbeddingSimilarityBlocker`). Ties are
    broken by entity id for deterministic output.

    Args:
        scores: Candidate entity id -> score.
        entities_by_id: Entity id -> ``Entity``, used to resolve the kept
            ids back to entities.
        k: Maximum number of entities to return.
        descending: Whether higher scores are better (e.g. similarity) or
            lower scores are better (e.g. distance).
        keep: Optional predicate; candidates for which it returns
            ``False`` are dropped before ranking.

    Returns:
        Up to ``k`` entities, best-scoring first.
    """
    items: Iterable[tuple[str, float]] = scores.items()
    if keep is not None:
        items = [item for item in items if keep(item[1])]
    sign = -1 if descending else 1
    top = heapq.nsmallest(k, items, key=lambda item: (sign * item[1], item[0]))
    return [entities_by_id[entity_id] for entity_id, _ in top]

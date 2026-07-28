"""Abstract interface for blocking strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod

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

"""Fuzzy blocking strategy based on token overlap between labels.

Planned for Phase 2 (see DESIGN.md milestones); not yet implemented.
"""

from __future__ import annotations

from linkingtk.blocking.base import BlockingStrategy
from linkingtk.core.entity import Entity


class LabelOverlap(BlockingStrategy):
    """Blocks entities whose labels share a minimum proportion of tokens."""

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold

    def candidate_pairs(
        self, dataset1: list[Entity], dataset2: list[Entity]
    ) -> list[tuple[Entity, Entity]]:
        raise NotImplementedError("LabelOverlap blocking is not yet implemented.")

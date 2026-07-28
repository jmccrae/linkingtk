"""Fuzzy blocking strategy based on embedding similarity.

Planned for a later phase (see DESIGN.md milestones); not yet implemented.
Requires the ``embeddings`` extra (``transformers``).
"""

from __future__ import annotations

from linkingtk.blocking.base import BlockingStrategy
from linkingtk.core.entity import Entity


class EmbeddingSimilarityBlocker(BlockingStrategy):
    """Blocks entities whose label/context embeddings are close in vector space."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        top_k: int = 10,
    ) -> None:
        self.model_name = model_name
        self.top_k = top_k

    def candidate_pairs(
        self, dataset1: list[Entity], dataset2: list[Entity]
    ) -> list[tuple[Entity, Entity]]:
        raise NotImplementedError("EmbeddingSimilarityBlocker is not yet implemented.")

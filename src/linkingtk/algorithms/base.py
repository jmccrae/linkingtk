"""Abstract interface implemented by every linking algorithm."""

from __future__ import annotations

from abc import ABC, abstractmethod

from linkingtk.blocking.base import BlockingStrategy
from linkingtk.blocking.exact import ExactMatch
from linkingtk.core.entity import Entity
from linkingtk.core.result import AlignmentResult
from linkingtk.core.source import EntitySource
from linkingtk.utils.graph import Graph

DEFAULT_BLOCKING = ExactMatch()


class BaseLinker(ABC):
    """Common interface for Entity Alignment, Entity Linking, WSD and WSA.

    Subclasses implement a single task by consuming two entity datasets
    (whose shape depends on the task, see DESIGN.md), an optional
    supporting graph, and a blocking strategy used to restrict the
    candidate search space.
    """

    @abstractmethod
    def link(
        self,
        dataset1: list[Entity],
        dataset2: list[Entity] | EntitySource,
        graph: Graph = None,
        blocking: BlockingStrategy = DEFAULT_BLOCKING,
    ) -> list[AlignmentResult]:
        """Link entities in ``dataset1`` to entities in ``dataset2``.

        Args:
            dataset1: Source entities (e.g. mentions, source KG entities).
            dataset2: Target entities (e.g. KB entries, target KG entities),
                or an [EntitySource][linkingtk.core.source.EntitySource]
                wrapping a target too large to materialize as a
                ``list[Entity]``. Support depends on ``blocking`` -- see
                [candidate_pairs][linkingtk.blocking.base.BlockingStrategy.candidate_pairs].
            graph: Optional supporting graph as a triple list, a
                ``networkx.Graph``, or an ``rdflib.Graph``.
            blocking: Strategy used to generate candidate pairs before
                scoring.

        Returns:
            A list of predicted links.
        """

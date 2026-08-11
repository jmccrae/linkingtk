"""Abstract interface implemented by every linking algorithm."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Union

from linkingtk.blocking.base import BlockingStrategy
from linkingtk.blocking.exact import ExactMatch
from linkingtk.core.entity import Entity
from linkingtk.core.result import AlignmentResult

if TYPE_CHECKING:
    import networkx as nx
    import rdflib

Triple = tuple[str, str, str]
Graph = Union[list[Triple], "nx.Graph", "rdflib.Graph", None]

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
        dataset2: list[Entity],
        graph: Graph = None,
        blocking: BlockingStrategy = DEFAULT_BLOCKING,
    ) -> list[AlignmentResult]:
        """Link entities in ``dataset1`` to entities in ``dataset2``.

        Args:
            dataset1: Source entities (e.g. mentions, source KG entities).
            dataset2: Target entities (e.g. KB entries, target KG entities).
            graph: Optional supporting graph as a triple list, a
                ``networkx.Graph``, or an ``rdflib.Graph``.
            blocking: Strategy used to generate candidate pairs before
                scoring.

        Returns:
            A list of predicted links.
        """

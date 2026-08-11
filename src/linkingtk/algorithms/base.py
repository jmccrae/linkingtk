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


def rank_candidates(candidates: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """Sort ``(target_id, score)`` candidates by descending score, id tie-break."""
    return sorted(candidates, key=lambda item: (-item[1], item[0]))


def greedy_matches(
    candidates_by_source: dict[str, list[tuple[str, float]]],
) -> list[AlignmentResult]:
    """Build one :class:`AlignmentResult` per source, each taking its highest-ranked candidate.

    Shared by linkers that score every candidate independently and pick
    each source's best match without a global one-to-one constraint
    (`~linkingtk.algorithms.string_similarity.StringSimilarityLinker`,
    `~linkingtk.algorithms.feature_classifier.FeatureClassifierLinker`).

    Args:
        candidates_by_source: Source id -> list of ``(target_id, score)``.

    Returns:
        One ``AlignmentResult`` per source id, with the rest of its
        ranked candidates as ``alternatives``.
    """
    results = []
    for source_id, candidates in candidates_by_source.items():
        ranked = rank_candidates(candidates)
        best_id, best_score = ranked[0]
        results.append(
            AlignmentResult(
                source_id=source_id,
                target_id=best_id,
                score=best_score,
                alternatives=[target_id for target_id, _ in ranked[1:]],
            )
        )
    return results

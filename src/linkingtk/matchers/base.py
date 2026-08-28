"""Abstract interface for matching strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod

from linkingtk.core.result import AlignmentResult


def _rank_candidates(candidates: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """Sort ``(target_id, score)`` candidates by descending score.

    Ties keep ``candidates``' original relative order (Python's ``sorted``
    is stable) rather than breaking them by ``target_id`` -- for an
    ``EntitySource``-backed candidate set this order is meaningful (e.g.
    [WnEntitySource.search][linkingtk.sources.wn.WnEntitySource.search]
    returns a lemma's senses in the lexicon's own most-frequent-sense-first
    order), so a plain alphabetical id tie-break previously discarded it
    and could surface an off-topic rare sense over the frequency-preferred
    one whenever every candidate scored the same -- e.g. a Lesk-style
    linker's zero-context-overlap case, where every candidate ties at
    ``0.0`` (issue #66).
    """
    return sorted(candidates, key=lambda item: -item[1])


class Matcher(ABC):
    """Turns per-source scored candidates into final
    [AlignmentResult][linkingtk.core.result.AlignmentResult]s.

    Not required to return exactly one result per source — e.g. a future
    hierarchical matcher could return several (broader/narrower-related)
    results for one source entity. Shared across linkers that produce a
    ``dict[str, list[tuple[target_id, score]]]`` scored candidate map
    ([StringSimilarityLinker][linkingtk.algorithms.string_similarity.StringSimilarityLinker],
    [FeatureClassifierLinker][linkingtk.algorithms.feature_classifier.FeatureClassifierLinker]).

    Note this only leaves room for *output* cardinality to change. A real
    broader/narrower matcher would also need relation-type/taxonomy
    information that ``match()``'s current input (plain target-id/score
    pairs, no ``Entity`` or graph access) doesn't carry — that's a
    deliberately deferred, separate signature change, not something this
    abstraction already supports.
    """

    @abstractmethod
    def match(
        self, candidates_by_source: dict[str, list[tuple[str, float]]]
    ) -> list[AlignmentResult]:
        """Resolve scored candidates into final links.

        Args:
            candidates_by_source: Source id -> list of ``(target_id, score)``.

        Returns:
            Predicted links.
        """

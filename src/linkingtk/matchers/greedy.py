"""Independent per-source argmax matching."""

from __future__ import annotations

from linkingtk.core.result import AlignmentResult
from linkingtk.matchers.base import Matcher, _rank_candidates


class GreedyMatcher(Matcher):
    """Picks each source entity's highest-scoring candidate independently.

    Multiple source entities may map to the same target — there's no
    global one-to-one constraint. The simplest, cheapest matcher, and the
    default for both ``StringSimilarityLinker`` and
    ``FeatureClassifierLinker``.
    """

    def match(
        self, candidates_by_source: dict[str, list[tuple[str, float]]]
    ) -> list[AlignmentResult]:
        results = []
        for source_id, candidates in candidates_by_source.items():
            ranked = _rank_candidates(candidates)
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

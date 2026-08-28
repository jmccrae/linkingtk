"""Globally optimal one-to-one assignment matching."""

from __future__ import annotations

from scipy.optimize import linear_sum_assignment

from linkingtk.core.result import AlignmentResult
from linkingtk.matchers.base import Matcher, _rank_candidates


class OptimalMatcher(Matcher):
    """Finds a single globally optimal one-to-one assignment.

    Uses the Hungarian algorithm (``scipy.optimize.linear_sum_assignment``)
    over a dense cost matrix — the right tradeoff at the scale this toolkit
    currently targets (blocking-reduced candidate sets on toy/small
    datasets), not thousands-of-entities KG-scale workloads (graph-
    embedding-based EA, a later milestone), which would need a sparse
    bipartite matching instead. Can outperform
    [GreedyMatcher][linkingtk.matchers.greedy.GreedyMatcher] when
    two source entities' individually-best candidate is the same target —
    see [EntMatcherLinker][linkingtk.algorithms.ea.entmatcher.EntMatcherLinker],
    which uses this matcher by default.
    """

    def match(
        self, candidates_by_source: dict[str, list[tuple[str, float]]]
    ) -> list[AlignmentResult]:
        source_ids = list(candidates_by_source.keys())
        if not source_ids:
            return []

        scores_by_pair: dict[tuple[str, str], float] = {
            (source_id, target_id): score
            for source_id, candidates in candidates_by_source.items()
            for target_id, score in candidates
        }
        target_ids = sorted({target_id for _, target_id in scores_by_pair})

        # The sentinel must be strictly worse than every real cost so a
        # source/target pair that was never an actual candidate is never
        # preferred over one that was. Scores aren't guaranteed to be in
        # [0, 1] here (e.g. StringSimilarityLinker's "word_overlap" metric
        # is an unbounded raw count, and a custom metric could be
        # anything), so the sentinel is derived from the actual scores
        # rather than assumed.
        worst_real_cost = max((1.0 - score for score in scores_by_pair.values()), default=0.0)
        sentinel_cost = worst_real_cost + 1.0

        cost_matrix = [
            [
                1.0 - score
                if (score := scores_by_pair.get((source_id, target_id))) is not None
                else sentinel_cost
                for target_id in target_ids
            ]
            for source_id in source_ids
        ]
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        results = []
        for row, col in zip(row_indices, col_indices, strict=True):
            source_id, target_id = source_ids[row], target_ids[col]
            if (source_id, target_id) not in scores_by_pair:
                continue  # matched to the sentinel: no real candidate available
            ranked = _rank_candidates(candidates_by_source[source_id])
            results.append(
                AlignmentResult(
                    source_id=source_id,
                    target_id=target_id,
                    score=scores_by_pair[(source_id, target_id)],
                    alternatives=[t for t, _ in ranked if t != target_id],
                )
            )
        return results

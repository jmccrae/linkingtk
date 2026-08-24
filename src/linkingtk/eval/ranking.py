"""Exhaustive, non-blocking ranked evaluation for EA linkers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Protocol

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity

if TYPE_CHECKING:
    import numpy.typing as npt

    from linkingtk.core.entity import Entity

_Metric = Literal["cosine", "manhattan", "inner"]


class _ScoringLinker(Protocol):
    """What [rank_exhaustive][linkingtk.eval.ranking.rank_exhaustive] needs from a fitted linker.

    Every linker in ``linkingtk.algorithms.ea`` implements this. Not a
    [BaseLinker][linkingtk.algorithms.base.BaseLinker] method -- "source"
    and "target" scoring vectors are an EA-linker-specific concept (EL/
    WSD/WSA linkers have no such thing), so this stays a structural
    protocol rather than growing the shared base interface.
    """

    def source_embedding(self, entity_id: str) -> npt.NDArray[np.floating[Any]]:
        """Vector used to score ``entity_id`` as a scored pair's source side."""

    def target_embedding(self, entity_id: str) -> npt.NDArray[np.floating[Any]]:
        """Vector used to score ``entity_id`` as a scored pair's target side."""


def rank_exhaustive(
    linker: _ScoringLinker,
    source_entities: list[Entity],
    target_entities: list[Entity],
    metric: _Metric = "cosine",
    csls_k: int = 0,
) -> list[tuple[str, list[str]]]:
    """Rank every ``source_entities`` entity against every ``target_entities`` entity.

    No blocking/candidate-restriction step at all -- a single dense
    similarity matrix, matching OpenEA's own ``greedy_alignment``
    methodology (``modules/finding/alignment.py``, itself a thin wrapper
    around ``modules/finding/similarity.py``'s ``sim``/``csls_sim``, ported
    here). Intended for benchmark scripts that need numbers directly
    comparable to a published target:
    [BlockingStrategy][linkingtk.blocking.base.BlockingStrategy]-restricted
    evaluation (the normal, production-scale path via
    [BaseLinker.link][linkingtk.algorithms.base.BaseLinker.link]) upper-bounds
    the reported metric by blocking's own recall, not purely by the
    linker's embedding/ranking quality -- see issue #37. Not intended for
    production-scale linking: this always computes a full
    ``len(source_entities) x len(target_entities)`` similarity matrix.

    Args:
        linker: A fitted EA linker exposing ``source_embedding``/
            ``target_embedding`` -- every linker in ``linkingtk.algorithms.ea``
            does.
        source_entities: Entities to rank from.
        target_entities: Entities to rank against.
        metric: Similarity metric -- ``"cosine"`` (default, this
            function's long-standing behavior), ``"manhattan"``
            (``1 - cityblock_distance``, unnormalized), or ``"inner"``
            (raw dot product, unnormalized). **Different EA methods'
            own OpenEA configs use different metrics for their published
            numbers** -- e.g. GCN-Align's and RDGCN's own
            ``eval_metric: "manhattan"``, ``eval_norm: false`` (confirmed
            by reading ``run/args/{gcnalign,rdgcn}_args_15K.json``
            directly) -- so a benchmark script targeting a specific
            published number should pass the metric *that method's own
            config* specifies, not assume cosine. Confirmed empirically
            on RDGCN (#43): switching from cosine to manhattan+csls_k=10
            closed a published-number gap from ~88% down to ~97% relative
            Hits@1.
        csls_k: If ``> 0``, applies Cross-domain Similarity Local Scaling
            (Lample et al. 2018) with this many nearest neighbors --
            ports OpenEA's own ``csls_sim``, which several EA methods'
            configs enable (typically ``10``) for their published numbers.
            ``0`` (default) disables it, this function's long-standing
            behavior.

    Returns:
        ``(source_id, ranked_target_ids)`` pairs, best match first, one
        entry per ``source_entities``. Each list contains **every**
        ``target_entities`` id -- never truncated, since
        [Evaluator.evaluate_ranked][linkingtk.eval.evaluator.Evaluator.evaluate_ranked]
        needs the true target's real rank for MRR even when it falls
        outside any Hits@k cutoff; truncating here would silently
        reintroduce a version of the denominator/miss bug that function's
        own docstring documents fixing. ``top_k`` cutoffs are
        ``evaluate_ranked``'s job, not this function's.
    """
    if not source_entities or not target_entities:
        return [(entity.id, []) for entity in source_entities]

    source_matrix = np.stack([linker.source_embedding(entity.id) for entity in source_entities])
    target_matrix = np.stack([linker.target_embedding(entity.id) for entity in target_entities])
    similarities = _similarity_matrix(source_matrix, target_matrix, metric)
    if csls_k > 0:
        similarities = _csls(similarities, csls_k)
    target_ids = [entity.id for entity in target_entities]

    order = np.argsort(-similarities, axis=1)
    return [
        (entity.id, [target_ids[i] for i in order[row]])
        for row, entity in enumerate(source_entities)
    ]


def _similarity_matrix(
    source_matrix: npt.NDArray[np.floating[Any]],
    target_matrix: npt.NDArray[np.floating[Any]],
    metric: _Metric,
) -> npt.NDArray[np.floating[Any]]:
    if metric == "cosine":
        result: npt.NDArray[np.floating[Any]] = cosine_similarity(source_matrix, target_matrix)
        return result
    if metric == "manhattan":
        manhattan: npt.NDArray[np.floating[Any]] = 1 - cdist(
            source_matrix, target_matrix, metric="cityblock"
        )
        return manhattan
    if metric == "inner":
        inner: npt.NDArray[np.floating[Any]] = source_matrix @ target_matrix.T
        return inner
    raise ValueError(
        f"Unsupported metric {metric!r} -- expected 'cosine', 'manhattan', or 'inner'."
    )


def _csls(similarities: npt.NDArray[np.floating[Any]], k: int) -> npt.NDArray[np.floating[Any]]:
    """Cross-domain Similarity Local Scaling, per OpenEA's own ``csls_sim``.

    ``csls[i, j] = 2 * sim[i, j] - mean_of_j's_k_nearest_to_i -
    mean_of_i's_k_nearest_to_j`` -- corrects for "hub" points that would
    otherwise dominate nearest-neighbor search from many rows/columns at
    once. Ported to match OpenEA's own ``calculate_nearest_k`` exactly,
    including its ``k + 1`` partition point (only the first ``k`` of the
    ``k + 1``-partitioned values are actually averaged -- an OpenEA
    quirk, not a rounding choice made here).
    """

    def nearest_k_mean(matrix: npt.NDArray[np.floating[Any]]) -> npt.NDArray[np.floating[Any]]:
        partition_point = min(k + 1, matrix.shape[1] - 1)
        sorted_matrix = -np.partition(-matrix, partition_point, axis=1)
        mean: npt.NDArray[np.floating[Any]] = np.mean(sorted_matrix[:, :k], axis=1)
        return mean

    nearest1 = nearest_k_mean(similarities)
    nearest2 = nearest_k_mean(similarities.T)
    result: npt.NDArray[np.floating[Any]] = (2 * similarities.T - nearest1).T - nearest2
    return result

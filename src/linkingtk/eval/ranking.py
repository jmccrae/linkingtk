"""Exhaustive, non-blocking ranked evaluation for EA linkers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

if TYPE_CHECKING:
    import numpy.typing as npt

    from linkingtk.core.entity import Entity


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
) -> list[tuple[str, list[str]]]:
    """Rank every ``source_entities`` entity against every ``target_entities`` entity.

    No blocking/candidate-restriction step at all -- a single dense
    cosine-similarity matrix, matching OpenEA's own ``greedy_alignment``
    methodology (``modules/finding/alignment.py``). Intended for benchmark
    scripts that need numbers directly comparable to a published target:
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
    similarities = cosine_similarity(source_matrix, target_matrix)
    target_ids = [entity.id for entity in target_entities]

    order = np.argsort(-similarities, axis=1)
    return [
        (entity.id, [target_ids[i] for i in order[row]])
        for row, entity in enumerate(source_entities)
    ]

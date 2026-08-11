"""Hard-negative sampling for training classical ML linkers.

Mines the most *confusable* non-matching candidates per source entity from
a :class:`~linkingtk.blocking.base.BlockingStrategy`, rather than sampling
negatives uniformly at random from the full candidate pool — a blocking
strategy already considered these plausible enough to surface, so they're
more informative training signal for a classifier
(`~linkingtk.algorithms.feature_classifier.FeatureClassifierLinker.fit`)
than a random draw. Shaped as a plain function over ``Entity``/
``BlockingStrategy`` (no dependency on ``TrainingArguments`` or anything
``Trainer``-specific) so it's equally usable by
`~linkingtk.train.Trainer` once that's implemented.
"""

from __future__ import annotations

import logging
from collections import defaultdict

from linkingtk.blocking.base import BlockingStrategy
from linkingtk.core.entity import Entity

logger = logging.getLogger("linkingtk")


def sample_hard_negatives(
    dataset1: list[Entity],
    dataset2: list[Entity],
    ground_truth: list[tuple[str, str]],
    blocking: BlockingStrategy,
    top_k: int = 5,
) -> list[tuple[Entity, Entity]]:
    """Sample the top-K most confusable non-matching candidates per source entity.

    For each distinct source entity in ``ground_truth``, takes that
    entity's candidates from ``blocking.candidate_pairs(dataset1,
    dataset2)`` in the best-first order documented by
    :meth:`~linkingtk.blocking.base.BlockingStrategy.candidate_pairs`,
    excluding any candidate that is itself a true match for that source
    entity, up to ``top_k``.

    A source entity with fewer than ``top_k`` candidates (or none at all)
    simply contributes fewer negatives — this is normal partial coverage,
    not an error.

    Args:
        dataset1: Source entities.
        dataset2: Target entities.
        ground_truth: List of ``(source_id, target_id)`` true pairs.
        blocking: Strategy used to generate candidates to mine negatives from.
        top_k: Maximum number of hard negatives to keep per source entity.

    Returns:
        A flat list of ``(entity1, entity2)`` negative pairs, suitable as
        :meth:`~linkingtk.algorithms.feature_classifier.FeatureClassifierLinker.fit`'s
        ``negatives`` argument. If you're about to pass the result straight
        to ``fit(..., blocking=blocking)``, note ``blocking.candidate_pairs()``
        runs once here and again inside ``fit()`` — fine at this class's
        intended toy/small-dataset scale, but worth knowing if ``blocking``
        is itself expensive.
    """
    if not blocking.ranked:
        logger.warning(
            "sample_hard_negatives called with a blocking strategy that isn't "
            "marked as returning best-first candidates (ranked=False); negatives "
            "may not actually be the hardest available, just whichever candidates "
            "`blocking` happened to return first."
        )

    entities1_by_id = {entity.id: entity for entity in dataset1}
    ground_truth_set = set(ground_truth)

    candidates_by_source: dict[str, list[Entity]] = defaultdict(list)
    for entity1, entity2 in blocking.candidate_pairs(dataset1, dataset2):
        candidates_by_source[entity1.id].append(entity2)

    source_ids = dict.fromkeys(source_id for source_id, _ in ground_truth)
    negatives: list[tuple[Entity, Entity]] = []
    for source_id in source_ids:
        source_entity = entities1_by_id.get(source_id)
        if source_entity is None:
            continue
        hard = [
            entity2
            for entity2 in candidates_by_source.get(source_id, [])
            if (source_id, entity2.id) not in ground_truth_set
        ][:top_k]
        negatives.extend((source_entity, entity2) for entity2 in hard)

    if not negatives:
        logger.warning(
            "sample_hard_negatives found zero negative candidates across %d "
            "ground-truth source entities; blocking may be too narrow (e.g. "
            "top_k/max_matches too small, or threshold too strict) to have "
            "anything but true matches to mine.",
            len(source_ids),
        )
    return negatives

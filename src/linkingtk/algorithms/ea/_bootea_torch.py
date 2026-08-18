"""Torch-touching training-step helpers for
[BootEALinker][linkingtk.algorithms.ea.bootea.BootEALinker].

Separated from ``_bootea_training.py`` (which stays plain-numpy and
independently testable without ``torch`` installed) because these
functions build/consume PyTorch tensors and modules directly. Callers must
already have confirmed ``torch`` is importable (``bootea.py``'s ``fit()``
does this via ``OptionalDependencyError`` before any of these run) --
that's why these import ``torch`` unconditionally rather than guarding
again.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from linkingtk.algorithms.ea._bootea_training import (
    encode_triples,
    sample_truncated_negative_triples,
)

if TYPE_CHECKING:
    import numpy.typing as npt
    import torch


class KGContext(NamedTuple):
    """One KG's own id-mapped triples plus everything derived from them.

    Unlike [IPTransELinker][linkingtk.algorithms.ea.iptranse.IPTransELinker]'s
    ``KGContext``, ``triples``/``real_triples`` here are built *after*
    seed-pair pseudo-triples have already been folded in (see
    ``bootea.py``'s ``fit()``) -- matching OpenEA's own ``KG`` object,
    which mutates ``rt_dict``/``hr_dict``/``relation_triples_set`` in
    place via ``add_sup_relation_triples`` at ``KGs`` construction time, so
    every later consumer (including future bootstrap rounds) already sees
    the seed pseudo-triples baked in.
    """

    triples: npt.NDArray[np.int64]
    real_triples: npt.NDArray[np.int64]
    """Sorted, [encode_triples][linkingtk.algorithms.ea._bootea_training.encode_triples]-encoded
    keys -- see
    [sample_truncated_negative_triples][linkingtk.algorithms.ea._bootea_training.sample_truncated_negative_triples]."""
    entity_pool: npt.NDArray[np.int64]
    by_head: dict[int, list[tuple[int, int]]]
    by_tail: dict[int, list[tuple[int, int]]]


def build_kg_context(mapped: npt.NDArray[np.int64]) -> KGContext:
    """Everything one KG's structural training step needs, from its own id-mapped triples."""
    real_triples = np.unique(encode_triples(mapped))
    entity_pool = (
        np.unique(np.concatenate([mapped[:, 0], mapped[:, 2]]))
        if len(mapped)
        else np.empty(0, dtype=np.int64)
    )
    by_head: dict[int, list[tuple[int, int]]] = defaultdict(list)
    by_tail: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for h, r, t in mapped:
        by_head[int(h)].append((int(r), int(t)))
        by_tail[int(t)].append((int(h), int(r)))
    return KGContext(
        triples=mapped,
        real_triples=real_triples,
        entity_pool=entity_pool,
        by_head=by_head,
        by_tail=by_tail,
    )


def _scores(
    entity_embeds: torch.nn.Parameter,
    relation_embeds: torch.nn.Parameter,
    triples: npt.NDArray[np.int64],
) -> torch.Tensor:
    import torch
    import torch.nn.functional as functional

    device = entity_embeds.device
    t = torch.from_numpy(triples).long().to(device)
    h = functional.normalize(entity_embeds[t[:, 0]], dim=1)
    r = functional.normalize(relation_embeds[t[:, 1]], dim=1)
    tail = functional.normalize(entity_embeds[t[:, 2]], dim=1)
    result: torch.Tensor = ((h + r - tail) ** 2).sum(dim=1)
    return result


def _limited_loss(
    entity_embeds: torch.nn.Parameter,
    relation_embeds: torch.nn.Parameter,
    pos: npt.NDArray[np.int64],
    neg: npt.NDArray[np.int64],
    pos_margin: float,
    neg_margin: float,
    neg_margin_balance: float,
) -> torch.Tensor:
    """``sum(relu(pos_score - pos_margin)) + balance * sum(relu(neg_margin - neg_score))``.

    Ports ``limited_loss`` (``modules/base/losses.py``) -- a *double*-margin
    hinge, unlike
    [MTransELinker][linkingtk.algorithms.ea.mtranse.MTransELinker]'s/
    [IPTransELinker][linkingtk.algorithms.ea.iptranse.IPTransELinker]'s
    single-hinge margin loss: positive triples are only penalized once
    their score exceeds ``pos_margin`` (not driven all the way to ``0``),
    and negatives are only penalized once their score drops *below*
    ``neg_margin`` (not compared relative to the positive's own score).
    """
    import torch

    pos_score = _scores(entity_embeds, relation_embeds, pos)
    neg_score = _scores(entity_embeds, relation_embeds, neg)
    pos_loss = torch.relu(pos_score - pos_margin).sum()
    neg_loss = torch.relu(neg_margin - neg_score).sum()
    result: torch.Tensor = pos_loss + neg_margin_balance * neg_loss
    return result


def _alignment_loss(
    entity_embeds: torch.nn.Parameter,
    relation_embeds: torch.nn.Parameter,
    pos: npt.NDArray[np.int64],
) -> torch.Tensor:
    """``-sum(log(sigmoid(-score)))`` over the labeled-alignment set's pseudo-triples.

    Ports ``bootea.py``'s ``_define_alignment_graph``'s
    ``alignment_loss`` -- a positive-only sigmoid loss with **no negative
    sampling** (unlike the main structural loss): it simply pushes every
    pseudo-triple's ``||h+r-t||^2`` towards ``0`` via a smooth sigmoid
    penalty, since these pairs are already (algorithmically) confident
    matches, not raw supervision needing contrastive negatives.
    """
    import torch.nn.functional as functional

    score = _scores(entity_embeds, relation_embeds, pos)
    return -functional.logsigmoid(-score).sum()


def train_structural_epoch(
    entity_embeds: torch.nn.Parameter,
    relation_embeds: torch.nn.Parameter,
    optimizer: torch.optim.Optimizer,
    ctx1: KGContext,
    ctx2: KGContext,
    neighbors1: dict[int, npt.NDArray[np.int64]],
    neighbors2: dict[int, npt.NDArray[np.int64]],
    rng: np.random.Generator,
    batch_size: int,
    pos_margin: float,
    neg_margin: float,
    neg_margin_balance: float,
    neg_triple_num: int,
) -> None:
    """One epoch of "limited"-loss training with truncated (hard) negative sampling.

    Mirrors OpenEA's ``launch_triple_training_1epo``: one pass over both
    KGs' (already seed-pseudo-triple-augmented) triples, minibatches split
    proportionally by triple count, each positive triple corrupted
    ``neg_triple_num`` times via
    [sample_truncated_negative_triples][linkingtk.algorithms.ea._bootea_training.sample_truncated_negative_triples]
    (candidates restricted to that entity's current truncated-neighbor
    set, falling back to the KG's full entity pool).
    """
    total_triples = len(ctx1.triples) + len(ctx2.triples)
    steps = max(1, math.ceil(total_triples / batch_size))
    batch1 = round(len(ctx1.triples) / total_triples * batch_size) if total_triples else 0
    batch2 = batch_size - batch1
    perm1 = rng.permutation(len(ctx1.triples))
    perm2 = rng.permutation(len(ctx2.triples))

    for step in range(steps):
        idx1 = perm1[step * batch1 : (step + 1) * batch1]
        idx2 = perm2[step * batch2 : (step + 1) * batch2]
        if len(idx1) + len(idx2) == 0:
            continue
        pos = np.concatenate([ctx1.triples[idx1], ctx2.triples[idx2]], axis=0)

        neg_parts = []
        for _ in range(neg_triple_num):
            neg1 = sample_truncated_negative_triples(
                ctx1.triples[idx1], neighbors1, ctx1.entity_pool, ctx1.real_triples, rng
            )
            neg2 = sample_truncated_negative_triples(
                ctx2.triples[idx2], neighbors2, ctx2.entity_pool, ctx2.real_triples, rng
            )
            neg_parts.append(np.concatenate([neg1, neg2], axis=0))
        neg = np.concatenate(neg_parts, axis=0)

        # `pos` stays unrepeated (one pos_loss term per real positive); `neg`
        # has `neg_triple_num` corruptions per positive (`neg_triple_num`
        # neg_loss terms each) -- matches OpenEA's own asymmetric batch
        # shapes in `limited_loss`, not a tiled/matched pos/neg pairing.
        loss = _limited_loss(
            entity_embeds, relation_embeds, pos, neg, pos_margin, neg_margin, neg_margin_balance
        )
        optimizer.zero_grad()
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()


def train_alignment_epoch(
    entity_embeds: torch.nn.Parameter,
    relation_embeds: torch.nn.Parameter,
    optimizer: torch.optim.Optimizer,
    triples1: npt.NDArray[np.int64],
    triples2: npt.NDArray[np.int64],
    batch_size: int,
) -> None:
    """One epoch of the separate alignment loss over the labeled-alignment set's pseudo-triples.

    Mirrors ``train_alignment``: retrains fresh over *all* currently
    labeled pairs' pseudo-triples each outer iteration (not just this
    round's newly-added ones), split proportionally between the two KGs'
    sides, no-op if both sides are empty (no labeled pairs yet).
    """
    total = len(triples1) + len(triples2)
    if total == 0:
        return
    steps = max(1, math.ceil(total / batch_size))
    batch1 = round(len(triples1) / total * batch_size)
    batch2 = batch_size - batch1

    for step in range(steps):
        part1 = triples1[step * batch1 : (step + 1) * batch1]
        part2 = triples2[step * batch2 : (step + 1) * batch2]
        if len(part1) + len(part2) == 0:
            continue
        pos = np.concatenate([part1, part2], axis=0)
        loss = _alignment_loss(entity_embeds, relation_embeds, pos)
        optimizer.zero_grad()
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()


def validation_hits1(
    embeds: npt.NDArray[np.floating[Any]],
    entity_to_id: dict[str, int],
    val_pairs: list[tuple[str, str]],
) -> float:
    """Cheap top-1 match rate over ``val_pairs``, for early-stopping only.

    Same role as
    [IPTransELinker][linkingtk.algorithms.ea.iptranse.IPTransELinker]'s
    own ``validation_hits1`` -- BootEA has no learned mapping either,
    alignment signal lives directly in the shared embedding space.
    """
    sources = [s for s, _ in val_pairs]
    targets = [t for _, t in val_pairs]
    source_matrix = np.stack([embeds[entity_to_id[s]] for s in sources])
    target_matrix = np.stack([embeds[entity_to_id[t]] for t in targets])
    similarities = cosine_similarity(source_matrix, target_matrix)
    predicted = np.argmax(similarities, axis=1)
    correct = sum(1 for i, j in enumerate(predicted) if j == i)
    return correct / len(val_pairs)

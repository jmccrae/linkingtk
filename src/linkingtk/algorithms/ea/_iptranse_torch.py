"""Torch-touching training-step helpers for
[IPTransELinker][linkingtk.algorithms.ea.iptranse.IPTransELinker].

Separated from ``_iptranse_training.py`` (which stays plain-numpy and
independently testable without ``torch`` installed) because these functions
build/consume PyTorch tensors and modules directly. Callers must already
have confirmed ``torch`` is importable (``iptranse.py``'s ``fit()`` does
this via ``OptionalDependencyError`` before any of these run) -- that's why
these import ``torch`` unconditionally rather than guarding again.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from linkingtk.algorithms.ea._iptranse_training import (
    find_new_pairs,
    generate_two_step_paths,
    pseudo_triples_for_pair,
    sample_negative_path_relations,
    sample_negative_triples,
)
from linkingtk.utils.graph import Triple

if TYPE_CHECKING:
    import numpy.typing as npt
    import torch


class KGContext(NamedTuple):
    """One KG's own id-mapped triples plus everything derived from them."""

    triples: npt.NDArray[np.int64]
    real_triples: set[tuple[int, int, int]]
    entity_pool: npt.NDArray[np.int64]
    relation_pool: npt.NDArray[np.int64]
    paths: npt.NDArray[np.int64]
    by_head: dict[int, list[tuple[int, int]]]
    by_tail: dict[int, list[tuple[int, int]]]


def build_kg_context(
    triples_labels: list[Triple],
    mapped: npt.NDArray[np.int64],
    relation_to_id: dict[str, int],
) -> KGContext:
    """Everything one KG's training step needs, derived once at `fit()` start."""
    real_triples = {(int(h), int(r), int(t)) for h, r, t in mapped}
    entity_pool = (
        np.unique(np.concatenate([mapped[:, 0], mapped[:, 2]]))
        if len(mapped)
        else np.empty(0, dtype=np.int64)
    )
    relation_pool = np.unique(mapped[:, 1]) if len(mapped) else np.empty(0, dtype=np.int64)
    by_head: dict[int, list[tuple[int, int]]] = defaultdict(list)
    by_tail: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for h, r, t in mapped:
        by_head[int(h)].append((int(r), int(t)))
        by_tail[int(t)].append((int(h), int(r)))

    path_labels = generate_two_step_paths(triples_labels)
    paths = (
        np.array(
            [
                (relation_to_id[r_x], relation_to_id[r_y], relation_to_id[r], weight)
                for r_x, r_y, r, weight in path_labels
            ],
            dtype=np.int64,
        )
        if path_labels
        else np.empty((0, 4), dtype=np.int64)
    )
    return KGContext(
        triples=mapped,
        real_triples=real_triples,
        entity_pool=entity_pool,
        relation_pool=relation_pool,
        paths=paths,
        by_head=by_head,
        by_tail=by_tail,
    )


def _margin_loss(
    entity_embeds: torch.nn.Parameter,
    relation_embeds: torch.nn.Parameter,
    pos: npt.NDArray[np.int64],
    neg: npt.NDArray[np.int64],
    margin: float,
) -> torch.Tensor:
    """``sum(relu(||h+r-t||^2 + margin - ||h'+r-t'||^2))`` over a (pos, neg) triple batch."""
    import torch
    import torch.nn.functional as functional

    device = entity_embeds.device
    pos_t = torch.from_numpy(pos).long().to(device)
    neg_t = torch.from_numpy(neg).long().to(device)
    h = functional.normalize(entity_embeds[pos_t[:, 0]], dim=1)
    r = functional.normalize(relation_embeds[pos_t[:, 1]], dim=1)
    t = functional.normalize(entity_embeds[pos_t[:, 2]], dim=1)
    nh = functional.normalize(entity_embeds[neg_t[:, 0]], dim=1)
    nr = functional.normalize(relation_embeds[neg_t[:, 1]], dim=1)
    nt = functional.normalize(entity_embeds[neg_t[:, 2]], dim=1)
    pos_score = ((h + r - t) ** 2).sum(dim=1)
    neg_score = ((nh + nr - nt) ** 2).sum(dim=1)
    loss: torch.Tensor = torch.relu(pos_score + margin - neg_score).sum()
    return loss


def train_epoch(
    entity_embeds: torch.nn.Parameter,
    relation_embeds: torch.nn.Parameter,
    optimizer: torch.optim.Optimizer,
    ctx1: KGContext,
    ctx2: KGContext,
    rng: np.random.Generator,
    batch_size: int,
    margin: float,
    path_parm: float,
) -> None:
    """One epoch of joint margin triple-loss + relation-composition path-loss training.

    Mirrors OpenEA's ``launch_ptranse_training_1epo``: one pass over both
    KGs' triples (minibatches split proportionally by triple count),
    negative-sampled per-KG, plus a proportionally-split path-loss batch
    each step, single joint backward/step per minibatch.
    """
    total_triples = len(ctx1.triples) + len(ctx2.triples)
    steps = max(1, math.ceil(total_triples / batch_size))
    batch1 = round(len(ctx1.triples) / total_triples * batch_size) if total_triples else 0
    batch2 = batch_size - batch1
    perm1 = rng.permutation(len(ctx1.triples))
    perm2 = rng.permutation(len(ctx2.triples))

    total_paths = len(ctx1.paths) + len(ctx2.paths)
    path_batch = total_paths // steps if steps else 0
    path_batch1 = round(len(ctx1.paths) / total_paths * path_batch) if total_paths else 0
    path_batch2 = path_batch - path_batch1

    for step in range(steps):
        idx1 = perm1[step * batch1 : (step + 1) * batch1]
        idx2 = perm2[step * batch2 : (step + 1) * batch2]
        if len(idx1) + len(idx2) == 0:
            continue
        pos = np.concatenate([ctx1.triples[idx1], ctx2.triples[idx2]], axis=0)
        neg1 = sample_negative_triples(ctx1.triples[idx1], ctx1.entity_pool, ctx1.real_triples, rng)
        neg2 = sample_negative_triples(ctx2.triples[idx2], ctx2.entity_pool, ctx2.real_triples, rng)
        neg = np.concatenate([neg1, neg2], axis=0)
        loss = _margin_loss(entity_embeds, relation_embeds, pos, neg, margin)

        if total_paths and (path_batch1 + path_batch2) > 0:
            loss = loss + path_parm * _path_loss(
                relation_embeds, ctx1, ctx2, path_batch1, path_batch2, rng, margin
            )

        optimizer.zero_grad()
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()


def _path_loss(
    relation_embeds: torch.nn.Parameter,
    ctx1: KGContext,
    ctx2: KGContext,
    path_batch1: int,
    path_batch2: int,
    rng: np.random.Generator,
    margin: float,
) -> torch.Tensor:
    """Relation-composition loss for one minibatch: ``r_x + r_y`` should approximate ``r``.

    Sampled with replacement (unlike the main triple loss's shuffled
    per-epoch slicing) -- a deliberate simplification of OpenEA's own
    ``random.sample`` calls, which assume a large enough path population;
    this avoids crashing on the tiny path counts a small test fixture can
    produce.
    """
    import torch
    import torch.nn.functional as functional

    device = relation_embeds.device
    empty = np.empty(0, dtype=np.int64)
    idxp1 = rng.integers(0, len(ctx1.paths), size=path_batch1) if path_batch1 else empty
    idxp2 = rng.integers(0, len(ctx2.paths), size=path_batch2) if path_batch2 else empty
    pos_paths = np.concatenate([ctx1.paths[idxp1], ctx2.paths[idxp2]], axis=0)
    if len(pos_paths) == 0:
        return torch.tensor(0.0, device=device)

    neg_paths1 = sample_negative_path_relations(ctx1.paths[idxp1], ctx1.relation_pool, rng)
    neg_paths2 = sample_negative_path_relations(ctx2.paths[idxp2], ctx2.relation_pool, rng)
    neg_paths = np.concatenate([neg_paths1, neg_paths2], axis=0)

    pos_paths_t = torch.from_numpy(pos_paths).long().to(device)
    neg_paths_t = torch.from_numpy(neg_paths).long().to(device)
    prx = functional.normalize(relation_embeds[pos_paths_t[:, 0]], dim=1)
    pry = functional.normalize(relation_embeds[pos_paths_t[:, 1]], dim=1)
    pr = functional.normalize(relation_embeds[pos_paths_t[:, 2]], dim=1)
    nrx = functional.normalize(relation_embeds[neg_paths_t[:, 0]], dim=1)
    nry = functional.normalize(relation_embeds[neg_paths_t[:, 1]], dim=1)
    nr = functional.normalize(relation_embeds[neg_paths_t[:, 2]], dim=1)
    pos_path_score = ((prx + pry - pr) ** 2).sum(dim=1)
    neg_path_score = ((nrx + nry - nr) ** 2).sum(dim=1)
    inv_weight = 1.0 / pos_paths_t[:, 3].float()
    loss: torch.Tensor = (inv_weight * torch.relu(pos_path_score + margin - neg_path_score)).sum()
    return loss


def bootstrap_round(
    entity_embeds: torch.nn.Parameter,
    relation_embeds: torch.nn.Parameter,
    alignment_optimizer: torch.optim.Optimizer,
    ctx1: KGContext,
    ctx2: KGContext,
    pool1_ids: npt.NDArray[np.int64],
    pool2_ids: npt.NDArray[np.int64],
    combined_entity_pool: npt.NDArray[np.int64],
    rng: np.random.Generator,
    batch_size: int,
    margin: float,
    sim_th: float,
) -> None:
    """One unsupervised self-training round: find new pairs, train on their pseudo-triples.

    Mirrors OpenEA's ``launch_alignment_training_1epo``: compute
    similarity between every not-yet-seeded entity on each side, keep
    each row's best match if it clears ``sim_th`` (see
    ``find_new_pairs``), turn each newly-found pair's real edges into
    weighted pseudo-triples, and train one epoch of weighted margin loss
    over them via ``alignment_optimizer`` (kept separate from the main
    triple+path optimizer, matching OpenEA's two-optimizer split).
    """
    import torch
    import torch.nn.functional as functional

    if len(pool1_ids) == 0 or len(pool2_ids) == 0:
        return

    with torch.no_grad():
        current = functional.normalize(entity_embeds, dim=1).cpu().numpy()
    sim_mat = current[pool1_ids] @ current[pool2_ids].T
    pairs = find_new_pairs(sim_mat, sim_th)
    if not pairs:
        return

    newly: set[tuple[int, int, int, float]] = set()
    for i, j, weight in pairs:
        a = int(pool1_ids[i])
        b = int(pool2_ids[j])
        newly |= pseudo_triples_for_pair(a, b, weight, ctx1.by_head, ctx1.by_tail)
        newly |= pseudo_triples_for_pair(b, a, weight, ctx2.by_head, ctx2.by_tail)
    if not newly:
        return

    newly_list = sorted(newly)
    pseudo_triples = np.array([(h, r, t) for h, r, t, _ in newly_list], dtype=np.int64)
    weights = np.array([w for _, _, _, w in newly_list], dtype=np.float32)

    steps = max(1, math.ceil(len(pseudo_triples) / batch_size))
    for _ in range(steps):
        size = min(batch_size, len(pseudo_triples))
        batch_idx = rng.choice(len(pseudo_triples), size=size, replace=False)
        pos = pseudo_triples[batch_idx]
        batch_weights = weights[batch_idx]
        neg = sample_negative_triples(pos, combined_entity_pool, set(), rng)

        loss = _weighted_margin_loss(
            entity_embeds, relation_embeds, pos, neg, batch_weights, margin
        )
        alignment_optimizer.zero_grad()
        loss.backward()  # type: ignore[no-untyped-call]
        alignment_optimizer.step()


def _weighted_margin_loss(
    entity_embeds: torch.nn.Parameter,
    relation_embeds: torch.nn.Parameter,
    pos: npt.NDArray[np.int64],
    neg: npt.NDArray[np.int64],
    weights: npt.NDArray[np.float32],
    margin: float,
) -> torch.Tensor:
    """``sum(w * relu(||h+r-t||^2 + margin - ||h'+r-t'||^2))`` for bootstrapped pseudo-triples."""
    import torch
    import torch.nn.functional as functional

    device = entity_embeds.device
    pos_t = torch.from_numpy(pos).long().to(device)
    neg_t = torch.from_numpy(neg).long().to(device)
    h = functional.normalize(entity_embeds[pos_t[:, 0]], dim=1)
    r = functional.normalize(relation_embeds[pos_t[:, 1]], dim=1)
    t = functional.normalize(entity_embeds[pos_t[:, 2]], dim=1)
    nh = functional.normalize(entity_embeds[neg_t[:, 0]], dim=1)
    nr = functional.normalize(relation_embeds[neg_t[:, 1]], dim=1)
    nt = functional.normalize(entity_embeds[neg_t[:, 2]], dim=1)
    pos_score = ((h + r - t) ** 2).sum(dim=1)
    neg_score = ((nh + nr - nt) ** 2).sum(dim=1)
    w = torch.from_numpy(weights).float().to(device)
    loss: torch.Tensor = (w * torch.relu(pos_score + margin - neg_score)).sum()
    return loss


def validation_hits1(
    embeds: npt.NDArray[np.floating[Any]],
    entity_to_id: dict[str, int],
    val_pairs: list[tuple[str, str]],
) -> float:
    """Cheap top-1 match rate over ``val_pairs``, for early-stopping only.

    Same deliberately simplified patience-counter role as ``mtranse.py``'s
    ``_validation_hits1``, minus the mapping-matrix projection step --
    IPTransE has no learned mapping, alignment signal lives directly in
    the shared embedding space.
    """
    sources = [s for s, _ in val_pairs]
    targets = [t for _, t in val_pairs]
    source_matrix = np.stack([embeds[entity_to_id[s]] for s in sources])
    target_matrix = np.stack([embeds[entity_to_id[t]] for t in targets])
    similarities = cosine_similarity(source_matrix, target_matrix)
    predicted = np.argmax(similarities, axis=1)
    correct = sum(1 for i, j in enumerate(predicted) if j == i)
    return correct / len(val_pairs)

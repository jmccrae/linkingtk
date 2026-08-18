"""Torch-touching training-step helpers for
[SEALinker][linkingtk.algorithms.ea.sea.SEALinker].

Separated from the orchestrating ``sea.py`` the same way
``_iptranse_torch.py`` is split from ``iptranse.py`` -- these functions
build/consume PyTorch tensors directly. Callers must already have confirmed
``torch`` is importable (``sea.py``'s ``fit()`` does this via
``OptionalDependencyError`` before any of these run).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from linkingtk.algorithms.ea._iptranse_training import sample_negative_triples
from linkingtk.utils.graph import Triple

if TYPE_CHECKING:
    import numpy.typing as npt
    import torch


class SEAKGContext(NamedTuple):
    """One KG's own id-mapped triples plus its negative-sampling pool."""

    triples: npt.NDArray[np.int64]
    real_triples: set[tuple[int, int, int]]
    entity_pool: npt.NDArray[np.int64]


def build_kg_context(triples_labels: list[Triple], mapped: npt.NDArray[np.int64]) -> SEAKGContext:
    """Everything one KG's structural training step needs, derived once at `fit()` start."""
    del triples_labels
    real_triples = {(int(h), int(r), int(t)) for h, r, t in mapped}
    entity_pool = (
        np.unique(np.concatenate([mapped[:, 0], mapped[:, 2]]))
        if len(mapped)
        else np.empty(0, dtype=np.int64)
    )
    return SEAKGContext(triples=mapped, real_triples=real_triples, entity_pool=entity_pool)


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


def train_structural_epoch(
    entity_embeds: torch.nn.Parameter,
    relation_embeds: torch.nn.Parameter,
    optimizer: torch.optim.Optimizer,
    ctx1: SEAKGContext,
    ctx2: SEAKGContext,
    rng: np.random.Generator,
    batch_size: int,
    margin: float,
) -> None:
    """One epoch of margin-based structural TransE training, matching OpenEA's
    ``launch_triple_training_1epo``: minibatches proportionally split between
    both KGs' triples by triple count, negative-sampled per-KG (uniform,
    each triple's own KG's entity pool), one joint backward/step per
    minibatch. No path loss, no bootstrapping -- SEA's structural half is
    plain TransE.
    """
    total_triples = len(ctx1.triples) + len(ctx2.triples)
    if total_triples == 0:
        return
    steps = max(1, math.ceil(total_triples / batch_size))
    batch1 = round(len(ctx1.triples) / total_triples * batch_size)
    batch2 = batch_size - batch1
    perm1 = rng.permutation(len(ctx1.triples))
    perm2 = rng.permutation(len(ctx2.triples))

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

        optimizer.zero_grad()
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()


def train_mapping_epoch(
    entity_embeds: torch.nn.Parameter,
    mapping_mat_1: torch.nn.Parameter,
    mapping_mat_2: torch.nn.Parameter,
    optimizer: torch.optim.Optimizer,
    seed_source: torch.Tensor,
    seed_target: torch.Tensor,
    unlabeled_pool1: npt.NDArray[np.int64],
    unlabeled_pool2: npt.NDArray[np.int64],
    rng: np.random.Generator,
    triple_steps: int,
    alpha_1: float,
    alpha_2: float,
) -> None:
    """One epoch of SEA's supervised + semi-supervised mapping-loss training.

    Mirrors OpenEA's ``launch_mapping_training_1epo``: each of
    ``triple_steps`` minibatches samples ``len(seed_source) // triple_steps``
    labeled pairs and (independently, no pairing needed) ``len(pool) //
    triple_steps`` unlabeled entities per side, and computes:

    - supervised: ``mat_1``-mapped source vs. real target (and
      symmetrically ``mat_2``-mapped target vs. real source);
    - semi-supervised (cycle-consistency): an unlabeled KG1 entity mapped
      through ``mat_1`` then ``mat_2`` should return to itself (and
      symmetrically for KG2 through ``mat_2`` then ``mat_1``) -- this term
      needs no ground-truth pairing, just entity ids from both sides.

    ``alpha_1``/``alpha_2`` weight the two terms. If either unlabeled pool
    is empty, the semi-supervised term is skipped entirely for this epoch
    (documented, not silently wrong -- can happen on tiny toy fixtures
    where every entity is a seed pair).

    OpenEA's own ``eye_mat_1``/``eye_mat_2`` (an orthogonality regularizer
    used by MTransE's mapping loss) are defined but never referenced by
    SEA's actual loss -- not ported here, matching this milestone's
    "port actual, not apparent, reference behavior" precedent (BootEA's
    skipped ``likelihood()``, JAPE's inert ``sim_optimizer``).
    """
    import torch
    import torch.nn.functional as functional

    device = entity_embeds.device
    labeled_batch_size = max(1, len(seed_source) // triple_steps)
    unlabeled_batch_size1 = len(unlabeled_pool1) // triple_steps
    unlabeled_batch_size2 = len(unlabeled_pool2) // triple_steps
    pool1_t = torch.from_numpy(unlabeled_pool1).long().to(device) if len(unlabeled_pool1) else None
    pool2_t = torch.from_numpy(unlabeled_pool2).long().to(device) if len(unlabeled_pool2) else None

    for _ in range(triple_steps):
        batch_idx = torch.randint(0, len(seed_source), (labeled_batch_size,), device=device)
        labeled1 = functional.normalize(entity_embeds[seed_source[batch_idx]], dim=1)
        labeled2 = functional.normalize(entity_embeds[seed_target[batch_idx]], dim=1)
        mapped_12 = functional.normalize(labeled1 @ mapping_mat_1, dim=1)
        mapped_21 = functional.normalize(labeled2 @ mapping_mat_2, dim=1)
        sup_loss = ((labeled2 - mapped_12) ** 2).sum() + ((labeled1 - mapped_21) ** 2).sum()
        loss = alpha_1 * sup_loss

        if (
            pool1_t is not None
            and pool2_t is not None
            and unlabeled_batch_size1 > 0
            and unlabeled_batch_size2 > 0
        ):
            idx1 = pool1_t[torch.randint(0, len(pool1_t), (unlabeled_batch_size1,), device=device)]
            idx2 = pool2_t[torch.randint(0, len(pool2_t), (unlabeled_batch_size2,), device=device)]
            unlabeled1 = functional.normalize(entity_embeds[idx1], dim=1)
            unlabeled2 = functional.normalize(entity_embeds[idx2], dim=1)
            cycle_121 = functional.normalize((unlabeled1 @ mapping_mat_1) @ mapping_mat_2, dim=1)
            cycle_212 = functional.normalize((unlabeled2 @ mapping_mat_2) @ mapping_mat_1, dim=1)
            semi_loss = (
                ((unlabeled1 - cycle_121) ** 2).sum() + ((unlabeled2 - cycle_212) ** 2).sum()
            )
            loss = loss + alpha_2 * semi_loss

        optimizer.zero_grad()
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()


def validation_hits1(
    embeds: npt.NDArray[np.floating[Any]],
    mapping_mat_1: npt.NDArray[np.floating[Any]],
    entity_to_id: dict[str, int],
    val_pairs: list[tuple[str, str]],
) -> float:
    """Cheap top-1 match rate over ``val_pairs``, for early-stopping only.

    Same simplified patience-counter role as ``mtranse.py``'s
    ``_validation_hits1`` -- projects the KG1 side through
    ``mapping_mat_1``, scores the KG2 side raw, same asymmetry as
    ``SEALinker.source_embedding``/``target_embedding``.
    """
    sources = [s for s, _ in val_pairs]
    targets = [t for _, t in val_pairs]
    source_matrix = np.stack([embeds[entity_to_id[s]] for s in sources]) @ mapping_mat_1
    target_matrix = np.stack([embeds[entity_to_id[t]] for t in targets])
    similarities = cosine_similarity(source_matrix, target_matrix)
    predicted = np.argmax(similarities, axis=1)
    correct = sum(1 for i, j in enumerate(predicted) if j == i)
    return correct / len(val_pairs)

"""Torch-touching training-step helpers for
[KDCoELinker][linkingtk.algorithms.ea.kdcoe.KDCoELinker].

Separated from ``_kdcoe_text.py`` (which stays plain-numpy and
independently testable without ``torch`` installed) because these
functions build/consume PyTorch tensors and modules directly. Callers must
already have confirmed ``torch`` is importable (``kdcoe.py``'s ``fit()``
does this via ``OptionalDependencyError`` before any of these run) --
that's why these import ``torch`` unconditionally rather than guarding
again.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from linkingtk.algorithms.ea._iptranse_training import sample_negative_triples

if TYPE_CHECKING:
    import numpy.typing as npt
    import torch


class KGContext(NamedTuple):
    """One KG's own id-mapped triples plus its entity pool for negative sampling."""

    triples: npt.NDArray[np.int64]
    real_triples: set[tuple[int, int, int]]
    entity_pool: npt.NDArray[np.int64]


def build_kg_context(mapped: npt.NDArray[np.int64]) -> KGContext:
    """Everything one KG's structural training step needs, derived once at `fit()` start."""
    real_triples = {(int(h), int(r), int(t)) for h, r, t in mapped}
    entity_pool = (
        np.unique(np.concatenate([mapped[:, 0], mapped[:, 2]]))
        if len(mapped)
        else np.empty(0, dtype=np.int64)
    )
    return KGContext(triples=mapped, real_triples=real_triples, entity_pool=entity_pool)


def build_description_encoder(wv_dim: int, default_desc_length: int) -> torch.nn.Module:
    """A two-GRU/Conv1D/attention description-text encoder.

    Ports ``_define_desc_graph`` (kdcoe.py:308-368) with real
    ``torch.nn.GRU``/``torch.nn.Conv1d`` rather than hand-rolled recurrence.
    Input: a batch of entities' word-embedding sequences, already looked
    up from a pretrained word-vector table, shape ``(batch,
    default_desc_length, wv_dim)``. Output: one L2-normalized ``(batch,
    wv_dim)`` vector per entity.

    Pipeline: GRU -> Conv1D (``kernel_size=3``, ``padding='valid'`` in
    OpenEA's terms -- shrinks the sequence dimension by 2) -> tanh ->
    attention-reweight (not reduce) over the sequence -> a second GRU ->
    a second attention-reweight -> sum over the sequence -> dense + tanh
    -> L2-normalize. A factory function rather than a module-level
    ``torch.nn.Module`` subclass, since ``torch.nn.Module`` can't be
    named at module scope without ``torch`` installed (see the module
    docstring).
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional

    class _DescriptionEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gru1 = nn.GRU(wv_dim, wv_dim, batch_first=True)
            self.conv1 = nn.Conv1d(wv_dim, wv_dim, kernel_size=3)
            self.gru2 = nn.GRU(wv_dim, wv_dim, batch_first=True)
            self.attn1 = nn.Linear(wv_dim, 1)
            self.attn2 = nn.Linear(wv_dim, 1)
            self.dense = nn.Linear(wv_dim, wv_dim)
            self.default_desc_length = default_desc_length

        def forward(self, desc_embeds: torch.Tensor) -> torch.Tensor:
            gru1_out, _ = self.gru1(desc_embeds)
            conv_out = torch.tanh(self.conv1(gru1_out.transpose(1, 2)).transpose(1, 2))
            attn1_w = torch.softmax(torch.tanh(self.attn1(conv_out)), dim=1)
            reweighted = conv_out * (self.default_desc_length * attn1_w)

            gru2_out, _ = self.gru2(reweighted)
            attn2_w = torch.softmax(torch.tanh(self.attn2(gru2_out)), dim=1)
            reweighted2 = gru2_out * attn2_w

            pooled = reweighted2.sum(dim=1)
            out = torch.tanh(self.dense(pooled))
            result: torch.Tensor = functional.normalize(out, dim=1)
            return result

    return _DescriptionEncoder()


def desc_contrastive_loss(embeds1: torch.Tensor, embeds2: torch.Tensor) -> torch.Tensor:
    """In-batch contrastive loss: pull matched pairs' descriptions together, push others apart.

    Ports the ``indicator``-matrix loss in ``_define_desc_graph``
    (kdcoe.py:358-365): diagonal entries (matched pairs, row *i* with
    column *i*) get weight ``1``, off-diagonal entries (in-batch
    negatives) get weight ``-1/batch_size``, and the loss is
    ``-sum(log(sigmoid(sim * indicator))) / batch_size`` where ``sim =
    embeds1 @ embeds2.T``.
    """
    import torch
    import torch.nn.functional as functional

    batch_size = embeds1.shape[0]
    indicator = torch.full((batch_size, batch_size), -1.0 / batch_size)
    indicator.fill_diagonal_(1.0)
    sim = embeds1 @ embeds2.T
    loss: torch.Tensor = -torch.sum(functional.logsigmoid(sim * indicator)) / batch_size
    return loss


def train_desc_epoch(
    encoder: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    word_embeds: torch.Tensor,
    word_ids: dict[str, npt.NDArray[np.int64]],
    pairs: list[tuple[str, str]],
    rng: np.random.Generator,
    batch_size: int,
) -> None:
    """One epoch of in-batch-contrastive description-encoder training.

    Mirrors OpenEA's ``launch_desc_1epo``: batches of aligned
    ``(source, target)`` pairs (seed + bootstrapped so far, chosen by the
    caller), sampled with replacement each batch (ports OpenEA's own
    ``np.random.choice(..., replace=True)``), each pair's word-id sequence
    looked up into the fixed (not trained further) ``word_embeds`` table
    and encoded independently per side. No-op if ``pairs`` is empty or
    smaller than ``batch_size`` would allow at least one full batch.
    """
    import torch

    if not pairs:
        return
    size = min(batch_size, len(pairs))
    num_batches = len(pairs) // size
    for _ in range(num_batches):
        chosen = rng.choice(len(pairs), size=size, replace=True)
        batch_pairs = [pairs[i] for i in chosen]
        ids1 = np.stack([word_ids[s] for s, _ in batch_pairs])
        ids2 = np.stack([word_ids[t] for _, t in batch_pairs])
        embeds1 = word_embeds[torch.from_numpy(ids1).long()]
        embeds2 = word_embeds[torch.from_numpy(ids2).long()]
        out1 = encoder(embeds1)
        out2 = encoder(embeds2)
        loss = desc_contrastive_loss(out1, out2)
        optimizer.zero_grad()
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()


def validation_hits1_desc(
    encoder: torch.nn.Module,
    word_embeds: torch.Tensor,
    word_ids: dict[str, npt.NDArray[np.int64]],
    val_pairs: list[tuple[str, str]],
) -> float:
    """Cheap top-1 match rate over ``val_pairs`` via the description encoder, for early stopping."""
    import torch

    if not val_pairs:
        return 0.0
    with torch.no_grad():
        ids1 = np.stack([word_ids[s] for s, _ in val_pairs])
        ids2 = np.stack([word_ids[t] for _, t in val_pairs])
        embeds1 = encoder(word_embeds[torch.from_numpy(ids1).long()]).numpy()
        embeds2 = encoder(word_embeds[torch.from_numpy(ids2).long()]).numpy()
    similarities = cosine_similarity(embeds1, embeds2)
    predicted = np.argmax(similarities, axis=1)
    correct = sum(1 for i, j in enumerate(predicted) if j == i)
    return correct / len(val_pairs)


def _margin_loss(
    entity_embeds: torch.nn.Parameter,
    relation_embeds: torch.nn.Parameter,
    pos: npt.NDArray[np.int64],
    neg: npt.NDArray[np.int64],
    margin: float,
) -> torch.Tensor:
    """``sum(relu(||h+r-t||^2 + margin - ||h'+r-t'||^2))`` over a (pos, neg) triple batch.

    Same formula as
    [IPTransELinker][linkingtk.algorithms.ea.iptranse.IPTransELinker]'s
    own private ``_margin_loss`` -- duplicated rather than cross-imported,
    matching this repo's established precedent of duplicating small
    losses/inits across sibling linker modules (see ``_jape_torch.py``).
    """
    import torch
    import torch.nn.functional as functional

    pos_t = torch.from_numpy(pos).long()
    neg_t = torch.from_numpy(neg).long()
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
    ctx1: KGContext,
    ctx2: KGContext,
    rng: np.random.Generator,
    batch_size: int,
    margin: float,
) -> None:
    """One epoch of margin triple-loss training over both KGs' triples.

    Mirrors OpenEA's ``launch_triple_training_1epo``: one pass over both
    KGs' triples (minibatches split proportionally by triple count),
    negative-sampled within each KG's own entities via
    ``sample_negative_triples`` (``_iptranse_training.py``). Same shape as
    [IPTransELinker][linkingtk.algorithms.ea.iptranse.IPTransELinker]'s
    own structural epoch, minus its relation-composition path loss (KDCoE
    has none) and minus its shared-id "sharing" alignment semantics
    (KDCoE bridges KGs with a learned mapping matrix instead -- see
    ``train_mapping_epoch``).
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
        neg1 = sample_negative_triples(ctx1.triples[idx1], ctx1.entity_pool, ctx1.real_triples, rng)
        neg2 = sample_negative_triples(ctx2.triples[idx2], ctx2.entity_pool, ctx2.real_triples, rng)
        neg = np.concatenate([neg1, neg2], axis=0)
        loss = _margin_loss(entity_embeds, relation_embeds, pos, neg, margin)
        optimizer.zero_grad()
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()


def train_mapping_epoch(
    entity_embeds: torch.nn.Parameter,
    mapping_mat: torch.nn.Parameter,
    eye: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    source_ids: npt.NDArray[np.int64],
    target_ids: npt.NDArray[np.int64],
    rng: np.random.Generator,
    steps: int,
    weight: float,
) -> None:
    """One epoch of ``weight``-scaled mapping-loss training.

    Mirrors OpenEA's ``launch_mapping_training_1epo``/
    ``launch_mapping_training_1epo_new`` -- same
    ``sum((t - s@M)^2) + sum((M@M.T - I)^2)`` formula
    [MTransELinker][linkingtk.algorithms.ea.mtranse.MTransELinker] already
    uses (``mapping_loss()`` in OpenEA's ``modules/base/losses.py``), with
    a different pair source and weight per call site: seed pairs
    (``weight=alpha``) or bootstrapped pairs (``weight=new_param``). Like
    MTransE's own mapping-loss training, samples with replacement each
    step rather than OpenEA's ``random.sample`` (same already-established
    deviation). No-ops if ``source_ids`` is empty (the bootstrapped-pairs
    case before any pairs have been found yet).
    """
    import torch
    import torch.nn.functional as functional

    if len(source_ids) == 0:
        return
    batch_size = max(1, len(source_ids) // steps) if steps else len(source_ids)
    source_t = torch.from_numpy(source_ids).long()
    target_t = torch.from_numpy(target_ids).long()
    for _ in range(steps):
        idx = torch.from_numpy(rng.integers(0, len(source_ids), size=batch_size)).long()
        s = functional.normalize(entity_embeds[source_t[idx]], dim=1)
        t = functional.normalize(entity_embeds[target_t[idx]], dim=1)
        mapped = s @ mapping_mat
        map_loss = ((t - mapped) ** 2).sum()
        orth_loss = ((mapping_mat @ mapping_mat.T - eye) ** 2).sum()
        loss = weight * (map_loss + orth_loss)
        optimizer.zero_grad()
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()


def validation_hits1_structural(
    embeds: npt.NDArray[np.floating[Any]],
    mapping: npt.NDArray[np.floating[Any]],
    entity_to_id: dict[str, int],
    val_pairs: list[tuple[str, str]],
) -> float:
    """Cheap top-1 match rate over ``val_pairs`` via mapped structural embeddings.

    Same role as
    [MTransELinker][linkingtk.algorithms.ea.mtranse.MTransELinker]'s
    ``_validation_hits1`` -- a plain patience-driving Hits@1 check, not
    OpenEA's own dual-flag ``early_stop()``.
    """
    if not val_pairs:
        return 0.0
    sources = [s for s, _ in val_pairs]
    targets = [t for _, t in val_pairs]
    source_matrix = np.stack([embeds[entity_to_id[s]] for s in sources]) @ mapping
    target_matrix = np.stack([embeds[entity_to_id[t]] for t in targets])
    similarities = cosine_similarity(source_matrix, target_matrix)
    predicted = np.argmax(similarities, axis=1)
    correct = sum(1 for i, j in enumerate(predicted) if j == i)
    return correct / len(val_pairs)

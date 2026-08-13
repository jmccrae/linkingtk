"""Torch-touching training-step helpers for
[MultiKELinker][linkingtk.algorithms.ea.multike.MultiKELinker].

Separated from ``_multike_text.py``/``_multike_literal.py`` (which stay
plain-numpy and independently testable without ``torch`` installed)
because these functions build/consume PyTorch tensors and modules
directly. Callers must already have confirmed ``torch`` is importable
(``multike.py``'s ``fit()`` does this via ``OptionalDependencyError``
before any of these run) -- that's why these import ``torch``
unconditionally rather than guarding again.

Ports ``src/openea/approaches/multi_ke.py``'s eight per-epoch training
methods (``train_relation_view_1epo``, ``train_attribute_view_1epo``,
``train_common_space_learning_1epo``,
``train_cross_kg_entity_inference_{relation,attribute}_view_1epo``,
``train_cross_kg_{relation,attribute}_inference_1epo``) plus the CNN
attribute scorer (``conv()``). One finding from reading every loss graph
in full, not obvious from a method-name skim: **only the main
relation-view loss ever uses negative sampling** -- every other loss here
(the main attribute-view loss, and all four cross-kg inference losses) is
a purely positive/regression-style loss with no contrastive term at all
(confirmed by their ``_define_*_graph`` methods never declaring a
``*_neg_*`` placeholder). ``neg_triple_num`` therefore only ever applies
to ``train_relation_view_epoch``.

Which embedding tables get L2-normalized on every read also isn't uniform
-- ported from ``initializers.py``'s ``xavier_init(shape, name,
is_l2_norm)``, which returns ``tf.nn.l2_normalize(embeddings, 1) if
is_l2_norm else embeddings`` (a *tensor*, re-evaluated fresh from the
underlying variable on every ``session.run`` -- equivalent to this
family's established ``functional.normalize(param, dim=1)``-at-every-use
convention): ``rv_ent_embeds``/``rel_embeds``/``av_ent_embeds``/
``ent_embeds`` are always normalized this way; ``attr_embeds`` is
constructed with ``is_l2_norm=False`` and is used raw everywhere;
``name_embeds``/``literal_embeds`` (frozen, from
``_multike_literal.encode_literals``) are already unit-norm and never
touched again.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt
    import torch

    from linkingtk.algorithms.ea._kdcoe_torch import KGContext


class AttributeContext(NamedTuple):
    """One KG's own id-mapped, weighted attribute triples."""

    triples: npt.NDArray[np.int64]  # (n, 3): entity, attribute, value ids
    weights: npt.NDArray[np.float32]
    entity_pool: npt.NDArray[np.int64]


def build_attribute_context(
    mapped: npt.NDArray[np.int64], weights: npt.NDArray[np.float32]
) -> AttributeContext:
    """Everything one KG's attribute-view training step needs, derived once at `fit()` start."""
    entity_pool = np.unique(mapped[:, 0]) if len(mapped) else np.empty(0, dtype=np.int64)
    return AttributeContext(triples=mapped, weights=weights, entity_pool=entity_pool)


def build_attribute_scorer(embedding_dim: int) -> torch.nn.Module:
    """A small CNN mapping ``(attribute, value)`` to a predicted head representation.

    Ports ``conv()`` (``multi_ke.py:154-176``): stack the attribute and
    value embeddings into a ``(2, dim)`` "image", two ``Conv2d`` (kernel
    ``2x4``, ``same`` padding, ``tanh``) + L2-normalize layers, flatten,
    one dense ``tanh`` + L2-normalize layer down to ``dim``. The caller
    scores the result against a real head embedding via negative squared
    distance (``conv()``'s own last line, kept outside this module since
    it's shared by several different "head" tables -- see
    ``train_attribute_view_epoch``). A factory function rather than a
    module-level ``torch.nn.Module`` subclass, since ``torch.nn.Module``
    can't be named at module scope without ``torch`` installed (matches
    ``_kdcoe_torch.build_description_encoder``'s convention).

    One deliberate, documented simplification: OpenEA's ``conv()`` also
    applies a ``tf.layers.batch_normalization`` along the "embedding
    dimension" axis (an unusual axis choice, not the channel axis) before
    the first ``Conv2d`` -- reproducing that exact axis-2 TF batchnorm
    would need a non-obvious tensor permutation for a detail that doesn't
    change the CNN's core combinatorial structure, so it's dropped here
    rather than reimplemented.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional

    feature_map_size = 2
    kernel_size = (2, 4)

    class _AttributeScorer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(1, feature_map_size, kernel_size=kernel_size, padding="same")
            self.conv2 = nn.Conv2d(
                feature_map_size, feature_map_size, kernel_size=kernel_size, padding="same"
            )
            self.dense = nn.Linear(2 * embedding_dim * feature_map_size, embedding_dim)

        def forward(self, attr_embeds: torch.Tensor, value_embeds: torch.Tensor) -> torch.Tensor:
            batch = attr_embeds.shape[0]
            stacked = torch.stack([attr_embeds, value_embeds], dim=1)  # (batch, 2, dim)
            x = stacked.unsqueeze(1)  # (batch, 1, 2, dim) -- NCHW
            x = torch.tanh(self.conv1(x))
            x = torch.tanh(self.conv2(x))
            x = functional.normalize(x, dim=3)  # normalize along the "dim" (width) axis
            x = x.reshape(batch, -1)
            predicted = torch.tanh(self.dense(x))
            result: torch.Tensor = functional.normalize(predicted, dim=1)
            return result

    return _AttributeScorer()


def _epoch_batches(
    n: int, batch_size: int, rng: np.random.Generator
) -> list[npt.NDArray[np.int64]]:
    """``ceil(n / batch_size)`` batches of freshly-drawn indices.

    Ports the ``steps = ceil(len(x) / batch_size); size = batch_size if
    steps > 1 else len(x); for i in range(steps): random.sample(x,
    size)`` pattern shared by ``train_common_space_learning_1epo`` and
    all four ``train_cross_kg_*_1epo`` methods -- each step's sample is
    drawn independently (without replacement *within* a step, but with
    replacement *across* steps, i.e. not a single shuffle-once-then-slice
    epoch). Same "full-list-re-run" epoch quirk already ported literally
    elsewhere in this family (AttrE's ``train_joint_epoch``, KDCoE's
    ``train_mapping_epoch``, IMUSE's ``train_align_epoch``).
    """
    if n == 0:
        return []
    steps = max(1, math.ceil(n / batch_size))
    size = min(batch_size if steps > 1 else n, n)
    return [rng.choice(n, size=size, replace=False) for _ in range(steps)]


def train_relation_view_epoch(
    rv_ent_embeds: torch.nn.Parameter,
    rel_embeds: torch.nn.Parameter,
    ent_embeds: torch.nn.Parameter,
    name_embeds: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    ctx1: KGContext,
    ctx2: KGContext,
    rng: np.random.Generator,
    batch_size: int,
    neg_triple_num: int,
) -> None:
    """One epoch of the relation view's logistic TransE loss plus its "relation cv" terms.

    Ports ``train_relation_view_1epo`` + the alignment terms baked into
    ``_define_relation_view_graph``: minibatches proportionally split by
    each KG's own triple count (mirrors ``_iptranse_torch.train_epoch``'s
    shuffle-once-then-slice scheme), ``neg_triple_num`` negatives per
    positive (head-or-tail corrupted, per-KG entity pool). The "relation
    cv" terms pull the shared ``ent_embeds`` toward both TransE-consistency
    (mixing a ``rv_ent_embeds`` endpoint with an ``ent_embeds`` endpoint,
    both directions) and toward ``name_embeds`` for the same entities. One
    combined backward/step -- the caller's ``optimizer`` must cover
    ``[rv_ent_embeds, rel_embeds, ent_embeds]`` (matches TF's default
    "update every trainable variable this loss touches" behavior, since
    OpenEA's own ``relation_optimizer`` has no ``var_list`` restriction).
    """
    import torch
    import torch.nn.functional as functional

    from linkingtk.algorithms.ea._iptranse_training import sample_negative_triples

    total = len(ctx1.triples) + len(ctx2.triples)
    if total == 0:
        return
    steps = max(1, math.ceil(total / batch_size))
    batch1 = round(len(ctx1.triples) / total * batch_size)
    batch2 = batch_size - batch1
    perm1 = rng.permutation(len(ctx1.triples))
    perm2 = rng.permutation(len(ctx2.triples))

    for step in range(steps):
        idx1 = perm1[step * batch1 : (step + 1) * batch1]
        idx2 = perm2[step * batch2 : (step + 1) * batch2]
        if len(idx1) + len(idx2) == 0:
            continue
        pos1 = np.tile(ctx1.triples[idx1], (neg_triple_num, 1))
        pos2 = np.tile(ctx2.triples[idx2], (neg_triple_num, 1))
        neg1 = (
            sample_negative_triples(pos1, ctx1.entity_pool, ctx1.real_triples, rng)
            if len(pos1)
            else pos1
        )
        neg2 = (
            sample_negative_triples(pos2, ctx2.entity_pool, ctx2.real_triples, rng)
            if len(pos2)
            else pos2
        )
        neg = np.concatenate([neg1, neg2], axis=0)
        pos = np.concatenate([ctx1.triples[idx1], ctx2.triples[idx2]], axis=0)

        pos_t = torch.from_numpy(pos).long()
        neg_t = torch.from_numpy(neg).long()

        h = functional.normalize(rv_ent_embeds[pos_t[:, 0]], dim=1)
        r = functional.normalize(rel_embeds[pos_t[:, 1]], dim=1)
        t = functional.normalize(rv_ent_embeds[pos_t[:, 2]], dim=1)
        nh = functional.normalize(rv_ent_embeds[neg_t[:, 0]], dim=1)
        nr = functional.normalize(rel_embeds[neg_t[:, 1]], dim=1)
        nt = functional.normalize(rv_ent_embeds[neg_t[:, 2]], dim=1)
        pos_score = ((h + r - t) ** 2).sum(dim=1)
        neg_score = ((nh + nr - nt) ** 2).sum(dim=1)
        loss = functional.softplus(pos_score).sum() + functional.softplus(-neg_score).sum()

        final_h = functional.normalize(ent_embeds[pos_t[:, 0]], dim=1)
        final_t = functional.normalize(ent_embeds[pos_t[:, 2]], dim=1)
        name_h = name_embeds[pos_t[:, 0]]
        name_t = name_embeds[pos_t[:, 2]]
        align_loss = ((final_h + r - t) ** 2).sum() + ((h + r - final_t) ** 2).sum()
        align_loss = align_loss + 0.5 * ((final_h - name_h) ** 2).sum()
        align_loss = align_loss + 0.5 * ((final_t - name_t) ** 2).sum()
        loss = loss + align_loss

        optimizer.zero_grad()
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()


def train_attribute_view_epoch(
    av_ent_embeds: torch.nn.Parameter,
    attr_embeds: torch.nn.Parameter,
    literal_embeds: torch.Tensor,
    ent_embeds: torch.nn.Parameter,
    name_embeds: torch.Tensor,
    scorer: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    ctx1: AttributeContext,
    ctx2: AttributeContext,
    rng: np.random.Generator,
    batch_size: int,
) -> None:
    """One epoch of the attribute view's weighted CNN loss plus its "attribute cv" terms.

    Ports ``train_attribute_view_1epo`` + the alignment terms baked into
    ``_define_attribute_view_graph``. No negative sampling (see the module
    docstring). Per-triple weights come from
    ``_multike_text.weight_attribute_triples`` (predicate-alignment-aware,
    baked into ``ctx1``/``ctx2`` by the caller). The CNN's ``(attribute,
    value) -> predicted`` forward pass doesn't depend on which "head" table
    it's scored against, so it's computed once per batch and reused for
    both the ``av_ent_embeds``-scored main loss and the
    ``ent_embeds``-scored "attribute cv" term -- mathematically identical
    to OpenEA calling ``conv()`` twice with the same ``(attr, value)``
    args, just without the redundant recomputation. Caller's ``optimizer``
    must cover ``[av_ent_embeds, attr_embeds, ent_embeds] +
    list(scorer.parameters())``.
    """
    import torch
    import torch.nn.functional as functional

    total = len(ctx1.triples) + len(ctx2.triples)
    if total == 0:
        return
    steps = max(1, math.ceil(total / batch_size))
    batch1 = round(len(ctx1.triples) / total * batch_size)
    batch2 = batch_size - batch1
    perm1 = rng.permutation(len(ctx1.triples))
    perm2 = rng.permutation(len(ctx2.triples))

    for step in range(steps):
        idx1 = perm1[step * batch1 : (step + 1) * batch1]
        idx2 = perm2[step * batch2 : (step + 1) * batch2]
        if len(idx1) + len(idx2) == 0:
            continue
        pos = np.concatenate([ctx1.triples[idx1], ctx2.triples[idx2]], axis=0)
        weights = np.concatenate([ctx1.weights[idx1], ctx2.weights[idx2]], axis=0)
        pos_t = torch.from_numpy(pos).long()
        w_t = torch.from_numpy(weights).float()

        attr = attr_embeds[pos_t[:, 1]]
        value = literal_embeds[pos_t[:, 2]]
        predicted = scorer(attr, value)

        head = functional.normalize(av_ent_embeds[pos_t[:, 0]], dim=1)
        main_score = -((head - predicted) ** 2).sum(dim=1)
        loss = (w_t * functional.softplus(-main_score)).sum()

        final_head = functional.normalize(ent_embeds[pos_t[:, 0]], dim=1)
        final_score = -((final_head - predicted) ** 2).sum(dim=1)
        name_head = name_embeds[pos_t[:, 0]]
        loss = loss + functional.softplus(-final_score).sum()
        loss = loss + 0.5 * ((final_head - name_head) ** 2).sum()

        optimizer.zero_grad()
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()


def train_common_space_epoch(
    ent_embeds: torch.nn.Parameter,
    rv_ent_embeds: torch.nn.Parameter,
    av_ent_embeds: torch.nn.Parameter,
    name_embeds: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    entity_ids: npt.NDArray[np.int64],
    rng: np.random.Generator,
    batch_size: int,
    cv_weight: float,
) -> None:
    """Pull the shared ``ent_embeds`` toward all three views, for every entity.

    Ports ``train_common_space_learning_1epo`` (``cross_name_loss``).
    Caller's ``optimizer`` must cover ``[ent_embeds, rv_ent_embeds,
    av_ent_embeds]`` at OpenEA's own distinct ``ITC_learning_rate``.
    """
    import torch
    import torch.nn.functional as functional

    for batch_idx in _epoch_batches(len(entity_ids), batch_size, rng):
        idx = torch.from_numpy(entity_ids[batch_idx]).long()
        final = functional.normalize(ent_embeds[idx], dim=1)
        rv = functional.normalize(rv_ent_embeds[idx], dim=1)
        av = functional.normalize(av_ent_embeds[idx], dim=1)
        name = name_embeds[idx]
        loss = cv_weight * (
            ((final - name) ** 2).sum() + ((final - rv) ** 2).sum() + ((final - av) ** 2).sum()
        )
        optimizer.zero_grad()
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()


def train_cross_kg_relation_entity_epoch(
    rv_ent_embeds: torch.nn.Parameter,
    rel_embeds: torch.nn.Parameter,
    optimizer: torch.optim.Optimizer,
    triples: npt.NDArray[np.int64],
    rng: np.random.Generator,
    batch_size: int,
) -> None:
    """MultiKE's core supervision channel, relation-view half.

    Ports ``train_cross_kg_entity_inference_relation_view_1epo``: plain
    (no softplus, no negatives) TransE-consistency loss over seed-
    substituted triples from
    ``_multike_text.generate_cross_kg_relation_triples``. Caller's
    ``optimizer`` must cover ``[rv_ent_embeds, rel_embeds]``.
    """
    import torch
    import torch.nn.functional as functional

    for batch_idx in _epoch_batches(len(triples), batch_size, rng):
        batch = torch.from_numpy(triples[batch_idx]).long()
        h = functional.normalize(rv_ent_embeds[batch[:, 0]], dim=1)
        r = functional.normalize(rel_embeds[batch[:, 1]], dim=1)
        t = functional.normalize(rv_ent_embeds[batch[:, 2]], dim=1)
        loss = 2 * ((h + r - t) ** 2).sum()
        optimizer.zero_grad()
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()


def train_cross_kg_attribute_entity_epoch(
    av_ent_embeds: torch.nn.Parameter,
    attr_embeds: torch.nn.Parameter,
    literal_embeds: torch.Tensor,
    scorer: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    triples: npt.NDArray[np.int64],
    rng: np.random.Generator,
    batch_size: int,
) -> None:
    """MultiKE's core supervision channel, attribute-view half.

    Ports ``train_cross_kg_entity_inference_attribute_view_1epo``:
    softplus CNN loss (no negatives, no per-triple weight) over seed-
    substituted triples from
    ``_multike_text.generate_cross_kg_attribute_triples``. Caller's
    ``optimizer`` must cover ``[av_ent_embeds, attr_embeds] +
    list(scorer.parameters())``.
    """
    import torch
    import torch.nn.functional as functional

    for batch_idx in _epoch_batches(len(triples), batch_size, rng):
        batch = torch.from_numpy(triples[batch_idx]).long()
        head = functional.normalize(av_ent_embeds[batch[:, 0]], dim=1)
        attr = attr_embeds[batch[:, 1]]
        value = literal_embeds[batch[:, 2]]
        predicted = scorer(attr, value)
        score = -((head - predicted) ** 2).sum(dim=1)
        loss = 2 * functional.softplus(-score).sum()
        optimizer.zero_grad()
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()


def train_cross_kg_relation_predicate_epoch(
    rv_ent_embeds: torch.nn.Parameter,
    rel_embeds: torch.nn.Parameter,
    optimizer: torch.optim.Optimizer,
    triples: npt.NDArray[np.int64],
    weights: npt.NDArray[np.float32],
    rng: np.random.Generator,
    batch_size: int,
) -> None:
    """Predicate-soft-alignment supervision, relation-view half.

    Ports ``train_cross_kg_relation_inference_1epo``
    (``positive_loss_with_weight``): softplus-wrapped, per-triple-weighted
    TransE-consistency loss over predicate-substituted triples from
    ``_multike_text.substitute_predicate_triples``. Caller's ``optimizer``
    must cover ``[rv_ent_embeds, rel_embeds]``; only meant to be called
    once training has warmed up past ``start_predicate_soft_alignment``.
    """
    import torch
    import torch.nn.functional as functional

    for batch_idx in _epoch_batches(len(triples), batch_size, rng):
        batch = torch.from_numpy(triples[batch_idx]).long()
        w = torch.from_numpy(weights[batch_idx]).float()
        h = functional.normalize(rv_ent_embeds[batch[:, 0]], dim=1)
        r = functional.normalize(rel_embeds[batch[:, 1]], dim=1)
        t = functional.normalize(rv_ent_embeds[batch[:, 2]], dim=1)
        distance_sq = ((h + r - t) ** 2).sum(dim=1)
        loss = 2 * (w * functional.softplus(distance_sq)).sum()
        optimizer.zero_grad()
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()


def train_cross_kg_attribute_predicate_epoch(
    av_ent_embeds: torch.nn.Parameter,
    attr_embeds: torch.nn.Parameter,
    literal_embeds: torch.Tensor,
    scorer: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    triples: npt.NDArray[np.int64],
    weights: npt.NDArray[np.float32],
    rng: np.random.Generator,
    batch_size: int,
) -> None:
    """Predicate-soft-alignment supervision, attribute-view half.

    Ports ``train_cross_kg_attribute_inference_1epo``: softplus-wrapped,
    per-triple-weighted CNN loss over predicate-substituted triples from
    ``_multike_text.substitute_predicate_triples``. **No ``2x`` loss
    multiplier** here, unlike the other three cross-kg training
    functions -- ported as a literal quirk from ``ckga_attribute_loss =
    pos_loss`` (vs. the ``2 * ...`` in every sibling method), not
    "fixed" for consistency since there's no evidence it's a bug rather
    than an intentional/harmless scalar-weight asymmetry. Caller's
    ``optimizer`` must cover ``[av_ent_embeds, attr_embeds] +
    list(scorer.parameters())``; only meant to be called once training has
    warmed up past ``start_predicate_soft_alignment``.
    """
    import torch
    import torch.nn.functional as functional

    for batch_idx in _epoch_batches(len(triples), batch_size, rng):
        batch = torch.from_numpy(triples[batch_idx]).long()
        w = torch.from_numpy(weights[batch_idx]).float()
        head = functional.normalize(av_ent_embeds[batch[:, 0]], dim=1)
        attr = attr_embeds[batch[:, 1]]
        value = literal_embeds[batch[:, 2]]
        predicted = scorer(attr, value)
        score = -((head - predicted) ** 2).sum(dim=1)
        loss = (w * functional.softplus(-score)).sum()
        optimizer.zero_grad()
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()

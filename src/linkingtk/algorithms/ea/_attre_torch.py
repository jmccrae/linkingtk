"""Torch-touching training-step helpers for
[AttrELinker][linkingtk.algorithms.ea.attre.AttrELinker].

Separated from ``_attre_text.py`` (which stays plain-numpy and
independently testable without ``torch`` installed) because these
functions build/consume PyTorch tensors directly. Callers must already
have confirmed ``torch`` is importable (``attre.py``'s ``fit()`` does this
via ``OptionalDependencyError`` before any of these run) -- that's why
these import ``torch`` unconditionally rather than guarding again.

Reuses [KGContext][linkingtk.algorithms.ea._kdcoe_torch.KGContext],
[build_kg_context][linkingtk.algorithms.ea._kdcoe_torch.build_kg_context],
and
[train_structural_epoch][linkingtk.algorithms.ea._kdcoe_torch.train_structural_epoch]
directly for AttrE's structural (SE) half -- they're already fully
generic over how ids were assigned or what a triple "means", they just
need an already-id-mapped ``(n, 3)`` array. ``build_kg_context`` is
likewise reused here for the *attribute*-triple side (the CE half): its
``(triples, real_triples, entity_pool)`` shape is equally valid whether
column semantics are ``(head, relation, tail)`` or ``(entity, attribute,
value)``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from linkingtk.algorithms.ea._attre_text import sample_negative_attribute_triples
from linkingtk.algorithms.ea._kdcoe_torch import KGContext

if TYPE_CHECKING:
    import numpy.typing as npt
    import torch


def compose_value_embeddings(char_embeds_lookup: torch.Tensor) -> torch.Tensor:
    """``(batch, literal_len, dim)`` char embeddings -> ``(batch, dim)`` composed value embeddings.

    Ports ``n_gram_compositional_func``/``calculate_ngram_weight``
    (``attre.py:88-109``) in closed form. Worked through by hand, the
    original's ``tf.while_loop`` computes, for every prefix length
    ``k=1..L``, the mean of the first ``k`` characters' embeddings, then
    sums those ``L`` prefix-means together -- a left-to-right,
    prefix-emphasizing composition (earlier characters appear in every
    prefix-mean term, so contribute the most overall; the last character
    only appears in the single longest-prefix term).
    ``add_compositional_func`` (plain mean + L2-normalize), also defined
    in ``attre.py`` but never actually wired into ``_define_embed_graph``,
    isn't ported.

    Vectorized here via ``torch.cumsum`` rather than the original's
    iterative ``tf.while_loop`` -- mathematically identical (mean of the
    first ``k`` characters = ``cumsum[k-1] / k``), not a behavioral
    deviation.

    Args:
        char_embeds_lookup: ``(batch, literal_len, dim)`` -- each value's
            already-looked-up (and, per this family's ``char_l2_norm``
            convention, already-normalized) character embeddings, in
            original left-to-right order.

    Returns:
        ``(batch, dim)`` composed value embeddings. **Not** itself
        L2-normalized after composition, matching
        ``n_gram_compositional_func``'s own lack of a final normalize
        step.
    """
    import torch

    length = char_embeds_lookup.shape[1]
    cumsum = char_embeds_lookup.cumsum(dim=1)
    denom = torch.arange(
        1, length + 1, dtype=char_embeds_lookup.dtype, device=char_embeds_lookup.device
    ).view(1, -1, 1)
    prefix_means = cumsum / denom
    result: torch.Tensor = prefix_means.sum(dim=1)
    return result


def _attr_margin_loss(
    entity_embeds_ce: torch.nn.Parameter,
    attr_embeds: torch.nn.Parameter,
    char_embeds: torch.nn.Parameter,
    value_char_ids: torch.Tensor,
    pos: npt.NDArray[np.int64],
    neg: npt.NDArray[np.int64],
    margin: float,
) -> torch.Tensor:
    """``sum(relu(||e+a-v||^2 + margin - ||e'+a'-v'||^2))`` over a (pos, neg) batch.

    Same margin formula as every other linker in this family's own
    ``_margin_loss``, adapted for attribute triples: ``v``/``v'`` are
    composed value embeddings
    ([compose_value_embeddings][linkingtk.algorithms.ea._attre_torch.compose_value_embeddings])
    rather than plain entity-table lookups -- this is AttrE's own core
    idea, that structural and attribute triples share one scoring
    function once a value has an embedding at all.
    """
    import torch
    import torch.nn.functional as functional

    device = entity_embeds_ce.device
    pos_t = torch.from_numpy(pos).long().to(device)
    neg_t = torch.from_numpy(neg).long().to(device)
    pos_chars = functional.normalize(char_embeds[value_char_ids[pos_t[:, 2]]], dim=2)
    neg_chars = functional.normalize(char_embeds[value_char_ids[neg_t[:, 2]]], dim=2)
    pe = functional.normalize(entity_embeds_ce[pos_t[:, 0]], dim=1)
    pa = functional.normalize(attr_embeds[pos_t[:, 1]], dim=1)
    pv = compose_value_embeddings(pos_chars)
    ne = functional.normalize(entity_embeds_ce[neg_t[:, 0]], dim=1)
    na = functional.normalize(attr_embeds[neg_t[:, 1]], dim=1)
    nv = compose_value_embeddings(neg_chars)
    pos_score = ((pe + pa - pv) ** 2).sum(dim=1)
    neg_score = ((ne + na - nv) ** 2).sum(dim=1)
    loss: torch.Tensor = torch.relu(pos_score + margin - neg_score).sum()
    return loss


def train_attr_epoch(
    entity_embeds_ce: torch.nn.Parameter,
    attr_embeds: torch.nn.Parameter,
    char_embeds: torch.nn.Parameter,
    value_char_ids: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    ctx1: KGContext,
    ctx2: KGContext,
    rng: np.random.Generator,
    batch_size: int,
    margin: float,
) -> None:
    """One epoch of margin-loss training over both KGs' attribute triples (the CE half).

    Same batch-splitting shape as
    [train_structural_epoch][linkingtk.algorithms.ea._kdcoe_torch.train_structural_epoch],
    adapted for attribute triples: negatives corrupt only the entity
    endpoint
    ([sample_negative_attribute_triples][linkingtk.algorithms.ea._attre_text.sample_negative_attribute_triples],
    not the relation-triple head-or-tail coin flip), and the "tail" side
    is a composed value embedding rather than a plain entity-embedding
    lookup.

    Args:
        value_char_ids: ``(num_values, literal_len)`` int64 lookup table
            -- row ``v`` is value id ``v``'s fixed-length character-id
            sequence.
    """

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
        neg1 = sample_negative_attribute_triples(
            ctx1.triples[idx1], ctx1.entity_pool, ctx1.real_triples, rng
        )
        neg2 = sample_negative_attribute_triples(
            ctx2.triples[idx2], ctx2.entity_pool, ctx2.real_triples, rng
        )
        neg = np.concatenate([neg1, neg2], axis=0)
        loss = _attr_margin_loss(
            entity_embeds_ce, attr_embeds, char_embeds, value_char_ids, pos, neg, margin
        )
        optimizer.zero_grad()
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()


def train_joint_epoch(
    entity_embeds: torch.nn.Parameter,
    entity_embeds_ce: torch.nn.Parameter,
    optimizer: torch.optim.Optimizer,
    entity_ids: npt.NDArray[np.int64],
    steps: int,
) -> None:
    """``steps`` repeated passes of the full-entity-list cosine-alignment ("joint") loss.

    Ports ``launch_joint_training_1epo`` literally, including its
    curiosity: it re-runs the joint loss over the *entire* entity list
    ``steps`` times per outer epoch (not sub-batched slices, despite
    ``steps`` being derived from ``batch_size``) -- ported as read, not
    "fixed", since that's what the published number came from. This is
    the mechanism that actually ties the structural (``entity_embeds``)
    and character/attribute (``entity_embeds_ce``) spaces together --
    without it they'd train independently.

    Args:
        entity_ids: Every entity id in both KGs (shared-id space).
        steps: Repeat count, derived by the caller from
            ``ceil(len(entities) / batch_size)`` matching OpenEA's own
            ``launch_joint_training_1epo``.
    """
    import torch
    import torch.nn.functional as functional

    entity_t = torch.from_numpy(entity_ids).long().to(entity_embeds.device)
    for _ in range(steps):
        se = functional.normalize(entity_embeds[entity_t], dim=1)
        ce = functional.normalize(entity_embeds_ce[entity_t], dim=1)
        cos_sim = (se * ce).sum(dim=1)
        loss = (1 - cos_sim).sum()
        optimizer.zero_grad()
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()

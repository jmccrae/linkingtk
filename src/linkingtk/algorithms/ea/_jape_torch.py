"""Torch-touching training-step helpers for [JAPELinker][linkingtk.algorithms.ea.jape.JAPELinker].

Separated from ``_jape_training.py`` (which stays plain-numpy and
independently testable without ``torch``) because these functions build/
consume PyTorch tensors and modules directly. Callers must already have
confirmed ``torch`` is importable (``jape.py``'s ``fit()`` does this via
``OptionalDependencyError`` before any of these run) -- that's why these
import ``torch`` unconditionally rather than guarding again.
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
    """One KG's own id-mapped triples plus its negative-sampling pool."""

    triples: npt.NDArray[np.int64]
    real_triples: set[tuple[int, int, int]]
    entity_pool: npt.NDArray[np.int64]


def build_kg_context(mapped: npt.NDArray[np.int64]) -> KGContext:
    """Everything one KG's triple-loss step needs, derived once at `fit()` start."""
    real_triples = {(int(h), int(r), int(t)) for h, r, t in mapped}
    entity_pool = (
        np.unique(np.concatenate([mapped[:, 0], mapped[:, 2]]))
        if len(mapped)
        else np.empty(0, dtype=np.int64)
    )
    return KGContext(triples=mapped, real_triples=real_triples, entity_pool=entity_pool)


def train_attr2vec(
    training_pairs_ids: npt.NDArray[np.int64],
    num_attributes: int,
    embedding_dim: int,
    attr_max_epoch: int,
    batch_size: int,
    learning_rate: float,
    rng: np.random.Generator,
    device: torch.device,
) -> npt.NDArray[np.floating[Any]]:
    """Train Attr2Vec's skip-gram-style attribute-correlation embedding.

    Ports OpenEA's ``Attr2Vec`` (``approaches/attr2vec.py``): one
    embedding table (Xavier-normal init, L2-normalized on every forward
    read, same per-step-renorm trick as
    [MTransELinker][linkingtk.algorithms.ea.mtranse.MTransELinker]/
    [IPTransELinker][linkingtk.algorithms.ea.iptranse.IPTransELinker])
    plus separate NCE weight/bias parameters, trained via a
    noise-contrastive-estimation-style loss over
    ``training_pairs_ids``' (input, label) pairs.

    **Deviation**: OpenEA's ``tf.nn.nce_loss`` defaults to a *log-uniform*
    candidate sampler, which assumes class ids are already frequency-
    sorted (a word2vec/NLP-vocabulary-scale convention) -- our attribute-
    predicate ids aren't assigned in frequency order, and PyTorch has no
    drop-in equivalent, so this trains a **uniform-negative-sampling**
    NCE loss instead: one shared set of ``num_attributes // 5`` negative
    classes drawn uniformly at random per minibatch (matching OpenEA's
    negative-count ratio, just not its noise distribution) -- the
    standard word2vec-SGNS-style loss, same purpose (approximate softmax
    via negative sampling), a simpler and well-justified noise
    distribution given our modest attribute-vocabulary sizes.

    Args:
        training_pairs_ids: ``(n, 2)`` int64 array of ``(input_attr_id,
            label_attr_id)`` skip-gram pairs.
        num_attributes: Size of the attribute-id vocabulary.
        embedding_dim: Embedding dimensionality.
        attr_max_epoch: Training epochs.
        batch_size: Mini-batch size (drawn without replacement per step
            from ``training_pairs_ids``, matching OpenEA's
            ``random.sample`` -- falls back to sampling with replacement
            if ``training_pairs_ids`` is smaller than ``batch_size``, so
            this doesn't crash on tiny inputs).
        learning_rate: Adagrad's learning rate.
        rng: Random generator, for reproducibility.
        device: Torch device to train on -- this builds its own embedding
            table from scratch rather than receiving one, so it takes
            ``device`` explicitly rather than deriving it from an
            argument.

    Returns:
        ``(num_attributes, embedding_dim)`` trained, L2-normalized
        attribute embeddings.
    """
    import torch
    import torch.nn.functional as functional

    num_sampled = max(1, num_attributes // 5)

    def init_xavier(rows: int) -> torch.nn.Parameter:
        return torch.nn.Parameter(
            functional.normalize(
                torch.nn.init.xavier_normal_(torch.empty(rows, embedding_dim, device=device)),
                dim=1,
            )
        )

    attr_embeds = init_xavier(num_attributes)
    nce_weights = init_xavier(num_attributes)
    nce_biases = torch.nn.Parameter(torch.zeros(num_attributes, device=device))
    optimizer = torch.optim.Adagrad([attr_embeds, nce_weights, nce_biases], lr=learning_rate)

    inputs = torch.from_numpy(training_pairs_ids[:, 0]).long().to(device)
    labels = torch.from_numpy(training_pairs_ids[:, 1]).long().to(device)
    batch_size = min(batch_size, len(inputs))
    steps = max(1, len(inputs) // batch_size)

    for _epoch in range(attr_max_epoch):
        for _step in range(steps):
            idx = rng.choice(len(inputs), size=batch_size, replace=False)
            batch_inputs = inputs[idx]
            batch_labels = labels[idx]

            normalized_attrs = functional.normalize(attr_embeds, dim=1)
            normalized_weights = functional.normalize(nce_weights, dim=1)
            pos_input = normalized_attrs[batch_inputs]
            pos_score = (pos_input * normalized_weights[batch_labels]).sum(dim=1) + nce_biases[
                batch_labels
            ]
            pos_loss = -functional.logsigmoid(pos_score)

            neg_ids = (
                torch.from_numpy(rng.integers(0, num_attributes, size=num_sampled))
                .long()
                .to(device)
            )
            neg_score = pos_input @ normalized_weights[neg_ids].T + nce_biases[neg_ids]
            neg_loss = -functional.logsigmoid(-neg_score).sum(dim=1)

            loss = (pos_loss + neg_loss).mean()
            optimizer.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()

    with torch.no_grad():
        final: npt.NDArray[np.floating[Any]] = (
            functional.normalize(attr_embeds, dim=1).cpu().numpy()
        )
    return final


def train_triple_epoch(
    entity_embeds: torch.nn.Parameter,
    relation_embeds: torch.nn.Parameter,
    optimizer: torch.optim.Optimizer,
    ctx1: KGContext,
    ctx2: KGContext,
    rng: np.random.Generator,
    batch_size: int,
    neg_alpha: float,
) -> None:
    """One epoch of JAPE's structural triple loss.

    ``sum(||h+r-t||^2) - neg_alpha * sum(||h'+r-t'||^2)`` -- no margin/
    hinge, unlike MTransE's/IPTransE's losses; this is OpenEA's actual
    (unbounded-below) formula for JAPE specifically. Mirrors
    ``launch_triple_training_1epo``.
    """
    import torch
    import torch.nn.functional as functional

    device = entity_embeds.device
    total = len(ctx1.triples) + len(ctx2.triples)
    steps = max(1, math.ceil(total / batch_size))
    batch1 = round(len(ctx1.triples) / total * batch_size) if total else 0
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
        loss = pos_score.sum() - neg_alpha * neg_score.sum()

        optimizer.zero_grad()
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()


def train_sim_epoch(
    entity_embeds: torch.nn.Parameter,
    sim_optimizer: torch.optim.Optimizer,
    attr_sim_mat: npt.NDArray[np.floating[Any]],
    pool1_ids: npt.NDArray[np.int64],
    pool2_ids: npt.NDArray[np.int64],
    rng: np.random.Generator,
    sub_mat_size: int,
    attr_sim_mat_beta: float,
) -> None:
    """One epoch of JAPE's attribute-similarity structural regularizer.

    Mirrors ``launch_sim_1epo``: repeatedly subsample ``sub_mat_size``
    reference-pool-1 entities, blend reference-pool-2's *current*
    structural embeddings by their (precomputed, fixed) attribute
    similarity, and pull the sampled entities' own structural embeddings
    toward that blend. If ``pool1_ids`` is smaller than ``sub_mat_size``,
    no steps run this epoch (matches OpenEA's own integer-division
    behavior, which floors to zero steps in that case).
    """
    import torch
    import torch.nn.functional as functional

    if len(pool1_ids) == 0 or len(pool2_ids) == 0:
        return
    steps = len(pool1_ids) // sub_mat_size
    if steps == 0:
        return

    device = entity_embeds.device
    pool2_t = torch.from_numpy(pool2_ids).long().to(device)
    for _ in range(steps):
        idx = rng.choice(len(pool1_ids), size=sub_mat_size, replace=False)
        batch_entities = torch.from_numpy(pool1_ids[idx]).long().to(device)
        sim_rows = torch.from_numpy(attr_sim_mat[idx, :]).float().to(device)

        ref1 = functional.normalize(entity_embeds[batch_entities], dim=1)
        ref2 = functional.normalize(entity_embeds[pool2_t], dim=1)
        ref2_trans = functional.normalize(sim_rows @ ref2, dim=1)
        loss = attr_sim_mat_beta * ((ref1 - ref2_trans) ** 2).sum()

        sim_optimizer.zero_grad()
        loss.backward()  # type: ignore[no-untyped-call]
        sim_optimizer.step()


def validation_hits1(
    embeds: npt.NDArray[np.floating[Any]],
    entity_to_id: dict[str, int],
    val_pairs: list[tuple[str, str]],
) -> float:
    """Cheap top-1 match rate over ``val_pairs``, for early-stopping only.

    Same deliberately simplified patience-counter role as ``mtranse.py``'s/
    ``iptranse.py``'s ``_validation_hits1``.
    """
    sources = [source for source, _ in val_pairs]
    targets = [target for _, target in val_pairs]
    source_matrix = np.stack([embeds[entity_to_id[source]] for source in sources])
    target_matrix = np.stack([embeds[entity_to_id[target]] for target in targets])
    similarities = cosine_similarity(source_matrix, target_matrix)
    predicted = np.argmax(similarities, axis=1)
    correct = sum(1 for i, j in enumerate(predicted) if j == i)
    return correct / len(val_pairs)

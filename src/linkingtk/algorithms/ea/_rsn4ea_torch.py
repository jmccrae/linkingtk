"""Torch-touching model + training-step helpers for
[RSN4EALinker][linkingtk.algorithms.ea.rsn4ea.RSN4EALinker].

Separated from the orchestrating ``rsn4ea.py`` the same way
``_kdcoe_torch.py``/``_multike_torch.py`` are split from their linkers.
Callers must already have confirmed ``torch`` is importable (``rsn4ea.py``'s
``fit()`` does this via ``OptionalDependencyError`` before any of these
run).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

if TYPE_CHECKING:
    import numpy.typing as npt
    import torch


def build_rsn_model(
    num_entities: int,
    num_relations: int,
    hidden_size: int,
    num_layers: int,
    keep_prob: float,
) -> torch.nn.Module:
    """Entity/relation embeddings + a hand-rolled RSN-LSTM + output heads.

    A factory function rather than a module-level ``torch.nn.Module``
    subclass, since ``torch.nn.Module`` can't be named at module scope
    without ``torch`` installed -- same precedent as
    ``_kdcoe_torch.build_description_encoder``.

    **Hand-rolled LSTM, not ``torch.nn.LSTM``**: OpenEA's own
    ``tf.contrib.rnn.LSTMCell`` is built with ``activation=tf.identity``,
    replacing the standard LSTM's ``tanh`` nonlinearity on *both* the
    candidate-cell update and the hidden-state output
    (``c_t = f*c_{t-1} + i*g`` and ``h_t = o*c_t``, no ``tanh`` either
    place) -- a real, load-bearing architectural choice (removing the
    squashing nonlinearity plausibly helps the same long-range gradient
    flow the "recurrent skipping" residual is designed for), not a
    throwaway default. ``torch.nn.LSTM`` hardcodes ``tanh`` internally with
    no override, so this port hand-rolls a small multi-layer LSTM instead.
    Also matches two more OpenEA specifics ``torch.nn.LSTM`` doesn't:
    orthogonal (not uniform) initialization of each layer's recurrent
    weight matrix, and dropout (``output_keep_prob=keep_prob``) applied to
    *every* layer's output including the last (``torch.nn.LSTM``'s own
    ``dropout`` kwarg only applies *between* stacked layers).

    Embeddings and the ``_ent_w``/``_rel_w`` output-head weight matrices
    use Xavier-normal init (``xavier_initializer(uniform=False)`` in
    OpenEA), **not** L2-normalized -- unlike every other linker in this
    family, RSN4EA's own reference never runs its embeddings through the
    ``init_embeddings()`` utility that applies ``l2_normalize`` (see
    ``rsn4ea.py``'s module docstring).

    Args:
        num_entities: Entity vocabulary size (post id-mapping).
        num_relations: Relation vocabulary size, *after* doubling for
            reverse edges (see
            [build_augmented_kb][linkingtk.algorithms.ea._rsn4ea_training.build_augmented_kb]).
        hidden_size: Embedding dimensionality and LSTM hidden size (OpenEA
            drives both from one ``hidden_size`` config value -- its
            separate ``dim`` config is never actually read, see
            ``rsn4ea.py``'s module docstring).
        num_layers: LSTM depth. OpenEA's published value is ``2``.
        keep_prob: Per-layer output dropout keep-probability. OpenEA's
            published value is ``0.6``.

    Returns:
        An ``nn.Module`` with ``entity_embedding``/``relation_embedding``
        (``nn.Embedding``), ``rel_w``/``rel_b``/``ent_w``/``ent_b``
        (``nn.Parameter``s, for ``logits = hidden @ w.T + b``),
        ``input_bn``/``output_bn`` (each a single shared ``nn.BatchNorm1d``,
        matching OpenEA's own single-batch-norm-reused-per-position
        pattern -- not one per sequence position), ``rel_proj``/``ent_proj``
        (bias-free ``nn.Linear``, the RSN residual's two projections), and
        an LSTM stack. ``forward(seq)`` takes a ``(batch, max_length)``
        int64 sequence and returns a list of ``max_length - 1``
        ``(batch, hidden_size)`` output-position tensors (entity positions
        raw, relation positions RSN-corrected, both output-batch-normed).
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional

    class _IdentityLSTMLayer(nn.Module):
        """One LSTM layer with ``identity`` (not ``tanh``) cell/hidden activation."""

        def __init__(self, input_size: int, hidden: int) -> None:
            super().__init__()
            self.hidden_size = hidden
            self.weight_ih = nn.Parameter(torch.empty(4 * hidden, input_size))
            self.weight_hh = nn.Parameter(torch.empty(4 * hidden, hidden))
            self.bias_ih = nn.Parameter(torch.zeros(4 * hidden))
            self.bias_hh = nn.Parameter(torch.zeros(4 * hidden))
            nn.init.xavier_normal_(self.weight_ih)
            nn.init.orthogonal_(self.weight_hh)
            with torch.no_grad():
                # forget_bias=1 in OpenEA's LSTMCell -- gate order i,f,g,o.
                self.bias_ih[hidden : 2 * hidden] = 1.0

        def forward(
            self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]
        ) -> tuple[torch.Tensor, torch.Tensor]:
            h, c = state
            gates = x @ self.weight_ih.T + self.bias_ih + h @ self.weight_hh.T + self.bias_hh
            i, f, g, o = gates.chunk(4, dim=1)
            i = torch.sigmoid(i)
            f = torch.sigmoid(f)
            o = torch.sigmoid(o)
            c_new = f * c + i * g  # identity activation on the candidate cell
            h_new = o * c_new  # identity activation on the hidden output
            return h_new, c_new

    class _IdentityLSTM(nn.Module):
        def __init__(self, input_size: int, hidden: int, layers: int, keep: float) -> None:
            super().__init__()
            self.hidden_size = hidden
            self.keep_prob = keep
            self.layers = nn.ModuleList(
                [
                    _IdentityLSTMLayer(input_size if i == 0 else hidden, hidden)
                    for i in range(layers)
                ]
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            batch, seq_len, _ = x.shape
            h = [torch.zeros(batch, self.hidden_size, device=x.device) for _ in self.layers]
            c = [torch.zeros(batch, self.hidden_size, device=x.device) for _ in self.layers]
            outputs = []
            for t in range(seq_len):
                layer_input = x[:, t, :]
                for layer_index, layer in enumerate(self.layers):
                    h[layer_index], c[layer_index] = layer(
                        layer_input, (h[layer_index], c[layer_index])
                    )
                    layer_input = functional.dropout(
                        h[layer_index], p=1.0 - self.keep_prob, training=self.training
                    )
                outputs.append(layer_input)
            return torch.stack(outputs, dim=1)

    class _RSNModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.entity_embedding = nn.Embedding(num_entities, hidden_size)
            self.relation_embedding = nn.Embedding(num_relations, hidden_size)
            nn.init.xavier_normal_(self.entity_embedding.weight)
            nn.init.xavier_normal_(self.relation_embedding.weight)

            self.rel_w = nn.Parameter(torch.empty(num_relations, hidden_size))
            self.rel_b = nn.Parameter(torch.zeros(num_relations))
            self.ent_w = nn.Parameter(torch.empty(num_entities, hidden_size))
            self.ent_b = nn.Parameter(torch.zeros(num_entities))
            nn.init.xavier_normal_(self.rel_w)
            nn.init.xavier_normal_(self.ent_w)

            self.input_bn = nn.BatchNorm1d(hidden_size)
            self.output_bn = nn.BatchNorm1d(hidden_size)
            self.rel_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.ent_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.lstm = _IdentityLSTM(hidden_size, hidden_size, num_layers, keep_prob)

        def forward(self, seq: torch.Tensor) -> list[torch.Tensor]:
            length = seq.shape[1]
            ent_ids = seq[:, 0 : length - 1 : 2]
            rel_ids = seq[:, 1::2]
            ent_em = self.entity_embedding(ent_ids)
            rel_em = self.relation_embedding(rel_ids)

            positions = length - 1
            em_seq = [
                ent_em[:, i // 2] if i % 2 == 0 else rel_em[:, i // 2] for i in range(positions)
            ]
            bn_em_seq = [self.input_bn(step) for step in em_seq]
            ent_bn_em = [bn_em_seq[i] for i in range(0, positions, 2)]

            stacked = torch.stack(bn_em_seq, dim=1)
            outputs = self.lstm(stacked)

            corrected = []
            for i in range(positions):
                if i % 2 == 0:
                    corrected.append(outputs[:, i, :])
                else:
                    entity_step = ent_bn_em[i // 2]
                    corrected.append(self.rel_proj(outputs[:, i, :]) + self.ent_proj(entity_step))
            return [self.output_bn(step) for step in corrected]

    return _RSNModel()


def train_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    paths: npt.NDArray[np.int64],
    rng: np.random.Generator,
    batch_size: int,
) -> None:
    """One epoch of RSN4EA's next-token training, matching OpenEA's ``seq_train``.

    Batches (sampled with replacement, matching OpenEA's own
    ``np.random.choice(..., replace=True)``) are fed through ``model``;
    at each of the ``max_length - 1`` output positions, entity-position
    outputs (even index) predict the next *relation*, RSN-corrected
    relation-position outputs (odd index) predict the next *entity* --
    via a full softmax cross-entropy against ``rel_w``/``rel_b`` or
    ``ent_w``/``ent_b`` respectively (this port's replacement for OpenEA's
    ``tf.nn.nce_loss`` sampled softmax -- see ``rsn4ea.py``'s module
    docstring for why). A single Bernoulli(0.5) per-example weight mask is
    drawn once per batch and reused, unmodified, across every position's
    loss term (matching OpenEA's own ``cal_loss``, which draws its mask
    once outside the per-position loop) -- effectively drops ~half of each
    batch's rows from the loss entirely, every position, every step.
    Gradients are clipped to global norm ``2.0`` (matching
    ``tf.clip_by_global_norm(grads, 2.0)``).

    No-op if ``paths`` is empty or smaller than ``batch_size``.
    """
    import torch
    import torch.nn.functional as functional

    if len(paths) < batch_size:
        return
    device = next(model.parameters()).device
    num_batches = len(paths) // batch_size
    choices = rng.choice(len(paths), size=len(paths), replace=True)

    model.train()
    for b in range(num_batches):
        batch_idx = choices[b * batch_size : (b + 1) * batch_size]
        batch = paths[batch_idx]
        seq = torch.from_numpy(batch).long().to(device)
        outputs = model(seq)

        mask = (rng.random(batch_size) < 0.5).astype(np.float32)
        mask_t = torch.from_numpy(mask).to(device)

        loss = torch.zeros((), device=device)
        for i, output in enumerate(outputs):
            target = seq[:, i + 1]
            if i % 2 == 0:
                logits = output @ model.rel_w.T + model.rel_b
            else:
                logits = output @ model.ent_w.T + model.ent_b
            per_example = functional.cross_entropy(logits, target, reduction="none")
            loss = loss + (per_example * mask_t).sum()
        loss = loss / batch_size

        optimizer.zero_grad()
        loss.backward()  # type: ignore[no-untyped-call]
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()


def validation_hits1(
    entity_embeds: npt.NDArray[np.floating[Any]],
    entity_to_id: dict[str, int],
    val_pairs: list[tuple[str, str]],
) -> float:
    """Cheap top-1 match rate over ``val_pairs``, for early-stopping only.

    Same simplified patience-counter role as every sibling's version. No
    mapping projection -- RSN4EA never builds one (its cross-KG alignment
    signal comes entirely from path training over alias-substituted
    triples, see ``rsn4ea.py``), matching
    ``_iptranse_torch.validation_hits1``'s raw-both-sides shape.
    """
    sources = [s for s, _ in val_pairs]
    targets = [t for _, t in val_pairs]
    source_matrix = np.stack([entity_embeds[entity_to_id[s]] for s in sources])
    target_matrix = np.stack([entity_embeds[entity_to_id[t]] for t in targets])
    similarities = cosine_similarity(source_matrix, target_matrix)
    predicted = np.argmax(similarities, axis=1)
    correct = sum(1 for i, j in enumerate(predicted) if j == i)
    return correct / len(val_pairs)

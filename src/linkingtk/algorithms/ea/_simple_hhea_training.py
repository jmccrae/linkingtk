"""Margin-ranking training loop for
[SimpleHHEALinker][linkingtk.algorithms.ea.simple_hhea.SimpleHHEALinker].

Ports `main_SimpleHHEA.py`'s `train()`/`get_train_set()`/`l1()` from
https://github.com/DataArcTech/Simple-HHEA. Negatives are uniform-random
entity ids, not hard-negative mining -- a simplification already present
in the reference itself, not something to "improve" here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn

from linkingtk.algorithms.ea._simple_hhea_torch import SimpleHHEAModel

if TYPE_CHECKING:
    import numpy.typing as npt


def _l1_distance(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    result: torch.Tensor = torch.sum(torch.abs(left - right), dim=-1)
    return result


def build_training_quads(
    train_pairs: npt.NDArray[np.int64], batch_size: int, node_size: int, rng: np.random.Generator
) -> npt.NDArray[np.int64]:
    """Builds a fixed ``(batch_size, 4)`` array of ``(l, r, random_l, random_r)`` rows.

    Ports `get_train_set` exactly, including its one real quirk: `l`/`r`
    are `train_pairs` *repeated* (not resampled) enough times to reach
    `batch_size` rows (`ceil(batch_size / len(train_pairs))` copies,
    shuffled, then truncated to exactly `batch_size`) -- built **once**,
    not per epoch. The reference's own call site passes the *node count*
    as `batch_size` (not e.g. `len(train_pairs)`), so `train_pairs` gets
    repeated roughly `node_size / len(train_pairs)` times; the two random
    -negative columns are drawn once here too, then held fixed for the
    entire training run (`train()`'s epoch loop never redraws them).
    """
    negative_ratio = batch_size // len(train_pairs) + 1
    repeated = np.tile(train_pairs, (negative_ratio, 1))
    rng.shuffle(repeated)
    repeated = repeated[:batch_size]
    negatives = rng.integers(0, node_size, size=repeated.shape)
    return np.concatenate([repeated, negatives], axis=-1)


def train(
    model: SimpleHHEAModel,
    train_pairs: npt.NDArray[np.int64],
    node_size: int,
    *,
    num_epochs: int = 1500,
    learning_rate: float = 0.01,
    weight_decay: float = 0.001,
    gamma: float = 1.0,
    random_state: int | None = None,
) -> None:
    """Trains ``model`` in place via margin ranking loss.

    One fixed set of training quads is built up front (see
    [build_training_quads][linkingtk.algorithms.ea._simple_hhea_training.build_training_quads],
    `batch_size=node_size` matching the reference's own call). Every
    epoch: one full forward pass over the entire entity set (`model()` --
    the reference's `Simple_HHEA.forward()` has no batching, every
    entity's fused embedding is recomputed every step), then
    ``sum(relu(gamma + d(l,r) - d(l,r')) + relu(gamma + d(l,r) - d(l',r)))
    / len(quads)`` where ``d`` is L1 distance -- ports `main_SimpleHHEA.py`'s
    `train()` exactly (minus its `retain_graph=True`, unneeded here since
    each epoch builds a fresh graph from a fresh `model()` call).
    """
    rng = np.random.default_rng(random_state)
    quads = build_training_quads(train_pairs, node_size, node_size, rng)
    device = model.name_emb.device
    indices = torch.from_numpy(quads).long().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    for _epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        features = model()[indices]
        anchor_l, anchor_r = features[:, 0, :], features[:, 1, :]
        negative_l, negative_r = features[:, 2, :], features[:, 3, :]

        positive_distance = _l1_distance(anchor_l, anchor_r)
        loss = torch.sum(
            nn.functional.relu(gamma + positive_distance - _l1_distance(anchor_l, negative_r))
            + nn.functional.relu(gamma + positive_distance - _l1_distance(negative_l, anchor_r))
        ) / len(quads)

        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()

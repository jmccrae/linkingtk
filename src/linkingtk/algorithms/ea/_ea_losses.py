"""Shared margin-ranking loss functions for the GNN-based EA linkers.

Both losses below are ported once from OpenEA's reference implementation and
reused across linkers, rather than each one reimplementing the same math in
its own private ``_<name>_torch.py``:

- `margin_ranking_loss_l1`: the L1-distance, double-sided-negative margin
  loss GCN-Align's ``align_loss`` (``approaches/gcn_align.py``) and RDGCN's
  ``get_loss`` (``approaches/rdgcn.py``) both use, verbatim -- confirmed by
  reading both directly, the two functions are identical modulo variable
  names.
- `margin_ranking_loss_l2_squared`: AliNet's squared-Euclidean single-sided
  variant (``AliNet.compute_loss``, ``approaches/alinet.py``), with its own
  ``neg_margin_balance`` weight on the negative term.

Both take an already-built embedding tensor plus index tensors into it
(rather than pre-gathered vectors) so callers can pass whichever entity
representation is currently live (e.g. AliNet's per-round concatenated
multi-layer output) without an extra gather step at the call site.

Callers must already have confirmed ``torch`` is importable (each linker's
``fit()`` does this via ``OptionalDependencyError`` before any of these
run) -- same precedent as ``_rsn4ea_torch.py``/``_kdcoe_torch.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def margin_ranking_loss_l1(
    embeddings: torch.Tensor,
    pos_left: torch.Tensor,
    pos_right: torch.Tensor,
    neg_left: torch.Tensor,
    neg_right: torch.Tensor,
    neg2_left: torch.Tensor,
    neg2_right: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """L1-distance margin loss over ``t`` positive pairs and ``k`` negatives per side.

    Ports OpenEA's ``align_loss``/``get_loss`` exactly: for each positive
    pair, both a left-corrupted (``neg_left``/``neg_right``) and a
    right-corrupted (``neg2_left``/``neg2_right``) set of ``k`` negatives
    contribute a ``relu(margin + pos_dist - neg_dist)`` term, averaged over
    ``2 * k * t``.

    Args:
        embeddings: ``(num_entities, dim)`` embedding table to index into.
        pos_left: ``(t,)`` ids, the positive pairs' left/source side.
        pos_right: ``(t,)`` ids, the positive pairs' right/target side.
        neg_left: ``(t * k,)`` ids -- left-corrupted negatives' left side
            (each positive pair's own left id, repeated ``k`` times; see
            each linker's negative-sampling helper).
        neg_right: ``(t * k,)`` ids -- left-corrupted negatives' right side
            (``k`` random/hard-mined ids per positive pair).
        neg2_left: ``(t * k,)`` ids -- right-corrupted negatives' left side.
        neg2_right: ``(t * k,)`` ids -- right-corrupted negatives' right
            side (each positive pair's own right id, repeated ``k`` times).
        gamma: Margin.

    Returns:
        A scalar loss tensor.
    """
    import torch

    t = pos_left.shape[0]
    k = neg_left.shape[0] // t
    pos_dist = (embeddings[pos_left] - embeddings[pos_right]).abs().sum(dim=1)
    margin = (pos_dist + gamma).reshape(t, 1)

    def _side_loss(left_ids: torch.Tensor, right_ids: torch.Tensor) -> torch.Tensor:
        neg_dist = (embeddings[left_ids] - embeddings[right_ids]).abs().sum(dim=1)
        return torch.relu(margin - neg_dist.reshape(t, k)).sum()

    loss = _side_loss(neg_left, neg_right) + _side_loss(neg2_left, neg2_right)
    return loss / (2.0 * k * t)


def margin_ranking_loss_l2_squared(
    embeddings: torch.Tensor,
    pos_left: torch.Tensor,
    pos_right: torch.Tensor,
    neg_left: torch.Tensor,
    neg_right: torch.Tensor,
    margin: float,
    neg_margin_balance: float,
) -> torch.Tensor:
    """Squared-Euclidean margin loss: pulls positives together, pushes negatives apart.

    Ports AliNet's ``compute_loss`` exactly: ``sum(||e1 - e2||^2)`` over
    positive pairs, plus ``neg_margin_balance * sum(relu(margin -
    ||e1 - e2||^2))`` over negative pairs (a single corruption side, unlike
    `margin_ranking_loss_l1`'s two).

    Args:
        embeddings: ``(num_entities, dim)`` embedding table to index into.
        pos_left: ``(t,)`` ids, positive pairs' left side.
        pos_right: ``(t,)`` ids, positive pairs' right side.
        neg_left: ``(n,)`` ids, negative pairs' left side.
        neg_right: ``(n,)`` ids, negative pairs' right side.
        margin: Margin subtracted from each negative pair's squared distance.
        neg_margin_balance: Weight applied to the summed negative-pair loss.

    Returns:
        A scalar loss tensor.
    """
    import torch

    pos_dist = ((embeddings[pos_left] - embeddings[pos_right]) ** 2).sum(dim=1)
    pos_loss = pos_dist.sum()
    neg_dist = ((embeddings[neg_left] - embeddings[neg_right]) ** 2).sum(dim=1)
    neg_loss = torch.relu(margin - neg_dist).sum()
    return pos_loss + neg_margin_balance * neg_loss

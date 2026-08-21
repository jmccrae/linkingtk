"""One linear WordNet-relation graph-propagation step over sense logits.

EWISER's central idea (Bevilacqua & Navigli, ACL 2020): instead of scoring
each sense independently, boost every synset's raw classification logit by
a weighted sum of its WordNet-neighbors' logits, so the model can transfer
confidence from a frequently-labeled sense (e.g. a common hypernym) to a
rarer one it's graph-adjacent to, even if that rarer sense was never seen
labeled in training. This module reimplements that *operation* --
``new_logits = (A @ old_logits) + old_logits`` for a sparse ``[V, V]``
adjacency ``A`` -- from the paper's description and the shape observed
directly in the released checkpoints' state dicts, not ported from
EWISER's own (CC-BY-NC-SA licensed) source.

Deliberately reimplemented with plain ``torch.sparse.mm``/
``torch.sparse_coo_tensor`` instead of the reference's own
``torch_scatter``/``torch_sparse`` dependency: those packages are
unconditionally imported by the reference's equivalent module for what is,
in substance, one sparse-dense matmul plus an optional row-sum
normalization -- and they're notoriously fragile, tightly pinned to exact
torch+CUDA build combinations. A deliberate deviation from upstream, not
an oversight.
"""

from __future__ import annotations

import torch
from torch import nn


class StructuredLogits(nn.Module):
    """Graph-propagation step: ``new_logits = (A @ old_logits) + old_logits``.

    Stores the sparse adjacency's `indices`/`values`/`size` as separate
    parameters under the same names
    (``adjacency_pars.0``/``adjacency_pars.1``/``adjacency_pars.2``) the
    released EWISER checkpoints use for their own equivalent module, so a
    checkpoint's state dict loads via plain ``load_state_dict(strict=True)``
    with no key renaming.

    Args:
        adjacency: A sparse (coalesced or not) ``[V, V]`` COO tensor, e.g.
            from
            [build_relation_adjacency][linkingtk.algorithms.wsd._ewiser_graph.build_relation_adjacency]
            or loaded directly from a checkpoint's own baked-in adjacency.
        trainable: Whether the adjacency's edge weights are updated during
            training (the indices/shape are always fixed).
        renormalize: If ``True``, additionally divide each node's
            propagated neighbor sum by that node's total incoming weight
            (computed via ``torch.sparse.sum``, no ``torch_scatter``
            needed). The three released checkpoints all use ``False``
            (their own edge weights are pre-normalized at graph-build
            time, see `build_relation_adjacency`).
    """

    def __init__(
        self,
        adjacency: torch.Tensor,
        trainable: bool = False,
        renormalize: bool = False,
    ) -> None:
        super().__init__()
        adjacency = adjacency.coalesce()
        self.adjacency_pars = nn.ParameterList(
            [
                nn.Parameter(adjacency.indices(), requires_grad=False),
                nn.Parameter(adjacency.values(), requires_grad=trainable),
                nn.Parameter(torch.tensor(adjacency.shape), requires_grad=False),
            ]
        )
        self.renormalize = renormalize

    def _adjacency(self) -> torch.Tensor:
        indices, values, size = self.adjacency_pars
        return torch.sparse_coo_tensor(
            indices, values, size=tuple(size.tolist()), device=values.device
        ).coalesce()

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Propagate `logits` (``[..., V]``) one step through the WordNet graph."""
        adjacency = self._adjacency()
        vocab_size = logits.shape[-1]
        flat = logits.reshape(-1, vocab_size)
        neighbors = torch.sparse.mm(adjacency, flat.transpose(0, 1)).transpose(0, 1)
        if self.renormalize:
            row_sum = torch.sparse.sum(adjacency, dim=1).to_dense().clamp(min=1e-12)
            neighbors = neighbors / row_sum.unsqueeze(0)
        result: torch.Tensor = (neighbors + flat).reshape(logits.shape)
        return result

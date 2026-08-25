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

The `trainable=True`, `renormalize=False` path (the default, and what
every released checkpoint and the reference's own ``bin/train-ewiser.sh``
actually use in both training stages) runs through `_SparsePropagate`
below instead of a bare ``torch.sparse.mm`` call: `torch.sparse.mm`'s
*default* backward, when the sparse operand itself requires grad,
materializes a dense ``[V, V]`` gradient rather than an ``O(nnz)`` one --
invisible at a toy vocabulary size, but confirmed directly (issue #58) to
OOM at EWISER's own real ~117,664-entry WordNet vocabulary (a dense
``[117664, 117664]`` float32 gradient is ~51.6GB). `_SparsePropagate`
computes the same values via an explicit, ``O(nnz)`` indexed
gather-multiply-sum instead, verified against `torch.autograd.gradcheck`
and a dense reference implementation, and measured at 3ms/0.4GB peak at
the full ~117,664-entry / ~168K-edge scale (vs. an unfixed OOM). The
`renormalize=True` combination is untouched (no released checkpoint or
`bin/train-ewiser.sh` config uses it, so it isn't scale-verified the way
the default path now is) and still goes through the original
`torch.sparse.mm` path.
"""

from __future__ import annotations

import torch
from torch import nn


class _SparsePropagate(torch.autograd.Function):
    """``flat @ A.T`` for a sparse ``[V, V]`` COO adjacency ``A``, with an
    ``O(nnz)`` (not ``O(V^2)``) backward with respect to `values`.

    See the module docstring for why this exists instead of a bare
    ``torch.sparse.mm`` call.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        indices: torch.Tensor,
        values: torch.Tensor,
        size: tuple[int, int],
        flat: torch.Tensor,
    ) -> torch.Tensor:
        adjacency = torch.sparse_coo_tensor(indices, values, size=size).coalesce()
        result: torch.Tensor = torch.sparse.mm(adjacency, flat.transpose(0, 1)).transpose(0, 1)
        ctx.save_for_backward(indices, values, flat)
        ctx.size = size  # type: ignore[attr-defined]
        return result

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> tuple[None, torch.Tensor | None, None, torch.Tensor | None]:
        indices, values, flat = ctx.saved_tensors  # type: ignore[attr-defined]
        row, col = indices[0], indices[1]

        grad_flat = None
        if ctx.needs_input_grad[3]:  # type: ignore[attr-defined]
            size = ctx.size  # type: ignore[attr-defined]
            adjacency_t = torch.sparse_coo_tensor(
                indices.flip(0), values, size=(size[1], size[0])
            ).coalesce()
            grad_flat = torch.sparse.mm(adjacency_t, grad_output.transpose(0, 1)).transpose(0, 1)

        grad_values = None
        if ctx.needs_input_grad[1]:  # type: ignore[attr-defined]
            grad_values = (grad_output[:, row] * flat[:, col]).sum(dim=0)

        return None, grad_values, None, grad_flat


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
        vocab_size = logits.shape[-1]
        flat = logits.reshape(-1, vocab_size)
        indices, values, size = self.adjacency_pars
        if self.renormalize:
            adjacency = self._adjacency()
            neighbors = torch.sparse.mm(adjacency, flat.transpose(0, 1)).transpose(0, 1)
            row_sum = torch.sparse.sum(adjacency, dim=1).to_dense().clamp(min=1e-12)
            neighbors = neighbors / row_sum.unsqueeze(0)
        else:
            neighbors = _SparsePropagate.apply(  # type: ignore[no-untyped-call]
                indices, values, tuple(size.tolist()), flat
            )
        result: torch.Tensor = (neighbors + flat).reshape(logits.shape)
        return result

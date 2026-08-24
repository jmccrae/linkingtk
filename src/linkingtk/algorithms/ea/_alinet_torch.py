"""PyTorch model and training-step functions for
[AliNetLinker][linkingtk.algorithms.ea.alinet.AliNetLinker].

Ports OpenEA's ``GraphConvolution``/``AliNetGraphAttentionLayer``/
``HighwayLayer``/``AliNet._define_model`` (``approaches/alinet.py``): a
stack of layers where every layer but the last combines a 1-hop
``GraphConvolution`` branch and a 2-hop ``AliNetGraphAttentionLayer``
branch, fused by a ``HighwayLayer`` gate; the last layer is
``GraphConvolution`` only. The final representation concatenates every
layer's (plus the initial embedding's) L2-normalized output -- a
jumping-knowledge-style multi-scale representation. Uses plain
``torch.sparse`` throughout (including ``torch.sparse.softmax``, same as
``_rdgcn_torch.py``) -- no `torch_geometric`/`dgl` dependency.

Callers must already have confirmed ``torch`` is importable
(``alinet.py``'s ``fit()`` does this via ``OptionalDependencyError``
before any of these run) -- same precedent as ``_gcn_align_torch.py``/
``_rdgcn_torch.py``.

**Dropout and L2 weight regularization are not implemented.** OpenEA's own
published config uses ``dropout: 0.0`` (never actually triggers the
reference's own dropout branches), and while every learnable weight in
the reference is declared with an ``l2_regularizer(scale=0.01)``, nothing
in ``AliNet``'s own training graph construction (``_generate_graph``/
``_generate_rel_graph``) ever retrieves and sums TensorFlow's
regularization-losses collection into the actual optimized loss --
confirmed by reading ``AliNet``'s class body directly. Both are faithful
no-ops to skip, not simplifications with real fidelity cost.

**This is the only linker in this package whose adjacency mutates during
training** -- ``AliNetModel.set_adjacency`` is called once before training
and again after every bootstrapping round (see
[AliNetLinker.fit][linkingtk.algorithms.ea.alinet.AliNetLinker.fit]),
updating every layer's stored sparse adjacency in place rather than
rebuilding the model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def build_alinet_model(num_entities: int, layer_dims: list[int]) -> torch.nn.Module:
    """Ports ``AliNet._define_model``'s layer stack as a live PyTorch model.

    A factory function rather than a module-level ``torch.nn.Module``
    subclass, since ``torch.nn.Module`` can't be named at module scope
    without ``torch`` installed -- same precedent as
    ``_gcn_align_torch.build_gcn_align_branch``/
    ``_rdgcn_torch.build_rdgcn_model``.

    Args:
        num_entities: Total entity count (both KGs combined).
        layer_dims: Layer width sequence, e.g. ``[500, 400, 300]`` for 2
            layers (``layer_dims[0]`` is the initial embedding's
            dimensionality). OpenEA's published EN-FR-15K-V1 config uses
            ``[500, 400, 300]``.

    Returns:
        A module with:

        - ``set_adjacency(one_hop_adjacency, two_hop_adjacency)``: must be
          called before the first ``forward()`` and again after every
          bootstrapping round. Both are coalesced sparse
          ``[num_entities, num_entities]`` tensors, e.g. from
          [coo_to_torch_sparse][linkingtk.utils.sparse_gcn.coo_to_torch_sparse].
        - ``forward() -> torch.Tensor``: returns
          ``(num_entities, sum(layer_dims) + layer_dims[0])`` -- the
          concatenated, L2-normalized multi-layer representation (every
          layer's output plus the initial embedding, matching OpenEA's
          own ``[input_embeds] + output_embeds_list`` concat order).
    """
    import torch
    import torch.nn.functional as functional
    from torch import nn

    class _GraphConvolution(nn.Module):
        """Ports ``GraphConvolution`` (single-adjacency case, the only one
        AliNet's own code ever uses)."""

        def __init__(self, in_dim: int, out_dim: int) -> None:
            super().__init__()
            weight = torch.empty(in_dim, out_dim)
            nn.init.xavier_uniform_(weight)
            self.weight = nn.Parameter(weight)
            self.bias = nn.Parameter(torch.zeros(out_dim))
            self.batch_norm = nn.BatchNorm1d(in_dim)
            self.adjacency: torch.Tensor | None = None

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            assert self.adjacency is not None  # noqa: S101 -- set via set_adjacency before use
            hidden = self.batch_norm(inputs)
            pre_sup = hidden @ self.weight
            out = torch.sparse.mm(self.adjacency, pre_sup) + self.bias
            return torch.tanh(out)

    class _AliNetGraphAttentionLayer(nn.Module):
        """Ports ``AliNetGraphAttentionLayer``: self+neighbor attention over
        the (typically 2-hop) adjacency."""

        def __init__(self, in_dim: int, out_dim: int) -> None:
            super().__init__()
            kernel = torch.empty(in_dim, out_dim)
            nn.init.xavier_uniform_(kernel)
            self.kernel = nn.Parameter(kernel)
            kernel1 = torch.empty(in_dim, in_dim)
            nn.init.xavier_uniform_(kernel1)
            self.kernel1 = nn.Parameter(kernel1)
            kernel2 = torch.empty(in_dim, in_dim)
            nn.init.xavier_uniform_(kernel2)
            self.kernel2 = nn.Parameter(kernel2)
            self.batch_norm = nn.BatchNorm1d(in_dim)
            self.adjacency: torch.Tensor | None = None

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            assert self.adjacency is not None  # noqa: S101 -- set via set_adjacency before use
            adjacency = self.adjacency.coalesce()
            hidden = self.batch_norm(inputs)
            mapped = hidden @ self.kernel
            attention1 = hidden @ self.kernel1
            attention2 = hidden @ self.kernel2
            self_score = (attention1 * hidden).sum(dim=1, keepdim=True)
            neighbor_score = (attention2 * hidden).sum(dim=1, keepdim=True)
            self_score = torch.tanh(self_score)
            neighbor_score = torch.tanh(neighbor_score)

            indices = adjacency.indices()
            row, col = indices[0], indices[1]
            edge_logits = self_score[row].squeeze(-1) + neighbor_score[col].squeeze(-1)
            edge_logits = functional.leaky_relu(edge_logits)
            size = adjacency.shape
            sparse_logits = torch.sparse_coo_tensor(indices, edge_logits, size=size).coalesce()
            attention_weights = torch.sparse.softmax(sparse_logits, dim=1)
            value = torch.sparse.mm(attention_weights, mapped)
            return torch.tanh(value)

    class _HighwayLayer(nn.Module):
        """Ports ``HighwayLayer`` -- including its one shared `BatchNormalization`
        instance applied to *both* inputs sequentially (confirmed from
        reading ``HighwayLayer.__init__``/``call`` directly: one
        `self.batch_normal`, called twice), and its ``tanh`` -> ``relu``
        gate (not the more common sigmoid gate)."""

        def __init__(self, dim: int) -> None:
            super().__init__()
            weight = torch.empty(dim, dim)
            nn.init.xavier_uniform_(weight)
            self.weight = nn.Parameter(weight)
            self.batch_norm = nn.BatchNorm1d(dim)

        def forward(self, layer1: torch.Tensor, layer2: torch.Tensor) -> torch.Tensor:
            layer1 = self.batch_norm(layer1)
            layer2 = self.batch_norm(layer2)
            gate = torch.relu(torch.tanh(layer1 @ self.weight))
            output = layer2 * (1.0 - gate) + layer1 * gate
            return torch.tanh(output)

    class _AliNetModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            init_table = torch.empty(num_entities, layer_dims[0])
            nn.init.xavier_uniform_(init_table)
            self.init_embedding = nn.Parameter(init_table)

            self.num_layers = len(layer_dims) - 1
            self.gcn_layers = nn.ModuleList(
                [
                    _GraphConvolution(layer_dims[i], layer_dims[i + 1])
                    for i in range(self.num_layers)
                ]
            )
            self.gat_layers = nn.ModuleList(
                [
                    _AliNetGraphAttentionLayer(layer_dims[i], layer_dims[i + 1])
                    for i in range(self.num_layers - 1)
                ]
            )
            self.highway_layers = nn.ModuleList(
                [_HighwayLayer(layer_dims[i + 1]) for i in range(self.num_layers - 1)]
            )

        def set_adjacency(
            self, one_hop_adjacency: torch.Tensor, two_hop_adjacency: torch.Tensor
        ) -> None:
            for gcn in self.gcn_layers:
                gcn.adjacency = one_hop_adjacency
            for gat in self.gat_layers:
                gat.adjacency = two_hop_adjacency

        def forward(self) -> torch.Tensor:
            output = self.init_embedding
            layer_outputs = []
            for i in range(self.num_layers):
                gcn_out = self.gcn_layers[i](output)
                if i < self.num_layers - 1:
                    gat_out = self.gat_layers[i](output)
                    output = self.highway_layers[i](gat_out, gcn_out)
                else:
                    output = gcn_out
                layer_outputs.append(output)

            all_layers = [self.init_embedding, *layer_outputs]
            normalized = [functional.normalize(layer, dim=1) for layer in all_layers]
            concatenated = torch.cat(normalized, dim=1)
            result: torch.Tensor = functional.normalize(concatenated, dim=1)
            return result

    return _AliNetModel()

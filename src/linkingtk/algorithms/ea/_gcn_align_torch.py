"""PyTorch model and training-step functions for
[GCNAlignLinker][linkingtk.algorithms.ea.gcn_align.GCNAlignLinker].

Ports OpenEA's ``GCN_Align_Unit``/``GraphConvolution`` (``approaches/gcn_align.py``)
both branches:

- The structural (``se``) branch: a 2-layer, purely-propagation GCN over a
  directly-learnable per-entity embedding table -- layer 1 is "featureless"
  (its only weight *is* the learnable embedding table itself, transformed by
  nothing but the adjacency), layer 2 has no weight at all (``transform=False``,
  ``input_dim == output_dim``), so the whole branch is
  ``A_norm @ relu(A_norm @ E)`` for a learnable ``E``.
- The attribute (``ae``) branch: the same 2-layer shape, but layer 1 has a
  real learnable weight matrix transforming the sparse attribute-presence
  input (from
  [build_attribute_features][linkingtk.algorithms.ea._gcn_align_training.build_attribute_features])
  before propagating -- ``A_norm @ relu(A_norm @ (attrs @ W))``.

Uses plain ``torch.sparse.mm`` throughout (see
[coo_to_torch_sparse][linkingtk.utils.sparse_gcn.coo_to_torch_sparse]) rather
than a GNN library -- see the module docstring on
[gcn_align][linkingtk.algorithms.ea.gcn_align] for why.

Callers must already have confirmed ``torch`` is importable
(``gcn_align.py``'s ``fit()`` does this via ``OptionalDependencyError``
before any of these run) -- same precedent as ``_rsn4ea_torch.py``/
``_kdcoe_torch.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def build_gcn_align_branch(num_entities: int, dim: int) -> torch.nn.Module:
    """OpenEA's ``GCN_Align_Unit`` structural branch: two featureless-first-layer GCN steps.

    A factory function rather than a module-level ``torch.nn.Module``
    subclass, since ``torch.nn.Module`` can't be named at module scope
    without ``torch`` installed -- same precedent as
    ``_kdcoe_torch.build_description_encoder``/``_rsn4ea_torch.build_rsn_model``.

    Args:
        num_entities: Total entity count (both KGs combined) -- both the
            embedding table's row count and (per OpenEA's ``trunc_normal``
            init) the scale of its initialization stddev.
        dim: Embedding/output dimensionality (OpenEA's ``se_dim``).

    Returns:
        A module whose ``forward(adjacency)`` takes a coalesced sparse
        ``[num_entities, num_entities]`` normalized adjacency (e.g. from
        [coo_to_torch_sparse][linkingtk.utils.sparse_gcn.coo_to_torch_sparse])
        and returns ``(num_entities, dim)`` entity embeddings.
    """
    import torch
    import torch.nn.functional as functional
    from torch import nn

    class _GCNAlignBranch(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            table = torch.empty(num_entities, dim)
            nn.init.trunc_normal_(table, std=1.0 / (num_entities**0.5))
            self.embedding = nn.Parameter(functional.normalize(table, dim=1))

        def forward(self, adjacency: torch.Tensor) -> torch.Tensor:
            hidden = torch.relu(torch.sparse.mm(adjacency, self.embedding))
            return torch.sparse.mm(adjacency, hidden)  # type: ignore[no-any-return]

    return _GCNAlignBranch()


def build_gcn_align_attr_branch(num_attrs: int, dim: int) -> torch.nn.Module:
    """OpenEA's ``GCN_Align_Unit`` attribute branch: a real layer-1 weight over sparse features.

    Same factory-function precedent as
    [build_gcn_align_branch][linkingtk.algorithms.ea._gcn_align_torch.build_gcn_align_branch].
    Layer 1's weight uses the same ``trunc_normal`` init (truncated normal,
    stddev ``1/sqrt(input_dim)``, then L2-row-normalized) OpenEA's own
    ``GCN_Align_Unit._build`` applies regardless of branch (``init=trunc_normal``
    is passed for both the ``se`` and ``ae`` branches' layer 1, confirmed by
    reading ``_build`` directly -- only ``featureless``/``sparse_inputs``
    differ between the two branches' constructor args).

    Args:
        num_attrs: Attribute feature-vocabulary size (layer 1's input
            dimension), e.g. from
            [build_attribute_features][linkingtk.algorithms.ea._gcn_align_training.build_attribute_features].
        dim: Output dimensionality (OpenEA's ``ae_dim``).

    Returns:
        A module whose ``forward(adjacency, attr_features)`` takes a
        coalesced sparse ``[num_entities, num_entities]`` normalized
        adjacency and a sparse ``[num_entities, num_attrs]`` attribute
        feature matrix, returning ``(num_entities, dim)`` entity
        embeddings.
    """
    import torch
    import torch.nn.functional as functional
    from torch import nn

    class _GCNAlignAttrBranch(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            weight = torch.empty(num_attrs, dim)
            nn.init.trunc_normal_(weight, std=1.0 / (num_attrs**0.5))
            self.weight = nn.Parameter(functional.normalize(weight, dim=1))

        def forward(self, adjacency: torch.Tensor, attr_features: torch.Tensor) -> torch.Tensor:
            pre_sup = torch.sparse.mm(attr_features, self.weight)
            hidden = torch.relu(torch.sparse.mm(adjacency, pre_sup))
            return torch.sparse.mm(adjacency, hidden)  # type: ignore[no-any-return]

    return _GCNAlignAttrBranch()

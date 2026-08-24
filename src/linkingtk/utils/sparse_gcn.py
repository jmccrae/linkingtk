"""Shared sparse-adjacency helpers for the GNN-based EA linkers
([GCNAlignLinker][linkingtk.algorithms.ea.gcn_align.GCNAlignLinker],
[RDGCNLinker][linkingtk.algorithms.ea.rdgcn.RDGCNLinker],
[AliNetLinker][linkingtk.algorithms.ea.alinet.AliNetLinker]).

Every one of the three needs the same symmetric-degree-normalize step over a
(weighted or unweighted) adjacency after self-loops are added --
``normalize_adjacency_coo`` below is that one piece of math, ported once. The
*edge weighting itself* (GCN-Align's relation-functionality weighting,
RDGCN's plain unweighted primal graph, AliNet's plain unweighted 1-/2-hop
graphs) is genuinely method-specific and stays in each linker's own private
``_<name>_training.py``.

Kept as its own module (not folded into
[linkingtk.utils.graph][linkingtk.utils.graph]) so that module can stay
torch-free -- ``coo_to_torch_sparse`` imports ``torch`` lazily inside the
function body, matching every other optional-torch import in this package
(see [resolve_device][linkingtk.utils.device.resolve_device]).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sp

if TYPE_CHECKING:
    import numpy.typing as npt
    import torch


def normalize_adjacency_coo(
    indices: npt.NDArray[np.int64],
    values: npt.NDArray[np.float64],
    num_nodes: int,
    add_self_loops: bool = True,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """Symmetrically degree-normalize a weighted adjacency: ``D^-0.5 A D^-0.5``.

    Degree ``D`` comes from ``A``'s **column** sums, not row sums -- this
    only matters when ``A`` is asymmetric (true for GCN-Align's
    relation-functionality-weighted edges; a no-op for RDGCN's/AliNet's
    plain unweighted, inherently-symmetric adjacencies, where row- and
    column-sums coincide). This is a deliberately-derived match for OpenEA's
    own ``GCN_Utils.normalize_adj``, not an arbitrary choice: that function
    computes ``D^-0.5 A_stored^T D^-0.5`` (row-sum degree, but with an
    extra transpose) over a matrix ``A_stored`` that OpenEA's own
    ``get_weighted_adj`` already builds row/col-swapped
    (``row.append(key[1]); col.append(key[0])`` -- storing each edge at
    ``[target, source]``, not ``[source, target]``). Working through both
    swaps algebraically: ``D^-0.5 A_stored^T D^-0.5`` computed over that
    swapped storage is *exactly* ``D^-0.5 A D^-0.5`` computed over the
    natural (source-row, target-column) orientation this function expects
    (see
    [build_weighted_adjacency][linkingtk.algorithms.ea._gcn_align_training.build_weighted_adjacency]),
    **using column sums for the degree** (since ``A_stored``'s row-sums are
    ``A``'s column-sums once the swap unwinds). Confirmed empirically, not
    just algebraically: on the real EN-FR-15K-V1 benchmark, this
    column-sum version reaches Hits@10=0.243/MRR=0.154 vs. a row-sum
    version's Hits@10=0.148/MRR=0.116 (300 epochs, otherwise identical) --
    a meaningfully different, and clearly better, result, not an
    inconsequential implementation detail. Duplicate ``(row, col)`` entries
    in ``indices`` are summed before normalizing.

    Args:
        indices: ``(2, nnz)`` int64 array, row 0 = row ids, row 1 = column
            ids -- the same axis layout `torch.sparse_coo_tensor` expects,
            so [coo_to_torch_sparse][linkingtk.utils.sparse_gcn.coo_to_torch_sparse]
            can consume this function's output directly.
        values: ``(nnz,)`` edge weights, same order as ``indices``' columns.
        num_nodes: Total node count (both KGs combined, in this package's
            usage -- entity ids from
            [build_id_mappings][linkingtk.utils.graph.build_id_mappings]
            are already in one shared id space).
        add_self_loops: Whether to add weight-``1.0`` self-loops before
            normalizing (summed with any self-loop weight already present
            in ``indices``/``values``, not overwritten).

    Returns:
        ``(out_indices, out_values)`` in the same ``(2, nnz)``/``(nnz,)``
        shape as the input, coalesced (no duplicate ``(row, col)`` pairs).
    """
    row, col = indices[0], indices[1]
    adjacency = sp.coo_matrix((values, (row, col)), shape=(num_nodes, num_nodes))
    adjacency.sum_duplicates()
    if add_self_loops:
        adjacency = adjacency + sp.eye(num_nodes, format="coo")
    adjacency = adjacency.tocoo()

    col_sum = np.asarray(adjacency.sum(axis=0)).flatten()
    with np.errstate(divide="ignore"):
        d_inv_sqrt = np.power(col_sum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    normalized = d_mat_inv_sqrt.dot(adjacency).dot(d_mat_inv_sqrt).tocoo()
    out_indices = np.stack([normalized.row, normalized.col]).astype(np.int64)
    out_values = normalized.data.astype(np.float64)
    return out_indices, out_values


def coo_to_torch_sparse(
    indices: npt.NDArray[np.int64],
    values: npt.NDArray[np.float64],
    size: tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    """Build a coalesced `torch.sparse_coo_tensor` from COO arrays.

    Thin wrapper around the same ``torch.sparse_coo_tensor(...).coalesce()``
    pattern already used in
    [StructuredLogits][linkingtk.algorithms.wsd._ewiser_structured_logits.StructuredLogits] --
    reused here instead of each GNN linker's own ``_<name>_torch.py``
    reimplementing it.

    Args:
        indices: ``(2, nnz)`` int64 array, e.g. from
            [normalize_adjacency_coo][linkingtk.utils.sparse_gcn.normalize_adjacency_coo].
        values: ``(nnz,)`` edge weights.
        size: ``(num_rows, num_cols)``.
        device: Target device.

    Returns:
        A coalesced sparse COO tensor on ``device``.
    """
    import torch

    idx = torch.from_numpy(indices).long().to(device)
    val = torch.from_numpy(values).float().to(device)
    return torch.sparse_coo_tensor(idx, val, size=size, device=device).coalesce()

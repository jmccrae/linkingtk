"""Unit tests for [linkingtk.utils.sparse_gcn][]."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from linkingtk.utils.sparse_gcn import coo_to_torch_sparse, normalize_adjacency_coo


class TestNormalizeAdjacencyCoo:
    def test_self_loops_added_by_default(self) -> None:
        # A single directed edge 0 -> 1, no self-loops of its own.
        indices = np.array([[0], [1]], dtype=np.int64)
        values = np.array([1.0], dtype=np.float64)

        out_indices, out_values = normalize_adjacency_coo(indices, values, num_nodes=2)

        pairs = {
            (int(out_indices[0, i]), int(out_indices[1, i])) for i in range(out_indices.shape[1])
        }
        assert (0, 0) in pairs
        assert (1, 1) in pairs
        assert (0, 1) in pairs
        assert np.all(np.isfinite(out_values))

    def test_without_self_loops(self) -> None:
        indices = np.array([[0], [1]], dtype=np.int64)
        values = np.array([1.0], dtype=np.float64)

        out_indices, _out_values = normalize_adjacency_coo(
            indices, values, num_nodes=2, add_self_loops=False
        )

        pairs = {
            (int(out_indices[0, i]), int(out_indices[1, i])) for i in range(out_indices.shape[1])
        }
        assert (0, 0) not in pairs

    def test_isolated_node_has_zero_weight_not_nan(self) -> None:
        # Column 0 has zero column-sum (node 0 receives no edges; its only
        # edge is outgoing, to node 1) and self-loops are disabled -> its
        # degree-normalization factor is 0, which would divide-by-zero
        # without the isinf-then-zero guard. A genuinely-zero-weight
        # product is legitimately absent from a sparse matmul's output
        # (not an explicit zero entry), so this checks the *dense*
        # reconstruction is 0.0 there, not NaN.
        indices = np.array([[0], [1]], dtype=np.int64)
        values = np.array([1.0], dtype=np.float64)

        out_indices, out_values = normalize_adjacency_coo(
            indices, values, num_nodes=3, add_self_loops=False
        )

        assert np.all(np.isfinite(out_values))
        dense = np.zeros((3, 3))
        for i in range(out_indices.shape[1]):
            dense[out_indices[0, i], out_indices[1, i]] = out_values[i]
        assert dense[0, 1] == 0.0

    def test_duplicate_entries_are_summed(self) -> None:
        # Edge (0, 1) given twice (weights summing to 3) plus its reverse
        # (1, 0) once at weight 3, so neither row's degree is zero and the
        # normalized entry survives (contrast the isolated-node case
        # above, where a zero-degree row's entries are legitimately
        # absent from the output).
        indices = np.array([[0, 0, 1], [1, 1, 0]], dtype=np.int64)
        values = np.array([1.0, 2.0, 3.0], dtype=np.float64)

        out_indices, _out_values = normalize_adjacency_coo(
            indices, values, num_nodes=2, add_self_loops=False
        )

        pairs = [
            (int(out_indices[0, i]), int(out_indices[1, i])) for i in range(out_indices.shape[1])
        ]
        assert pairs.count((0, 1)) == 1


class TestCooToTorchSparse:
    def test_round_trips_to_dense(self) -> None:
        indices = np.array([[0, 1], [1, 0]], dtype=np.int64)
        values = np.array([0.5, 0.25], dtype=np.float64)

        sparse = coo_to_torch_sparse(indices, values, size=(2, 2), device=torch.device("cpu"))
        dense = sparse.to_dense()

        assert dense[0, 1] == pytest.approx(0.5)
        assert dense[1, 0] == pytest.approx(0.25)
        assert dense[0, 0] == 0.0

    def test_result_is_coalesced(self) -> None:
        indices = np.array([[0], [0]], dtype=np.int64)
        values = np.array([1.0], dtype=np.float64)

        sparse = coo_to_torch_sparse(indices, values, size=(1, 1), device=torch.device("cpu"))

        assert sparse.is_coalesced()

"""Unit tests for the pure-numpy pieces of
[GCNAlignLinker][linkingtk.algorithms.ea.gcn_align.GCNAlignLinker] --
independently testable without torch installed, same precedent as
test_rsn4ea_training.py's private-helper tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from linkingtk.algorithms.ea._gcn_align_training import (
    build_weighted_adjacency,
    compute_relation_functionality,
    sample_negatives,
)

# A 4-node directed cycle (relation 0) plus one "bridge" relation (1) that's
# functional in both directions (each head/tail is unique per triple) --
# gives compute_relation_functionality something non-trivial to compute for
# two different relations.
_TRIPLES = np.array(
    [
        [0, 0, 1],
        [1, 0, 2],
        [2, 0, 3],
        [3, 0, 0],
        [0, 1, 2],
    ],
    dtype=np.int64,
)


class TestComputeRelationFunctionality:
    def test_cycle_relation_is_fully_functional(self) -> None:
        r2f, r2if = compute_relation_functionality(_TRIPLES)

        # Relation 0 has 4 triples, 4 distinct heads, 4 distinct tails.
        assert r2f[0] == 1.0
        assert r2if[0] == 1.0

    def test_bridge_relation_functionality(self) -> None:
        r2f, r2if = compute_relation_functionality(_TRIPLES)

        # Relation 1 has 1 triple, 1 distinct head, 1 distinct tail.
        assert r2f[1] == 1.0
        assert r2if[1] == 1.0

    def test_empty_triples(self) -> None:
        r2f, r2if = compute_relation_functionality(np.empty((0, 3), dtype=np.int64))
        assert r2f == {}
        assert r2if == {}


class TestBuildWeightedAdjacency:
    def test_self_loop_triples_contribute_no_edges(self) -> None:
        triples = np.array([[0, 0, 0]], dtype=np.int64)
        r2f, r2if = compute_relation_functionality(triples)

        indices, values = build_weighted_adjacency(triples, r2f, r2if)

        assert indices.shape == (2, 0)
        assert values.shape == (0,)

    def test_edges_both_directions_respect_min_weight_floor(self) -> None:
        r2f, r2if = compute_relation_functionality(_TRIPLES)

        indices, values = build_weighted_adjacency(_TRIPLES, r2f, r2if, min_weight=0.3)

        assert indices.shape[0] == 2
        assert len(values) == indices.shape[1]
        assert np.all(values >= 0.3)
        # Cycle edge 0->1 and its reverse 1->0 must both be present.
        pairs = {(int(indices[0, i]), int(indices[1, i])) for i in range(indices.shape[1])}
        assert (0, 1) in pairs
        assert (1, 0) in pairs

    def test_duplicate_triples_sum_weights(self) -> None:
        # Both triples are (0, 0, 1): relation 0 has 1 distinct head, 1
        # distinct tail, 2 triples -> r2f == r2if == 0.5, so each triple
        # contributes weight 0.5 to edge (0, 1); summed over 2 duplicate
        # triples that's 1.0.
        triples = np.array([[0, 0, 1], [0, 0, 1]], dtype=np.int64)
        r2f, r2if = compute_relation_functionality(triples)

        indices, values = build_weighted_adjacency(triples, r2f, r2if, min_weight=0.3)

        pairs = [(int(indices[0, i]), int(indices[1, i])) for i in range(indices.shape[1])]
        forward_weight = values[pairs.index((0, 1))]
        assert forward_weight == pytest.approx(1.0)


class TestSampleNegatives:
    def test_output_shapes(self) -> None:
        seed_pairs = [(0, 10), (1, 11), (2, 12)]
        rng = np.random.default_rng(0)

        neg_left, neg_right, neg2_left, neg2_right = sample_negatives(
            seed_pairs, num_entities=20, neg_triple_num=4, rng=rng
        )

        t, k = len(seed_pairs), 4
        assert neg_left.shape == (t * k,)
        assert neg_right.shape == (t * k,)
        assert neg2_left.shape == (t * k,)
        assert neg2_right.shape == (t * k,)

    def test_anchor_side_is_repeated_own_id(self) -> None:
        seed_pairs = [(0, 10), (1, 11)]
        rng = np.random.default_rng(0)

        neg_left, _neg_right, _neg2_left, neg2_right = sample_negatives(
            seed_pairs, num_entities=20, neg_triple_num=3, rng=rng
        )

        assert neg_left.tolist() == [0, 0, 0, 1, 1, 1]
        assert neg2_right.tolist() == [10, 10, 10, 11, 11, 11]

    def test_random_side_within_entity_range(self) -> None:
        seed_pairs = [(0, 10)]
        rng = np.random.default_rng(0)

        _neg_left, neg_right, neg2_left, _neg2_right = sample_negatives(
            seed_pairs, num_entities=5, neg_triple_num=50, rng=rng
        )

        assert neg_right.min() >= 0
        assert neg_right.max() < 5
        assert neg2_left.min() >= 0
        assert neg2_left.max() < 5

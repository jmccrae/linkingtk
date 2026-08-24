"""Unit tests for the pure-numpy pieces of
[AliNetLinker][linkingtk.algorithms.ea.alinet.AliNetLinker] --
independently testable without torch installed, same precedent as
test_rdgcn_training.py's/test_gcn_align_training.py's private-helper tests.
"""

from __future__ import annotations

import numpy as np

from linkingtk.algorithms.ea._alinet_training import (
    enhance_triples,
    generate_2hop_pairs,
    pairs_to_symmetric_adjacency,
    run_bootstrapping_round,
    sample_uniform_cross_kg_negatives,
)


class TestEnhanceTriples:
    def test_infers_bridging_triple(self) -> None:
        triples1 = np.array([[0, 5, 1]], dtype=np.int64)
        triples2 = np.empty((0, 3), dtype=np.int64)
        seed_pairs = [(0, 10), (1, 11)]

        enhanced1, enhanced2 = enhance_triples(triples1, triples2, seed_pairs)

        assert enhanced2 == {(10, 5, 11)}
        assert enhanced1 == set()

    def test_skips_if_target_already_directly_connected(self) -> None:
        triples1 = np.array([[0, 5, 1]], dtype=np.int64)
        triples2 = np.array([[10, 9, 11]], dtype=np.int64)
        seed_pairs = [(0, 10), (1, 11)]

        _enhanced1, enhanced2 = enhance_triples(triples1, triples2, seed_pairs)

        assert enhanced2 == set()

    def test_skips_if_either_entity_unseeded(self) -> None:
        triples1 = np.array([[0, 5, 2]], dtype=np.int64)
        triples2 = np.empty((0, 3), dtype=np.int64)
        seed_pairs = [(0, 10)]  # entity 2 has no counterpart

        _enhanced1, enhanced2 = enhance_triples(triples1, triples2, seed_pairs)

        assert enhanced2 == set()


class TestGenerate2HopPairs:
    def test_direct_neighbor_pair_excluded(self) -> None:
        # 0 -> 1 -> 2, but 0 -> 2 is also a direct edge -> excluded.
        triples = np.array([[0, 1, 1], [1, 2, 2], [0, 3, 2]], dtype=np.int64)
        pairs = generate_2hop_pairs(triples, skip_top_k=0)
        assert (0, 2) not in pairs

    def test_two_hop_pair_included_when_not_direct(self) -> None:
        triples = np.array([[0, 1, 1], [1, 2, 2]], dtype=np.int64)
        pairs = generate_2hop_pairs(triples, skip_top_k=0)
        assert (0, 2) in pairs

    def test_skip_top_k_excludes_most_frequent_pattern(self) -> None:
        # Pattern (1, 2) occurs 3 times (via 0/10/20 -> mid -> 2/12/22),
        # pattern (1, 3) occurs once -- skip_top_k=1 should drop only the
        # frequent pattern's pairs.
        triples = np.array(
            [
                [0, 1, 100],
                [100, 2, 2],
                [10, 1, 101],
                [101, 2, 12],
                [20, 1, 102],
                [102, 2, 22],
                [30, 1, 103],
                [103, 3, 33],
            ],
            dtype=np.int64,
        )
        pairs = generate_2hop_pairs(triples, skip_top_k=1)
        assert (0, 2) not in pairs
        assert (10, 12) not in pairs
        assert (30, 33) in pairs


class TestPairsToSymmetricAdjacency:
    def test_both_directions_present(self) -> None:
        indices, values = pairs_to_symmetric_adjacency({(0, 1)}, num_entities=2)
        pairs = {(int(indices[0, i]), int(indices[1, i])) for i in range(indices.shape[1])}
        assert pairs == {(0, 1), (1, 0)}
        assert np.all(values == 1.0)

    def test_empty(self) -> None:
        indices, values = pairs_to_symmetric_adjacency(set(), num_entities=3)
        assert indices.shape == (2, 0)
        assert values.shape == (0,)


class TestSampleUniformCrossKgNegatives:
    def test_shapes(self) -> None:
        rng = np.random.default_rng(0)
        neg_left, neg_right = sample_uniform_cross_kg_negatives(
            [0, 1, 2], [10, 11, 12], batch_size=4, neg_triple_num=3, rng=rng, exclude_pairs=set()
        )
        assert len(neg_left) == 12
        assert len(neg_right) == 12
        assert set(neg_left.tolist()) <= {0, 1, 2}
        assert set(neg_right.tolist()) <= {10, 11, 12}

    def test_excludes_known_positives(self) -> None:
        rng = np.random.default_rng(0)
        # Only one possible pair, and it's excluded -> nothing survives.
        neg_left, neg_right = sample_uniform_cross_kg_negatives(
            [0], [10], batch_size=5, neg_triple_num=2, rng=rng, exclude_pairs={(0, 10)}
        )
        assert len(neg_left) == 0
        assert len(neg_right) == 0


class TestRunBootstrappingRound:
    def test_finds_reciprocal_nearest_pair(self) -> None:
        # Entity 0 (unaligned1) and entity 10 (unaligned2) share the same
        # embedding -- an obvious reciprocal top-1 match.
        embeddings = np.zeros((20, 4))
        embeddings[0] = [1.0, 0.0, 0.0, 0.0]
        embeddings[10] = [1.0, 0.0, 0.0, 0.0]
        embeddings[1] = [0.0, 1.0, 0.0, 0.0]
        embeddings[11] = [0.0, 0.0, 1.0, 0.0]

        new_seed_pairs, remaining1, remaining2 = run_bootstrapping_round(
            embeddings, seed_pairs=[], unaligned1=[0, 1], unaligned2=[10, 11]
        )

        assert (0, 10) in new_seed_pairs
        assert 0 not in remaining1
        assert 10 not in remaining2

    def test_empty_unaligned_is_a_noop(self) -> None:
        embeddings = np.zeros((5, 2))
        new_seed_pairs, remaining1, remaining2 = run_bootstrapping_round(
            embeddings, seed_pairs=[(0, 1)], unaligned1=[], unaligned2=[2, 3]
        )
        assert new_seed_pairs == [(0, 1)]
        assert remaining1 == []
        assert remaining2 == [2, 3]

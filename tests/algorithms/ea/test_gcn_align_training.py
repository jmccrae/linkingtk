"""Unit tests for the pure-numpy pieces of
[GCNAlignLinker][linkingtk.algorithms.ea.gcn_align.GCNAlignLinker] --
independently testable without torch installed, same precedent as
test_rsn4ea_training.py's private-helper tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from linkingtk.algorithms.ea._gcn_align_training import (
    build_attribute_features,
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


class TestBuildAttributeFeatures:
    def test_entity_gets_column_for_its_predicate(self) -> None:
        entity_to_id = {"kg1:a": 0, "kg1:b": 1, "kg2:w": 2}
        attrs1 = [
            ("kg1:a", "http://dbpedia.org/ontology/birthDate", "1990-01-01"),
            ("kg1:b", "http://dbpedia.org/ontology/birthDate", "1985-01-01"),
        ]
        attrs2: list[tuple[str, str, str]] = []

        # top_fraction=1.0 keeps this deterministic -- see
        # test_small_vocabulary_rounds_down_to_zero_attributes for the
        # default 0.7's rounding behavior on a tiny vocabulary.
        indices, values, num_attrs = build_attribute_features(
            attrs1, attrs2, entity_to_id, top_fraction=1.0
        )

        assert num_attrs == 1
        pairs = {(int(indices[0, i]), int(indices[1, i])) for i in range(indices.shape[1])}
        assert (0, 0) in pairs
        assert (1, 0) in pairs
        assert np.all(values == 1.0)

    def test_small_vocabulary_rounds_down_to_zero_attributes(self) -> None:
        # A real, faithfully-ported OpenEA quirk: int(0.7 * len(predicates))
        # rounds down, so a vocabulary of a single predicate keeps zero
        # feature columns at the default top_fraction.
        entity_to_id = {"kg1:a": 0}
        attrs1 = [("kg1:a", "http://example.org/p", "v")]

        _indices, _values, num_attrs = build_attribute_features(attrs1, [], entity_to_id)

        assert num_attrs == 0

    def test_low_frequency_predicate_dropped_by_top_fraction(self) -> None:
        # 10 distinct predicates -> top_fraction=0.5 keeps only the 5 most
        # frequent (by distinct-entity count). 9 predicates each have
        # frequency 2 (shared with a filler entity); the 10th ("rare_only"'s
        # only predicate) has frequency 1, strictly below all 9 -- so it's
        # dropped regardless of tie-breaking among the frequency-2 group.
        entity_to_id = {"common": 0, "rare_only": 1}
        entity_to_id.update({f"filler{i}": i + 2 for i in range(9)})
        attrs1 = [("common", f"http://example.org/p{p}", "v") for p in range(9)]
        attrs1 += [(f"filler{p}", f"http://example.org/p{p}", "v") for p in range(9)]
        attrs1.append(("rare_only", "http://example.org/p_rare", "v"))

        indices, _values, num_attrs = build_attribute_features(
            attrs1, [], entity_to_id, top_fraction=0.5
        )

        assert num_attrs == 5
        rare_only_id = entity_to_id["rare_only"]
        rare_only_columns = indices[1][indices[0] == rare_only_id]
        assert len(rare_only_columns) == 0

    def test_entity_not_in_entity_to_id_is_dropped(self) -> None:
        entity_to_id = {"kg1:a": 0}
        attrs1 = [("kg1:unknown", "http://example.org/p", "v")]

        indices, values, _num_attrs = build_attribute_features(attrs1, [], entity_to_id)

        assert indices.shape == (2, 0)
        assert values.shape == (0,)

    def test_no_attribute_triples_gives_zero_attributes(self) -> None:
        indices, values, num_attrs = build_attribute_features([], [], {"kg1:a": 0})
        assert num_attrs == 0
        assert indices.shape == (2, 0)
        assert values.shape == (0,)

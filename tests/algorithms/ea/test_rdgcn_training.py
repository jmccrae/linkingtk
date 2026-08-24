"""Unit tests for the pure-numpy pieces of
[RDGCNLinker][linkingtk.algorithms.ea.rdgcn.RDGCNLinker] --
independently testable without torch installed, same precedent as
test_gcn_align_training.py's/test_rsn4ea_training.py's private-helper tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from linkingtk.algorithms.ea._rdgcn_training import (
    build_dual_graph,
    build_edge_relations,
    build_primal_adjacency,
    build_relation_masks,
    init_name_embeddings,
)
from linkingtk.core.entity import Entity

# Relation 0: a 4-node directed cycle (0->1->2->3->0). Relation 1: a single
# disjoint edge (4, 5) sharing no entities with relation 0 -- gives
# build_dual_graph a genuinely-zero off-diagonal pair to check alongside
# relation 0's self-similarity.
_TRIPLES = np.array(
    [
        [0, 0, 1],
        [1, 0, 2],
        [2, 0, 3],
        [3, 0, 0],
        [4, 1, 5],
    ],
    dtype=np.int64,
)


class TestBuildPrimalAdjacency:
    def test_edges_are_symmetric(self) -> None:
        indices, values = build_primal_adjacency(_TRIPLES, num_entities=6)

        pairs = {(int(indices[0, i]), int(indices[1, i])) for i in range(indices.shape[1])}
        assert (0, 1) in pairs
        assert (1, 0) in pairs
        assert np.all(values == 1.0)

    def test_self_loop_triples_contribute_no_edges(self) -> None:
        triples = np.array([[0, 0, 0]], dtype=np.int64)
        indices, values = build_primal_adjacency(triples, num_entities=1)
        assert indices.shape == (2, 0)
        assert values.shape == (0,)

    def test_duplicate_triples_do_not_accumulate_weight(self) -> None:
        triples = np.array([[0, 0, 1], [0, 1, 1]], dtype=np.int64)
        indices, values = build_primal_adjacency(triples, num_entities=2)

        pairs = [(int(indices[0, i]), int(indices[1, i])) for i in range(indices.shape[1])]
        assert pairs.count((0, 1)) == 1
        assert values[pairs.index((0, 1))] == 1.0


class TestBuildDualGraph:
    def test_diagonal_is_two(self) -> None:
        dual = build_dual_graph(_TRIPLES, num_relations=2, num_entities=6)
        assert dual[0, 0] == 2.0
        assert dual[1, 1] == 2.0

    def test_disjoint_relations_have_zero_off_diagonal(self) -> None:
        dual = build_dual_graph(_TRIPLES, num_relations=2, num_entities=6)
        assert dual[0, 1] == 0.0
        assert dual[1, 0] == 0.0

    def test_symmetric(self) -> None:
        dual = build_dual_graph(_TRIPLES, num_relations=2, num_entities=6)
        assert np.allclose(dual, dual.T)


class TestBuildRelationMasks:
    def test_rows_sum_to_one(self) -> None:
        (head_indices, head_values), (tail_indices, tail_values) = build_relation_masks(
            _TRIPLES, num_relations=2
        )

        for relation in (0, 1):
            head_row_sum = head_values[head_indices[0] == relation].sum()
            tail_row_sum = tail_values[tail_indices[0] == relation].sum()
            assert head_row_sum == pytest.approx(1.0)
            assert tail_row_sum == pytest.approx(1.0)

    def test_relation_0_has_four_distinct_heads(self) -> None:
        (head_indices, _head_values), _tail = build_relation_masks(_TRIPLES, num_relations=2)
        heads_of_0 = set(head_indices[1][head_indices[0] == 0].tolist())
        assert heads_of_0 == {0, 1, 2, 3}


class TestBuildEdgeRelations:
    def test_shapes_and_values_match_input(self) -> None:
        heads, tails, relations = build_edge_relations(_TRIPLES)

        assert heads.tolist() == _TRIPLES[:, 0].tolist()
        assert tails.tolist() == _TRIPLES[:, 2].tolist()
        assert relations.tolist() == _TRIPLES[:, 1].tolist()

    def test_no_deduplication(self) -> None:
        triples = np.array([[0, 0, 1], [0, 0, 1]], dtype=np.int64)
        heads, _tails, _relations = build_edge_relations(triples)
        assert len(heads) == 2

    def test_empty_triples(self) -> None:
        heads, tails, relations = build_edge_relations(np.empty((0, 3), dtype=np.int64))
        assert len(heads) == 0
        assert len(tails) == 0
        assert len(relations) == 0


class TestInitNameEmbeddings:
    def test_sums_known_word_vectors(self) -> None:
        entities = [Entity(id="e1", labels=["red car"])]
        word_vectors = {
            "red": np.array([1.0, 0.0], dtype=np.float32),
            "car": np.array([0.0, 1.0], dtype=np.float32),
        }

        result = init_name_embeddings(entities, word_vectors, dim=2)

        assert np.allclose(result["e1"], [1.0, 1.0])

    def test_unknown_words_contribute_zero(self) -> None:
        entities = [Entity(id="e1", labels=["totally unknown"])]
        result = init_name_embeddings(entities, word_vectors={}, dim=3)
        assert np.allclose(result["e1"], [0.0, 0.0, 0.0])

    def test_entity_with_no_labels_gets_zero_vector(self) -> None:
        entities = [Entity(id="e1", labels=[])]
        result = init_name_embeddings(entities, word_vectors={}, dim=4)
        assert np.allclose(result["e1"], np.zeros(4))

    def test_truncates_to_default_length(self) -> None:
        entities = [Entity(id="e1", labels=["a b c d e"])]
        word_vectors = {word: np.array([1.0]) for word in "abcde"}

        result = init_name_embeddings(entities, word_vectors, dim=1, default_length=3)

        # Only "a", "b", "c" contribute -- "d", "e" are truncated away.
        assert result["e1"][0] == 3.0

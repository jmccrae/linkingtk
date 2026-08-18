from __future__ import annotations

import numpy as np
import pytest
import torch

from linkingtk.algorithms.ea import BootEALinker
from linkingtk.algorithms.ea._bootea_training import (
    compute_truncated_neighbors,
    edit_labeled_alignment,
    find_mwgm_pairs,
    pseudo_triples_for_pairs,
    sample_truncated_negative_triples,
)
from linkingtk.blocking.base import BlockingStrategy
from linkingtk.core.entity import Entity
from linkingtk.eval import Evaluator
from linkingtk.exceptions import LinkingTKError

# Two isomorphic 4-node chains ("next"-linked), same fixture as
# test_iptranse.py's -- a pipeline-correctness check (does id-mapping,
# seed-pseudo-triple generation, negative sampling, training loop, link()
# wire up correctly), not a generalization benchmark. See
# test_bootea_benchmark.py for held-out generalization at real-dataset
# shape.
_KG1 = [Entity(id=f"kg1:{c}", labels=[c]) for c in "abcd"]
_KG2 = [Entity(id=f"kg2:{c}", labels=[c]) for c in "wxyz"]
_GRAPH = [
    ("kg1:a", "next", "kg1:b"),
    ("kg1:b", "next", "kg1:c"),
    ("kg1:c", "next", "kg1:d"),
    ("kg2:w", "next", "kg2:x"),
    ("kg2:x", "next", "kg2:y"),
    ("kg2:y", "next", "kg2:z"),
]
_GROUND_TRUTH = [("kg1:a", "kg2:w"), ("kg1:b", "kg2:x"), ("kg1:c", "kg2:y"), ("kg1:d", "kg2:z")]


class _AllPairs(BlockingStrategy):
    """Blocking strategy that lets every pair through -- entities here don't
    share labels across KGs, so ExactMatch's default blocking would find
    nothing.
    """

    def candidate_pairs(
        self, dataset1: list[Entity], dataset2: list[Entity]
    ) -> list[tuple[Entity, Entity]]:
        return [(e1, e2) for e1 in dataset1 for e2 in dataset2]


class TestFitAndLink:
    def test_recovers_seeded_alignment(self) -> None:
        # Full seeding: sub_epoch > num_epochs disables bootstrapping
        # entirely, isolating this to the seed-pseudo-triple + structural
        # loss pipeline -- checks the pipeline (id-mapping, pseudo-triple
        # generation, truncated negative sampling, training loop, link())
        # runs end-to-end and converges, not learned generalization (see
        # test_partial_seed_generalizes_to_unseeded_pairs for that).
        linker = BootEALinker(
            embedding_dim=16,
            num_epochs=200,
            sub_epoch=1000,
            batch_size=32,
            learning_rate=0.1,
            neg_triple_num=2,
        )
        linker.fit(_KG1, _KG2, _GROUND_TRUTH, graph=_GRAPH, random_state=0)

        results = linker.link(_KG1, _KG2, blocking=_AllPairs())
        predictions = [(r.source_id, r.target_id) for r in results]

        report = Evaluator.evaluate(predictions=predictions, ground_truth=_GROUND_TRUTH)
        assert report.metrics["precision@1"] == 1.0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
    def test_recovers_seeded_alignment_on_cuda(self) -> None:
        linker = BootEALinker(
            embedding_dim=16,
            num_epochs=200,
            sub_epoch=1000,
            batch_size=32,
            learning_rate=0.1,
            neg_triple_num=2,
            device="cuda",
        )
        linker.fit(_KG1, _KG2, _GROUND_TRUTH, graph=_GRAPH, random_state=0)

        results = linker.link(_KG1, _KG2, blocking=_AllPairs())
        predictions = [(r.source_id, r.target_id) for r in results]

        report = Evaluator.evaluate(predictions=predictions, ground_truth=_GROUND_TRUTH)
        assert report.metrics["precision@1"] == 1.0

    def test_partial_seed_generalizes_to_unseeded_pairs(self) -> None:
        # Seed only 2 of 4 pairs; sub_epoch > num_epochs disables
        # bootstrapping, isolating this test to pure seed-pseudo-triple +
        # structural learning. The two seeded anchors' pseudo-triples pull
        # both isomorphic chains into one coordinate frame, so the
        # remaining two pairs should be recoverable from structure alone.
        partial_ground_truth = [("kg1:a", "kg2:w"), ("kg1:c", "kg2:y")]
        linker = BootEALinker(
            embedding_dim=16,
            num_epochs=300,
            sub_epoch=1000,
            batch_size=32,
            learning_rate=0.1,
            neg_triple_num=2,
        )
        linker.fit(_KG1, _KG2, partial_ground_truth, graph=_GRAPH, random_state=0)

        results = linker.link(_KG1, _KG2, blocking=_AllPairs())
        predictions = [(r.source_id, r.target_id) for r in results]

        report = Evaluator.evaluate(predictions=predictions, ground_truth=_GROUND_TRUTH)
        assert report.metrics["precision@1"] == 1.0

    def test_bootstrapping_runs_without_error(self) -> None:
        # sim_th=0.0 and a generous k guarantee a bootstrap round finds
        # matches. Not asserting which pairs it finds (a timing/threshold
        # -sensitive internal detail) -- just that the full pipeline runs
        # without error and still produces a usable fitted linker, same
        # precedent as test_iptranse.py's test_bootstrapping_runs_without_error.
        linker = BootEALinker(
            embedding_dim=16,
            num_epochs=20,
            sub_epoch=5,
            batch_size=32,
            neg_triple_num=2,
            sim_th=0.0,
            k=10,
        )
        linker.fit(_KG1, _KG2, [("kg1:a", "kg2:w")], graph=_GRAPH, random_state=0)

        results = linker.link(_KG1, _KG2, blocking=_AllPairs())
        assert {r.source_id for r in results} == {entity.id for entity in _KG1}

    def test_early_stopping_runs_without_error(self) -> None:
        linker = BootEALinker(
            embedding_dim=16,
            num_epochs=100,
            sub_epoch=1000,
            batch_size=32,
            neg_triple_num=2,
        )
        linker.fit(
            _KG1,
            _KG2,
            _GROUND_TRUTH,
            graph=_GRAPH,
            random_state=0,
            val_ground_truth=_GROUND_TRUTH,
            patience=1,
            eval_every=5,
        )

        results = linker.link(_KG1, _KG2, blocking=_AllPairs())
        assert {r.source_id for r in results} == {entity.id for entity in _KG1}


class TestErrors:
    def test_link_before_fit_raises(self) -> None:
        linker = BootEALinker()
        with pytest.raises(LinkingTKError, match="before fit"):
            linker.link(_KG1, _KG2, blocking=_AllPairs())

    def test_fit_with_no_ground_truth_ids_in_graph_raises(self) -> None:
        linker = BootEALinker()
        bogus_ground_truth = [("not:an:id", "also:not:an:id")]
        with pytest.raises(LinkingTKError, match="no seed pairs"):
            linker.fit(_KG1, _KG2, bogus_ground_truth, graph=_GRAPH, random_state=0)

    def test_link_with_entity_absent_from_training_graph_raises(self) -> None:
        linker = BootEALinker(
            embedding_dim=8, num_epochs=5, sub_epoch=1000, batch_size=32, neg_triple_num=2
        )
        linker.fit(_KG1, _KG2, _GROUND_TRUTH, graph=_GRAPH, random_state=0)

        unseen = Entity(id="kg1:unseen", labels=["unseen"])
        with pytest.raises(LinkingTKError, match="no trained embedding"):
            linker.link([unseen], _KG2, blocking=_AllPairs())


class TestPseudoTriplesForPairs:
    def test_swaps_each_direction_into_its_own_kg_side(self) -> None:
        # KG1: a --r--> b. KG2: c --s--> d. Pair (a, c) should produce a
        # KG1-side triple (c, r, b) [a's real edge, a replaced by c] and a
        # KG2-side triple (a, s, d) [c's real edge, c replaced by a].
        by_head1 = {0: [(10, 1)]}  # a(=0) --r(=10)--> b(=1)
        by_tail1: dict[int, list[tuple[int, int]]] = {}
        by_head2 = {2: [(20, 3)]}  # c(=2) --s(=20)--> d(=3)
        by_tail2: dict[int, list[tuple[int, int]]] = {}

        triples1, triples2 = pseudo_triples_for_pairs(
            [(0, 2)], by_head1, by_tail1, by_head2, by_tail2
        )

        assert triples1 == [(2, 10, 1)]
        assert triples2 == [(0, 20, 3)]

    def test_no_edges_gives_no_pseudo_triples(self) -> None:
        triples1, triples2 = pseudo_triples_for_pairs([(0, 1)], {}, {}, {}, {})
        assert triples1 == []
        assert triples2 == []


class TestFindMwgmPairs:
    def test_resolves_conflicting_row_argmax_to_a_true_assignment(self) -> None:
        # Both rows' best match is column 0 -- plain row-argmax would
        # (wrongly) map both to column 0; MWGM must instead pick the
        # single globally-best 1-to-1 assignment.
        sim_mat = np.array([[0.9, 0.85], [0.95, 0.1]])

        pairs = find_mwgm_pairs(sim_mat, sim_th=0.5, k=10)

        assert {(i, j) for i, j, _ in pairs} == {(0, 1), (1, 0)}

    def test_threshold_excludes_low_similarity_candidates(self) -> None:
        sim_mat = np.array([[0.9, 0.1], [0.2, 0.3]])

        pairs = find_mwgm_pairs(sim_mat, sim_th=0.8, k=10)

        assert {(i, j) for i, j, _ in pairs} == {(0, 0)}

    def test_k_limits_candidates_per_row(self) -> None:
        sim_mat = np.array([[0.9, 0.85, 0.8]])

        pairs_k1 = find_mwgm_pairs(sim_mat, sim_th=0.0, k=1)
        assert {(i, j) for i, j, _ in pairs_k1} == {(0, 0)}

    def test_empty_matrix_returns_no_pairs(self) -> None:
        assert find_mwgm_pairs(np.empty((0, 0)), sim_th=0.5, k=10) == []


class TestEditLabeledAlignment:
    def test_higher_similarity_replacement_overwrites_earlier_label(self) -> None:
        sim_mat = np.array([[0.5, 0.9]])
        pre = {0: 0}

        updated = edit_labeled_alignment(pre, {(0, 1)}, sim_mat)

        assert updated == {0: 1}

    def test_lower_similarity_replacement_is_rejected(self) -> None:
        sim_mat = np.array([[0.9, 0.5]])
        pre = {0: 0}

        updated = edit_labeled_alignment(pre, {(0, 1)}, sim_mat)

        assert updated == {0: 0}

    def test_multiple_sources_claiming_same_target_keeps_highest_similarity(self) -> None:
        sim_mat = np.array([[0.6, 0.0], [0.9, 0.0]])

        updated = edit_labeled_alignment({}, {(0, 0), (1, 0)}, sim_mat)

        assert updated == {1: 0}


class TestSampleTruncatedNegativeTriples:
    def test_uses_neighbor_candidates_when_available(self) -> None:
        positive = np.array([[0, 1, 2]], dtype=np.int64)
        rng = np.random.default_rng(0)

        negatives = sample_truncated_negative_triples(
            positive,
            neighbor_candidates={0: np.array([9]), 2: np.array([9])},
            fallback_pool=np.array([5]),
            real_triples=np.empty(0, dtype=np.int64),
            rng=rng,
        )

        assert negatives[0, 0] in (0, 9)
        assert negatives[0, 2] in (2, 9)
        assert 5 not in negatives[0]

    def test_falls_back_to_pool_for_unknown_entity(self) -> None:
        positive = np.array([[0, 1, 2]], dtype=np.int64)
        rng = np.random.default_rng(0)

        negatives = sample_truncated_negative_triples(
            positive,
            neighbor_candidates={},
            fallback_pool=np.array([7]),
            real_triples=np.empty(0, dtype=np.int64),
            rng=rng,
        )

        assert negatives[0, 0] in (0, 7)
        assert negatives[0, 2] in (2, 7)


class TestComputeTruncatedNeighbors:
    def test_returns_k_nearest_restricted_to_entity_ids(self) -> None:
        embeds = np.array(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
                [-1.0, 0.0],
            ]
        )
        entity_ids = np.array([0, 1, 2], dtype=np.int64)

        neighbors = compute_truncated_neighbors(embeds, entity_ids, k=1)

        assert set(neighbors) == {0, 1, 2}
        assert 3 not in neighbors[0]
        # entity 0's nearest (excluding a tie with itself) among {0,1,2} is itself or 1.
        assert neighbors[0][0] in (0, 1)

    def test_empty_entity_ids_returns_empty_dict(self) -> None:
        embeds = np.zeros((3, 2))
        assert compute_truncated_neighbors(embeds, np.empty(0, dtype=np.int64), k=1) == {}

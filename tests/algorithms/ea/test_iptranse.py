import numpy as np
import pytest

from linkingtk.algorithms.ea import IPTransELinker
from linkingtk.algorithms.ea._iptranse_training import (
    build_shared_id_mappings,
    find_new_pairs,
    generate_two_step_paths,
)
from linkingtk.blocking.base import BlockingStrategy
from linkingtk.core.entity import Entity
from linkingtk.eval import Evaluator
from linkingtk.exceptions import LinkingTKError

# Two isomorphic 4-node chains ("next"-linked), same fixture as
# test_mtranse.py's -- a pipeline-correctness check (does training/
# scoring/matching wire up correctly), not a generalization benchmark. See
# test_iptranse_benchmark.py for held-out generalization at real-dataset
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
        # With full seeding, IPTransE's "sharing" mode merges every
        # entity 1:1 by construction -- this mainly checks the pipeline
        # (id-merge, path generation, negative sampling, training loop,
        # link()) runs end-to-end without shape errors, not learned
        # generalization (see test_partial_seed_generalizes_to_unseeded_pairs
        # for that).
        linker = IPTransELinker(
            embedding_dim=16, num_epochs=100, batch_size=32, bootstrap_every=1000
        )
        linker.fit(_KG1, _KG2, _GROUND_TRUTH, graph=_GRAPH, random_state=0)

        results = linker.link(_KG1, _KG2, blocking=_AllPairs())
        predictions = [(r.source_id, r.target_id) for r in results]

        report = Evaluator.evaluate(predictions=predictions, ground_truth=_GROUND_TRUTH)
        assert report.metrics["precision@1"] == 1.0

    def test_partial_seed_generalizes_to_unseeded_pairs(self) -> None:
        # Seed only 2 of 4 pairs; bootstrap_every > num_epochs disables
        # bootstrapping entirely, isolating this test to pure
        # triple+path-loss structural learning. The two seeded anchors
        # already pin both isomorphic chains into one coordinate frame
        # (kg1:a and kg2:w share one embedding row, likewise kg1:c/kg2:y),
        # so the remaining two pairs should be recoverable from structure
        # alone.
        partial_ground_truth = [("kg1:a", "kg2:w"), ("kg1:c", "kg2:y")]
        linker = IPTransELinker(
            embedding_dim=16,
            num_epochs=300,
            batch_size=32,
            learning_rate=0.1,
            bootstrap_every=1000,
        )
        linker.fit(_KG1, _KG2, partial_ground_truth, graph=_GRAPH, random_state=0)

        results = linker.link(_KG1, _KG2, blocking=_AllPairs())
        predictions = [(r.source_id, r.target_id) for r in results]

        report = Evaluator.evaluate(predictions=predictions, ground_truth=_GROUND_TRUTH)
        assert report.metrics["precision@1"] == 1.0

    def test_bootstrapping_runs_without_error(self) -> None:
        # sim_th=0.0 guarantees every reference-pool row's top-1 neighbor
        # clears the threshold, so a bootstrap round is guaranteed to
        # fire and find pairs. Not asserting which pairs it finds (that's
        # a timing/threshold-sensitive internal detail) -- just that the
        # full pipeline runs without error and still produces a usable
        # fitted linker, same precedent as test_mtranse.py's
        # test_early_stopping_runs_without_error.
        linker = IPTransELinker(
            embedding_dim=16,
            num_epochs=20,
            batch_size=32,
            bootstrap_every=5,
            sim_th=0.0,
        )
        linker.fit(_KG1, _KG2, [("kg1:a", "kg2:w")], graph=_GRAPH, random_state=0)

        results = linker.link(_KG1, _KG2, blocking=_AllPairs())
        assert {r.source_id for r in results} == {entity.id for entity in _KG1}

    def test_early_stopping_runs_without_error(self) -> None:
        linker = IPTransELinker(
            embedding_dim=16, num_epochs=100, batch_size=32, bootstrap_every=1000
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
        linker = IPTransELinker()
        with pytest.raises(LinkingTKError, match="before fit"):
            linker.link(_KG1, _KG2, blocking=_AllPairs())

    def test_fit_with_no_ground_truth_ids_in_graph_raises(self) -> None:
        linker = IPTransELinker()
        bogus_ground_truth = [("not:an:id", "also:not:an:id")]
        with pytest.raises(LinkingTKError, match="no seed pairs"):
            linker.fit(_KG1, _KG2, bogus_ground_truth, graph=_GRAPH, random_state=0)

    def test_link_with_entity_absent_from_training_graph_raises(self) -> None:
        linker = IPTransELinker(embedding_dim=8, num_epochs=5, batch_size=32, bootstrap_every=1000)
        linker.fit(_KG1, _KG2, _GROUND_TRUTH, graph=_GRAPH, random_state=0)

        unseen = Entity(id="kg1:unseen", labels=["unseen"])
        with pytest.raises(LinkingTKError, match="no trained embedding"):
            linker.link([unseen], _KG2, blocking=_AllPairs())


class TestBuildSharedIdMappings:
    def test_seed_pair_shares_one_id(self) -> None:
        triples = [("a", "r", "b"), ("c", "r", "d")]
        entity_to_id, _ = build_shared_id_mappings(triples, seed_pairs=[("a", "c")])

        assert entity_to_id["a"] == entity_to_id["c"]
        assert len({entity_to_id["a"], entity_to_id["b"], entity_to_id["d"]}) == 3
        assert len(set(entity_to_id.values())) == 3

    def test_no_seed_pairs_keeps_all_entities_distinct(self) -> None:
        triples = [("a", "r", "b"), ("c", "r", "d")]
        entity_to_id, _ = build_shared_id_mappings(triples, seed_pairs=[])

        assert len(set(entity_to_id.values())) == 4

    def test_relations_are_not_merged(self) -> None:
        triples = [("a", "r1", "b"), ("c", "r1", "d")]
        _, relation_to_id = build_shared_id_mappings(triples, seed_pairs=[("a", "c")])

        assert relation_to_id == {"r1": 0}


class TestGenerateTwoStepPaths:
    def test_finds_composed_path_with_direct_edge(self) -> None:
        triples = [("h", "r1", "m"), ("m", "r2", "t"), ("h", "rd", "t")]

        paths = generate_two_step_paths(triples)

        assert paths == [("r1", "r2", "rd", 1)]

    def test_no_direct_edge_means_no_path(self) -> None:
        triples = [("h", "r1", "m"), ("m", "r2", "t")]

        assert generate_two_step_paths(triples) == []

    def test_high_weight_chain_is_dropped(self) -> None:
        # fanout(h, r1) = 100, fanout(m, r2) = 2 -> weight 200, >= 101 cutoff.
        triples = [("h", "r1", f"m{i}") for i in range(100)]
        triples += [("m0", "r2", "t1"), ("m0", "r2", "t2")]
        triples += [("h", "rd", "t1")]

        assert generate_two_step_paths(triples) == []


class TestFindNewPairs:
    def test_row_wise_top1_above_threshold(self) -> None:
        sim_mat = np.array([[0.9, 0.1], [0.2, 0.95], [0.5, 0.5]])

        pairs = find_new_pairs(sim_mat, sim_th=0.6)

        assert pairs == [(0, 0, pytest.approx(0.9)), (1, 1, pytest.approx(0.95))]

    def test_not_mutual_matching(self) -> None:
        sim_mat = np.array([[0.9, 0.1], [0.8, 0.2]])

        pairs = find_new_pairs(sim_mat, sim_th=0.5)

        assert [(i, j) for i, j, _ in pairs] == [(0, 0), (1, 0)]

    def test_empty_matrix_returns_no_pairs(self) -> None:
        assert find_new_pairs(np.empty((0, 0)), sim_th=0.5) == []

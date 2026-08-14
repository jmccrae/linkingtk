import numpy as np
import pytest
import torch

from linkingtk.algorithms.ea import JAPELinker
from linkingtk.algorithms.ea._jape_training import (
    build_entity_attributes,
    generate_training_pairs,
    pool_entity_attribute_vectors,
    reference_pools,
    select_popular_attributes,
    sparsify_to_row_argmax,
)
from linkingtk.blocking.base import BlockingStrategy
from linkingtk.core.entity import Entity
from linkingtk.eval import Evaluator
from linkingtk.exceptions import LinkingTKError

# Two isomorphic 4-node chains ("next"-linked), same fixture as
# test_mtranse.py's/test_iptranse.py's -- a pipeline-correctness check, not
# a generalization benchmark. See test_jape_benchmark.py for held-out
# generalization at real-dataset shape.
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

# Two attributes per entity (attribute correlation needs >=2 per entity to
# generate any skip-gram training pairs). KG2 deliberately uses different
# predicate names ("colour"/"dimension" vs "color"/"size") -- exercising
# the cross-lingual attribute-vocabulary merge that seed pairs enable.
_ATTR1 = [
    ("kg1:a", "color", "red"),
    ("kg1:a", "size", "small"),
    ("kg1:b", "color", "blue"),
    ("kg1:b", "size", "medium"),
    ("kg1:c", "color", "green"),
    ("kg1:c", "size", "large"),
    ("kg1:d", "color", "yellow"),
    ("kg1:d", "size", "xlarge"),
]
_ATTR2 = [
    ("kg2:w", "colour", "red"),
    ("kg2:w", "dimension", "small"),
    ("kg2:x", "colour", "blue"),
    ("kg2:x", "dimension", "medium"),
    ("kg2:y", "colour", "green"),
    ("kg2:y", "dimension", "large"),
    ("kg2:z", "colour", "yellow"),
    ("kg2:z", "dimension", "xlarge"),
]


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
        linker = JAPELinker(
            embedding_dim=16,
            num_epochs=100,
            batch_size=32,
            attr_max_epoch=20,
            top_attr_threshold=1.0,
        )
        linker.fit(
            _KG1,
            _KG2,
            _GROUND_TRUTH,
            graph=_GRAPH,
            random_state=0,
            attribute_triples1=_ATTR1,
            attribute_triples2=_ATTR2,
        )

        results = linker.link(_KG1, _KG2, blocking=_AllPairs())
        predictions = [(r.source_id, r.target_id) for r in results]

        report = Evaluator.evaluate(predictions=predictions, ground_truth=_GROUND_TRUTH)
        assert report.metrics["precision@1"] == 1.0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
    def test_recovers_seeded_alignment_on_cuda(self) -> None:
        linker = JAPELinker(
            embedding_dim=16,
            num_epochs=100,
            batch_size=32,
            attr_max_epoch=20,
            top_attr_threshold=1.0,
            device="cuda",
        )
        linker.fit(
            _KG1,
            _KG2,
            _GROUND_TRUTH,
            graph=_GRAPH,
            random_state=0,
            attribute_triples1=_ATTR1,
            attribute_triples2=_ATTR2,
        )

        results = linker.link(_KG1, _KG2, blocking=_AllPairs())
        predictions = [(r.source_id, r.target_id) for r in results]

        report = Evaluator.evaluate(predictions=predictions, ground_truth=_GROUND_TRUTH)
        assert report.metrics["precision@1"] == 1.0

    def test_structural_only_when_no_attributes_given(self) -> None:
        linker = JAPELinker(embedding_dim=16, num_epochs=100, batch_size=32)
        linker.fit(_KG1, _KG2, _GROUND_TRUTH, graph=_GRAPH, random_state=0)

        results = linker.link(_KG1, _KG2, blocking=_AllPairs())
        predictions = [(r.source_id, r.target_id) for r in results]

        report = Evaluator.evaluate(predictions=predictions, ground_truth=_GROUND_TRUTH)
        assert report.metrics["precision@1"] == 1.0

    def test_partial_seed_generalizes_to_unseeded_pairs(self) -> None:
        # Seed only 2 of 4 pairs -- attributes plus the structural signal
        # together should still recover the rest (mirrors IPTransE's
        # equivalent test). sub_mat_size=2 matches the resulting
        # reference-pool size (2 entities/side) so the attribute
        # regularization step actually fires every epoch.
        partial_ground_truth = [("kg1:a", "kg2:w"), ("kg1:c", "kg2:y")]
        linker = JAPELinker(
            embedding_dim=16,
            num_epochs=300,
            batch_size=32,
            learning_rate=0.1,
            attr_max_epoch=20,
            top_attr_threshold=1.0,
            sub_mat_size=2,
        )
        linker.fit(
            _KG1,
            _KG2,
            partial_ground_truth,
            graph=_GRAPH,
            random_state=0,
            attribute_triples1=_ATTR1,
            attribute_triples2=_ATTR2,
        )

        results = linker.link(_KG1, _KG2, blocking=_AllPairs())
        predictions = [(r.source_id, r.target_id) for r in results]

        report = Evaluator.evaluate(predictions=predictions, ground_truth=_GROUND_TRUTH)
        assert report.metrics["precision@1"] == 1.0

    def test_early_stopping_runs_without_error(self) -> None:
        linker = JAPELinker(embedding_dim=16, num_epochs=100, batch_size=32)
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
        linker = JAPELinker()
        with pytest.raises(LinkingTKError, match="before fit"):
            linker.link(_KG1, _KG2, blocking=_AllPairs())

    def test_fit_with_no_ground_truth_ids_in_graph_raises(self) -> None:
        linker = JAPELinker()
        bogus_ground_truth = [("not:an:id", "also:not:an:id")]
        with pytest.raises(LinkingTKError, match="no seed pairs"):
            linker.fit(_KG1, _KG2, bogus_ground_truth, graph=_GRAPH, random_state=0)

    def test_link_with_entity_absent_from_training_graph_raises(self) -> None:
        linker = JAPELinker(embedding_dim=8, num_epochs=5, batch_size=32)
        linker.fit(_KG1, _KG2, _GROUND_TRUTH, graph=_GRAPH, random_state=0)

        unseen = Entity(id="kg1:unseen", labels=["unseen"])
        with pytest.raises(LinkingTKError, match="no trained embedding"):
            linker.link([unseen], _KG2, blocking=_AllPairs())


class TestSelectPopularAttributes:
    def test_keeps_top_threshold_fraction_by_frequency(self) -> None:
        triples1 = [("e1", "a", "v"), ("e1", "a", "v"), ("e1", "b", "v"), ("e2", "c", "v")]

        selected = select_popular_attributes(triples1, [], threshold=1 / 3)

        assert selected == {"a"}

    def test_unions_across_both_kgs(self) -> None:
        selected = select_popular_attributes([("e1", "a", "v")], [("e2", "b", "v")], threshold=1.0)

        assert selected == {"a", "b"}


class TestBuildEntityAttributes:
    def test_merges_seed_pair_attribute_sets(self) -> None:
        triples1 = [("e1", "a", "v"), ("e1", "b", "v")]
        triples2 = [("e2", "c", "v")]

        merged = build_entity_attributes(triples1, triples2, {"a", "b", "c"}, [("e1", "e2")])

        assert merged["e1"] == {"a", "b", "c"}
        assert merged["e2"] == {"a", "b", "c"}

    def test_omits_entities_with_no_selected_attributes(self) -> None:
        merged = build_entity_attributes(
            [("e1", "a", "v")], [], selected_attributes={"b"}, seed_pairs=[]
        )

        assert merged == {}


class TestGenerateTrainingPairs:
    def test_all_pairwise_combinations_per_entity(self) -> None:
        pairs = generate_training_pairs({"e1": {"a", "b", "c"}})

        assert set(pairs) == {("a", "b"), ("a", "c"), ("b", "c")}
        assert len(pairs) == 3


class TestPoolEntityAttributeVectors:
    def test_mean_pools_and_normalizes(self) -> None:
        attr_to_id = {"a": 0, "b": 1}
        attr_embeds = np.array([[3.0, 4.0], [0.0, 0.0]])

        vectors = pool_entity_attribute_vectors(
            ["e1", "e2"], {"e1": {"a"}}, attr_to_id, attr_embeds
        )

        assert np.allclose(vectors[0], [0.6, 0.8])
        assert np.allclose(vectors[1], [0.0, 0.0])


class TestSparsifyToRowArgmax:
    def test_keeps_only_each_rows_maximum(self) -> None:
        sim_mat = np.array([[0.9, 0.8, 0.85], [0.1, 0.95, 0.2]])

        sparse = sparsify_to_row_argmax(sim_mat)

        assert np.array_equal(sparse, np.array([[0.9, 0.0, 0.0], [0.0, 0.95, 0.0]]))

    def test_all_zero_row_stays_zero(self) -> None:
        sim_mat = np.array([[0.0, 0.0], [0.5, 0.3]])

        sparse = sparsify_to_row_argmax(sim_mat)

        assert np.array_equal(sparse, np.array([[0.0, 0.0], [0.5, 0.0]]))

    def test_empty_matrix(self) -> None:
        assert sparsify_to_row_argmax(np.empty((0, 0))).shape == (0, 0)


class TestReferencePools:
    def test_excludes_seed_pair_endpoints(self) -> None:
        pool1, pool2 = reference_pools(["e1", "e2", "e3"], ["f1", "f2"], [("e1", "f1")])

        assert pool1 == ["e2", "e3"]
        assert pool2 == ["f2"]

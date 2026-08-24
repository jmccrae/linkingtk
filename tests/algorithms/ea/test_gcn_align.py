import numpy as np
import pytest
import torch

from linkingtk.algorithms.ea import GCNAlignLinker
from linkingtk.blocking.base import BlockingStrategy
from linkingtk.core.entity import Entity
from linkingtk.eval import Evaluator
from linkingtk.exceptions import LinkingTKError

# Two isomorphic 4-node chains ("next"-linked), fully seeded (every
# ground-truth pair is given to fit() as a seed pair) -- this is a
# pipeline-correctness check (does training/scoring/matching wire up
# correctly and recover what it was directly taught), not a generalization
# benchmark. See test_gcn_align_benchmark.py for held-out generalization.
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

_ATTR1 = [(e.id, "http://example.org/type", "Thing") for e in _KG1] + [
    ("kg1:a", "http://example.org/special", "yes")
]
_ATTR2 = [(e.id, "http://example.org/type", "Thing") for e in _KG2] + [
    ("kg2:w", "http://example.org/special", "yes")
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
        linker = GCNAlignLinker(
            embedding_dim=16, num_epochs=300, learning_rate=0.5, neg_triple_num=3
        )
        linker.fit(_KG1, _KG2, _GROUND_TRUTH, graph=_GRAPH, random_state=0)

        results = linker.link(_KG1, _KG2, blocking=_AllPairs())
        predictions = [(r.source_id, r.target_id) for r in results]

        report = Evaluator.evaluate(predictions=predictions, ground_truth=_GROUND_TRUTH)
        assert report.metrics["precision@1"] == 1.0

    def test_early_stopping_runs_without_error(self) -> None:
        linker = GCNAlignLinker(embedding_dim=16, num_epochs=100, learning_rate=0.5)
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

    def test_source_embedding_equals_target_embedding(self) -> None:
        # GCN-Align has no learned mapping matrix -- both sides share the
        # same propagated embedding table, unlike MTransE.
        linker = GCNAlignLinker(embedding_dim=16, num_epochs=20, learning_rate=0.5)
        linker.fit(_KG1, _KG2, _GROUND_TRUTH, graph=_GRAPH, random_state=0)

        entity_id = _KG1[0].id
        assert np.array_equal(
            linker.source_embedding(entity_id), linker.target_embedding(entity_id)
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
    def test_recovers_seeded_alignment_on_cuda(self) -> None:
        linker = GCNAlignLinker(
            embedding_dim=16, num_epochs=300, learning_rate=0.5, neg_triple_num=3, device="cuda"
        )
        linker.fit(_KG1, _KG2, _GROUND_TRUTH, graph=_GRAPH, random_state=0)

        results = linker.link(_KG1, _KG2, blocking=_AllPairs())
        predictions = [(r.source_id, r.target_id) for r in results]

        report = Evaluator.evaluate(predictions=predictions, ground_truth=_GROUND_TRUTH)
        assert report.metrics["precision@1"] == 1.0


class TestAttributeBranch:
    def test_recovers_seeded_alignment_with_attributes(self) -> None:
        linker = GCNAlignLinker(
            embedding_dim=16,
            attr_dim=16,
            num_epochs=300,
            learning_rate=0.5,
            neg_triple_num=3,
            use_attributes=True,
            attr_top_fraction=1.0,
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

    def test_combined_embedding_is_wider_than_structural_only(self) -> None:
        linker = GCNAlignLinker(
            embedding_dim=16,
            attr_dim=8,
            num_epochs=20,
            learning_rate=0.5,
            use_attributes=True,
            attr_top_fraction=1.0,
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

        vector = linker.source_embedding(_KG1[0].id)
        assert vector.shape == (16 + 8,)


class TestAttributeBranchErrors:
    def test_missing_attribute_triples_raises(self) -> None:
        linker = GCNAlignLinker(use_attributes=True)
        with pytest.raises(LinkingTKError, match="attribute_triples"):
            linker.fit(_KG1, _KG2, _GROUND_TRUTH, graph=_GRAPH, random_state=0)

    def test_top_fraction_rounding_to_zero_attrs_raises(self) -> None:
        # A single distinct predicate at the default top_fraction=0.7
        # rounds down to zero kept columns -- nothing to train with.
        linker = GCNAlignLinker(use_attributes=True)
        attrs1 = [(e.id, "http://example.org/type", "Thing") for e in _KG1]
        with pytest.raises(LinkingTKError, match="no attribute predicate"):
            linker.fit(
                _KG1,
                _KG2,
                _GROUND_TRUTH,
                graph=_GRAPH,
                random_state=0,
                attribute_triples1=attrs1,
                attribute_triples2=[],
            )


class TestErrors:
    def test_link_before_fit_raises(self) -> None:
        linker = GCNAlignLinker()
        with pytest.raises(LinkingTKError, match="before fit"):
            linker.link(_KG1, _KG2, blocking=_AllPairs())

    def test_fit_with_no_ground_truth_ids_in_graph_raises(self) -> None:
        linker = GCNAlignLinker()
        bogus_ground_truth = [("not:an:id", "also:not:an:id")]
        with pytest.raises(LinkingTKError, match="no seed pairs"):
            linker.fit(_KG1, _KG2, bogus_ground_truth, graph=_GRAPH, random_state=0)

    def test_link_with_entity_absent_from_training_graph_raises(self) -> None:
        linker = GCNAlignLinker(embedding_dim=8, num_epochs=5, learning_rate=0.5)
        linker.fit(_KG1, _KG2, _GROUND_TRUTH, graph=_GRAPH, random_state=0)

        unseen = Entity(id="kg1:unseen", labels=["unseen"])
        with pytest.raises(LinkingTKError, match="no trained embedding"):
            linker.link([unseen], _KG2, blocking=_AllPairs())

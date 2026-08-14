import numpy as np
import pytest
import torch

from linkingtk.algorithms.ea import IMUSELinker
from linkingtk.algorithms.ea._imuse_text import (
    align_attributes_by_name,
    align_entities_by_attribute_values,
    levenshtein_ratio,
    local_name,
)
from linkingtk.blocking.base import BlockingStrategy
from linkingtk.core.entity import Entity
from linkingtk.eval import Evaluator
from linkingtk.exceptions import LinkingTKError

# Two isomorphic 4-node chains ("next"-linked), same fixture shape as
# test_iptranse.py's/test_jape.py's -- a pipeline-correctness check, not a
# generalization benchmark. See test_imuse_benchmark.py for held-out
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

# IMUSE has no ground_truth argument at all -- entities are matched purely
# from attribute-value similarity. Matching *values* across KGs give the
# bootstrap unambiguous entity signal; a similar-but-not-identical
# *predicate* local name ("color" vs "colour", ratio ~0.91) exercises the
# name-alignment step genuinely rather than trivially (mirrors AttrE's
# _ATTR1/_ATTR2 fixture reasoning).
_ATTR1 = [
    ("kg1:a", "http://ex.org/color", "red"),
    ("kg1:b", "http://ex.org/color", "blue"),
    ("kg1:c", "http://ex.org/color", "green"),
    ("kg1:d", "http://ex.org/color", "gold"),
]
_ATTR2 = [
    ("kg2:w", "http://ex.org/colour", "red"),
    ("kg2:x", "http://ex.org/colour", "blue"),
    ("kg2:y", "http://ex.org/colour", "green"),
    ("kg2:z", "http://ex.org/colour", "gold"),
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
    def test_bootstrap_recovers_alignment_without_ground_truth(self) -> None:
        linker = IMUSELinker(embedding_dim=16, num_epochs=200, batch_size=32, learning_rate=0.5)
        linker.fit(
            _KG1,
            _KG2,
            graph=_GRAPH,
            attribute_triples1=_ATTR1,
            attribute_triples2=_ATTR2,
            random_state=0,
        )

        results = linker.link(_KG1, _KG2, blocking=_AllPairs())
        predictions = [(r.source_id, r.target_id) for r in results]

        report = Evaluator.evaluate(predictions=predictions, ground_truth=_GROUND_TRUTH)
        assert report.metrics["precision@1"] == 1.0

    def test_source_and_target_embedding_are_identical(self) -> None:
        linker = IMUSELinker(embedding_dim=16, num_epochs=50, batch_size=32, learning_rate=0.5)
        linker.fit(
            _KG1,
            _KG2,
            graph=_GRAPH,
            attribute_triples1=_ATTR1,
            attribute_triples2=_ATTR2,
            random_state=0,
        )

        entity_id = _KG1[0].id
        assert np.array_equal(
            linker.source_embedding(entity_id), linker.target_embedding(entity_id)
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
    def test_bootstrap_recovers_alignment_without_ground_truth_on_cuda(self) -> None:
        linker = IMUSELinker(
            embedding_dim=16, num_epochs=200, batch_size=32, learning_rate=0.5, device="cuda"
        )
        linker.fit(
            _KG1,
            _KG2,
            graph=_GRAPH,
            attribute_triples1=_ATTR1,
            attribute_triples2=_ATTR2,
            random_state=0,
        )

        results = linker.link(_KG1, _KG2, blocking=_AllPairs())
        predictions = [(r.source_id, r.target_id) for r in results]

        report = Evaluator.evaluate(predictions=predictions, ground_truth=_GROUND_TRUTH)
        assert report.metrics["precision@1"] == 1.0

    def test_multi_iteration_bootstrap_runs_without_error(self) -> None:
        linker = IMUSELinker(embedding_dim=8, num_epochs=5, batch_size=32, bootstrap_iterations=2)
        linker.fit(
            _KG1,
            _KG2,
            graph=_GRAPH,
            attribute_triples1=_ATTR1,
            attribute_triples2=_ATTR2,
            random_state=0,
        )

        assert linker._fitted

    def test_early_stopping_runs_without_error(self) -> None:
        linker = IMUSELinker(embedding_dim=16, num_epochs=100, batch_size=32)
        linker.fit(
            _KG1,
            _KG2,
            graph=_GRAPH,
            attribute_triples1=_ATTR1,
            attribute_triples2=_ATTR2,
            random_state=0,
            val_ground_truth=_GROUND_TRUTH,
            patience=1,
            eval_every=5,
        )

        results = linker.link(_KG1, _KG2, blocking=_AllPairs())
        assert {r.source_id for r in results} == {entity.id for entity in _KG1}


class TestErrors:
    def test_link_before_fit_raises(self) -> None:
        linker = IMUSELinker()
        with pytest.raises(LinkingTKError, match="before fit"):
            linker.link(_KG1, _KG2, blocking=_AllPairs())

    def test_fit_with_no_attribute_triples_raises(self) -> None:
        linker = IMUSELinker()
        with pytest.raises(LinkingTKError, match="attribute signal"):
            linker.fit(
                _KG1,
                _KG2,
                graph=_GRAPH,
                attribute_triples1=[],
                attribute_triples2=[],
                random_state=0,
            )

    def test_fit_with_no_bootstrappable_pairs_raises(self) -> None:
        # Predicate local names too dissimilar to ever clear the
        # name-similarity threshold -- align_attributes_by_name finds
        # nothing, so there's no attribute pair to score entities with.
        linker = IMUSELinker()
        with pytest.raises(LinkingTKError, match="no confident aligned entity pairs"):
            linker.fit(
                _KG1,
                _KG2,
                graph=_GRAPH,
                attribute_triples1=[("kg1:a", "http://ex.org/zzzzzzz1", "red")],
                attribute_triples2=[("kg2:w", "http://ex.org/qqqqqqq2", "red")],
                random_state=0,
            )

    def test_link_with_entity_absent_from_training_graph_raises(self) -> None:
        linker = IMUSELinker(embedding_dim=8, num_epochs=5, batch_size=32)
        linker.fit(
            _KG1,
            _KG2,
            graph=_GRAPH,
            attribute_triples1=_ATTR1,
            attribute_triples2=_ATTR2,
            random_state=0,
        )

        unseen = Entity(id="kg1:unseen", labels=["unseen"])
        with pytest.raises(LinkingTKError, match="no trained embedding"):
            linker.link([unseen], _KG2, blocking=_AllPairs())


class TestLevenshteinRatio:
    def test_identical_strings(self) -> None:
        assert levenshtein_ratio("abc", "abc") == 1.0

    def test_empty_strings(self) -> None:
        assert levenshtein_ratio("", "") == 1.0

    def test_hand_computed_insertion(self) -> None:
        # a="ab", b="a": indel distance 1, ratio = (2+1-1)/(2+1) = 2/3.
        assert levenshtein_ratio("ab", "a") == pytest.approx(2 / 3)


class TestLocalName:
    def test_takes_last_path_segment(self) -> None:
        assert local_name("http://ex.org/color") == "color"

    def test_no_slash_returns_whole_string(self) -> None:
        assert local_name("color") == "color"


class TestAlignAttributesByName:
    def test_pairs_similar_predicate_names(self) -> None:
        result = align_attributes_by_name(_ATTR1, _ATTR2, threshold=0.6, top_k=10)

        assert result == {("http://ex.org/color", "http://ex.org/colour")}

    def test_dissimilar_names_below_threshold_are_dropped(self) -> None:
        result = align_attributes_by_name(
            [("kg1:a", "http://ex.org/zzzzzzz1", "red")],
            [("kg2:w", "http://ex.org/qqqqqqq2", "red")],
            threshold=0.6,
            top_k=10,
        )

        assert result == set()

    def test_keeps_only_top_k_by_combined_triple_count(self) -> None:
        triples1 = (
            [("e", "http://ex.org/p0", "v")] * 5
            + [("e", "http://ex.org/p1", "v")] * 3
            + [("e", "http://ex.org/p2", "v")] * 1
        )
        triples2 = (
            [("e", "http://ex.org/p0", "v")] * 5
            + [("e", "http://ex.org/p1", "v")] * 3
            + [("e", "http://ex.org/p2", "v")] * 1
        )

        result = align_attributes_by_name(triples1, triples2, threshold=0.99, top_k=2)

        assert result == {
            ("http://ex.org/p0", "http://ex.org/p0"),
            ("http://ex.org/p1", "http://ex.org/p1"),
        }


class TestAlignEntitiesByAttributeValues:
    _ALIGNED_ATTRS = {("attr:name1", "attr:name2"), ("attr:tag1", "attr:tag2")}

    def test_keeps_only_best_match_not_every_improving_candidate(self) -> None:
        # e2a and e2b both share e1's exact "apple" name value (so both
        # become index-pruned candidates), but only e2b's tag value is a
        # near-exact match to e1's -- guards against regressing to
        # OpenEA's own accept-on-every-improvement bug (see
        # align_entities_by_attribute_values's docstring): only the
        # single best-scoring candidate should end up in the result.
        triples1 = [
            ("e1", "attr:name1", "apple"),
            ("e1", "attr:tag1", "fruit"),
        ]
        triples2 = [
            ("e2a", "attr:name2", "apple"),
            ("e2a", "attr:tag2", "fruits"),
            ("e2b", "attr:name2", "apple"),
            ("e2b", "attr:tag2", "fruit"),
        ]

        result = align_entities_by_attribute_values(
            triples1, triples2, self._ALIGNED_ATTRS, threshold=0.6
        )

        assert result == {("e1", "e2b")}

    def test_entity_with_no_shared_value_is_never_matched(self) -> None:
        triples1 = [
            ("e1", "attr:name1", "apple"),
            ("e_unmatched", "attr:name1", "no-overlap-anywhere"),
        ]
        triples2 = [("e2", "attr:name2", "apple")]

        result = align_entities_by_attribute_values(
            triples1, triples2, {("attr:name1", "attr:name2")}, threshold=0.6
        )

        assert result == {("e1", "e2")}
        assert "e_unmatched" not in {source for source, _ in result}

    def test_empty_aligned_attrs_returns_empty(self) -> None:
        result = align_entities_by_attribute_values(_ATTR1, _ATTR2, set(), threshold=0.6)

        assert result == set()

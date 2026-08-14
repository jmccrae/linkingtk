import numpy as np
import pytest
import torch

from linkingtk.algorithms.ea import MultiKELinker
from linkingtk.algorithms.ea._multike_literal import encode_literals
from linkingtk.algorithms.ea._multike_text import (
    align_predicates_by_name,
    clean_attribute_value,
    filter_frequent_predicates,
    generate_cross_kg_attribute_triples,
    generate_cross_kg_relation_triples,
    substitute_attribute_triples_one_link,
    substitute_relation_triples_one_link,
    weight_attribute_triples,
    zoom_weight,
)
from linkingtk.blocking.base import BlockingStrategy
from linkingtk.core.entity import Entity
from linkingtk.eval import Evaluator
from linkingtk.exceptions import LinkingTKError

# Two isomorphic 4-node chains ("next"-linked), same fixture shape as
# test_attre.py's/test_iptranse.py's -- a pipeline-correctness check, not a
# generalization benchmark. See test_multike_benchmark.py for held-out
# generalization at real-dataset shape. A "next"-linked *chain* (not a
# cycle) is deliberate: a cycle is rotationally symmetric, giving the
# relation view zero signal to distinguish which specific rotation offset
# is correct.
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

# Each entity gets a short attribute value; matching values across KGs give
# the attribute view real cross-lingual signal. Predicate local names are
# similar-but-not-identical ("color"/"colour") so predicate-name alignment
# is genuinely exercised, not trivial.
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

# hf-internal-testing/tiny-random-DistilBertModel: a tiny random-weight HF
# test fixture model (~1MB vs. the real default's ~540MB) published
# specifically for fast unit testing -- same testability pattern as
# test_kdcoe.py's `word_embed_url` injection, adapted to the transformers
# ecosystem's own tiny-test-model convention.
_TINY_MODEL = "hf-internal-testing/tiny-random-DistilBertModel"


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
        # MultiKE's supervision is indirect (seed pairs -> per-view tables
        # -> shared `ent_embeds` via common-space learning), so it needs
        # noticeably more epochs than this family's other, more directly
        # supervised methods to converge even on a 4-node toy -- confirmed
        # empirically (300 epochs recovers exactly, 150 does not yet).
        linker = MultiKELinker(
            embedding_dim=16,
            num_epochs=300,
            batch_size=32,
            entity_batch_size=32,
            attribute_batch_size=32,
            min_predicate_triple_count=1,
            literal_encoder_model=_TINY_MODEL,
        )
        linker.fit(
            _KG1,
            _KG2,
            _GROUND_TRUTH,
            graph=_GRAPH,
            attribute_triples1=_ATTR1,
            attribute_triples2=_ATTR2,
            random_state=0,
        )

        results = linker.link(_KG1, _KG2, blocking=_AllPairs())
        predictions = [(r.source_id, r.target_id) for r in results]

        report = Evaluator.evaluate(predictions=predictions, ground_truth=_GROUND_TRUTH)
        assert report.metrics["precision@1"] == 1.0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
    def test_recovers_seeded_alignment_on_cuda(self) -> None:
        linker = MultiKELinker(
            embedding_dim=16,
            num_epochs=300,
            batch_size=32,
            entity_batch_size=32,
            attribute_batch_size=32,
            min_predicate_triple_count=1,
            literal_encoder_model=_TINY_MODEL,
            device="cuda",
        )
        linker.fit(
            _KG1,
            _KG2,
            _GROUND_TRUTH,
            graph=_GRAPH,
            attribute_triples1=_ATTR1,
            attribute_triples2=_ATTR2,
            random_state=0,
        )

        results = linker.link(_KG1, _KG2, blocking=_AllPairs())
        predictions = [(r.source_id, r.target_id) for r in results]

        report = Evaluator.evaluate(predictions=predictions, ground_truth=_GROUND_TRUTH)
        assert report.metrics["precision@1"] == 1.0

    def test_early_stopping_runs_without_error(self) -> None:
        linker = MultiKELinker(
            embedding_dim=16,
            num_epochs=50,
            batch_size=32,
            entity_batch_size=32,
            attribute_batch_size=32,
            min_predicate_triple_count=1,
            literal_encoder_model=_TINY_MODEL,
        )
        linker.fit(
            _KG1,
            _KG2,
            _GROUND_TRUTH,
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
        linker = MultiKELinker()
        with pytest.raises(LinkingTKError, match="before fit"):
            linker.link(_KG1, _KG2, blocking=_AllPairs())

    def test_fit_with_no_attribute_triples_raises(self) -> None:
        linker = MultiKELinker()
        with pytest.raises(LinkingTKError, match="attribute triples"):
            linker.fit(
                _KG1,
                _KG2,
                _GROUND_TRUTH,
                graph=_GRAPH,
                attribute_triples1=[],
                attribute_triples2=[],
                random_state=0,
            )

    def test_fit_with_no_seed_pairs_in_graph_raises(self) -> None:
        linker = MultiKELinker()
        bogus_ground_truth = [("not:an:id", "also:not:an:id")]
        with pytest.raises(LinkingTKError, match="no seed pairs"):
            linker.fit(
                _KG1,
                _KG2,
                bogus_ground_truth,
                graph=_GRAPH,
                attribute_triples1=_ATTR1,
                attribute_triples2=_ATTR2,
                random_state=0,
            )

    def test_link_with_entity_absent_from_training_graph_raises(self) -> None:
        linker = MultiKELinker(
            embedding_dim=8,
            num_epochs=2,
            batch_size=32,
            min_predicate_triple_count=1,
            literal_encoder_model=_TINY_MODEL,
        )
        linker.fit(
            _KG1,
            _KG2,
            _GROUND_TRUTH,
            graph=_GRAPH,
            attribute_triples1=_ATTR1,
            attribute_triples2=_ATTR2,
            random_state=0,
        )

        unseen = Entity(id="kg1:unseen", labels=["unseen"])
        with pytest.raises(LinkingTKError, match="no trained embedding"):
            linker.link([unseen], _KG2, blocking=_AllPairs())


class TestCleanAttributeValue:
    def test_strips_light_punctuation(self) -> None:
        assert clean_attribute_value("U.S.A., Inc.") == "USA Inc"

    def test_replaces_underscore_hyphen_slash_with_space(self) -> None:
        assert clean_attribute_value("New_York-City/Metro") == "New York City Metro"

    def test_drops_values_still_containing_http(self) -> None:
        assert clean_attribute_value("http://example.org/leftover") == ""


class TestFilterFrequentPredicates:
    def test_keeps_predicate_at_combined_threshold(self) -> None:
        triples1 = [("e1", "p", "v")] * 7
        triples2 = [("e2", "p", "v")] * 3

        kept1, kept2 = filter_frequent_predicates(triples1, triples2, min_count=10)

        assert len(kept1) == 7
        assert len(kept2) == 3

    def test_drops_predicate_below_combined_threshold(self) -> None:
        triples1 = [("e1", "p", "v")] * 5
        triples2 = [("e2", "p", "v")] * 3

        kept1, kept2 = filter_frequent_predicates(triples1, triples2, min_count=10)

        assert kept1 == []
        assert kept2 == []


class TestAlignPredicatesByName:
    def test_pairs_mutually_best_matching_names(self) -> None:
        aligned = align_predicates_by_name(
            {"http://ex.org/color"}, {"http://ex.org/colour"}, threshold=0.5
        )

        assert set(aligned) == {("http://ex.org/color", "http://ex.org/colour")}

    def test_drops_pair_that_is_not_mutually_best(self) -> None:
        # predicates2 has only one option ("color"), so it's the *only*
        # (hence best) match for both "color" and "colors" in predicates1.
        # But predicates2's own best match, scanning back over predicates1,
        # is the exact "color" -- not "colors". The ("colors", "color")
        # pair must be dropped despite "colors" having no better option of
        # its own, since the match isn't mutual.
        predicates1 = {"http://ex.org/color", "http://ex.org/colors"}
        predicates2 = {"http://ex.org/color"}

        aligned = align_predicates_by_name(predicates1, predicates2, threshold=0.5)

        assert ("http://ex.org/colors", "http://ex.org/color") not in aligned
        assert ("http://ex.org/color", "http://ex.org/color") in aligned

    def test_below_threshold_pair_is_dropped(self) -> None:
        aligned = align_predicates_by_name(
            {"http://ex.org/color"}, {"http://ex.org/unrelated"}, threshold=0.9
        )

        assert aligned == {}


class TestZoomWeight:
    def test_hand_computed_value(self) -> None:
        assert zoom_weight(0.9, min_w_before=0.8, min_w_after=0.5) == pytest.approx(0.75)

    def test_weight_at_floor_maps_to_min_w_after(self) -> None:
        assert zoom_weight(0.8, min_w_before=0.8, min_w_after=0.5) == pytest.approx(0.5)


class TestWeightAttributeTriples:
    def test_aligned_predicate_gets_rescaled_weight(self) -> None:
        weighted = weight_attribute_triples(
            [("e", "p1", "v")],
            {("p1", "p2"): 0.9},
            predicate_soft_sim=0.8,
            is_kg1=True,
        )

        assert weighted == [("e", "p1", "v", pytest.approx(0.75))]

    def test_unaligned_predicate_gets_flat_weight(self) -> None:
        weighted = weight_attribute_triples(
            [("e", "p_other", "v")],
            {("p1", "p2"): 0.9},
            predicate_soft_sim=0.8,
            is_kg1=True,
        )

        assert weighted == [("e", "p_other", "v", 0.2)]

    def test_is_kg1_selects_correct_side_of_aligned_pairs(self) -> None:
        # KG2's own triples use `p2` as the predicate label -- `is_kg1=False`
        # must match against the pair's *second* element, not the first.
        weighted = weight_attribute_triples(
            [("e", "p2", "v")],
            {("p1", "p2"): 0.9},
            predicate_soft_sim=0.8,
            is_kg1=False,
        )

        assert weighted == [("e", "p2", "v", pytest.approx(0.75))]


class TestSubstituteRelationTriplesOneLink:
    def test_relabels_head_and_tail_occurrences(self) -> None:
        by_head = {"a": {("r", "b")}}
        by_tail = {"a": {("c", "r")}}

        substituted = substitute_relation_triples_one_link("a", "x", by_head, by_tail)

        assert substituted == {("x", "r", "b"), ("c", "r", "x")}

    def test_entity_with_no_triples_produces_nothing(self) -> None:
        assert substitute_relation_triples_one_link("a", "x", {}, {}) == set()


class TestSubstituteAttributeTriplesOneLink:
    def test_relabels_entity_occurrences(self) -> None:
        av_dict = {"a": {("p", "v1"), ("p", "v2")}}

        substituted = substitute_attribute_triples_one_link("a", "x", av_dict)

        assert substituted == {("x", "p", "v1"), ("x", "p", "v2")}


class TestGenerateCrossKgTriples:
    def test_relation_triples_both_directions(self) -> None:
        triples1 = [("a", "r", "b"), ("c", "r", "a")]
        triples2 = [("x", "r", "y")]

        new1, new2 = generate_cross_kg_relation_triples([("a", "x")], triples1, triples2)

        assert set(new1) == {("c", "r", "x"), ("x", "r", "b")}
        assert set(new2) == {("a", "r", "y")}

    def test_attribute_triples_both_directions(self) -> None:
        attr1 = [("a", "p", "v")]
        attr2: list[tuple[str, str, str]] = []

        new1, new2 = generate_cross_kg_attribute_triples([("a", "x")], attr1, attr2)

        assert new1 == [("x", "p", "v")]
        assert new2 == []


class TestEncodeLiterals:
    def test_output_shape_and_normalization(self) -> None:
        vectors = encode_literals(
            ["Paris", "Berlin", "United States"], _TINY_MODEL, embedding_dim=16, random_state=0
        )

        assert vectors.shape == (3, 16)
        norms = np.linalg.norm(vectors, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_deterministic_given_same_random_state(self) -> None:
        texts = ["Paris", "Berlin"]
        first = encode_literals(texts, _TINY_MODEL, embedding_dim=16, random_state=0)
        second = encode_literals(texts, _TINY_MODEL, embedding_dim=16, random_state=0)

        assert np.allclose(first, second)

    def test_empty_input_returns_empty_array(self) -> None:
        vectors = encode_literals([], _TINY_MODEL, embedding_dim=16)

        assert vectors.shape == (0, 16)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
    def test_output_shape_and_normalization_on_cuda(self) -> None:
        vectors = encode_literals(
            ["Paris", "Berlin", "United States"],
            _TINY_MODEL,
            embedding_dim=16,
            random_state=0,
            device="cuda",
        )

        assert vectors.shape == (3, 16)
        norms = np.linalg.norm(vectors, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

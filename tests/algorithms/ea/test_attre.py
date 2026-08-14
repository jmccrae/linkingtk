import numpy as np
import pytest
import torch

from linkingtk.algorithms.ea import AttrELinker
from linkingtk.algorithms.ea._attre_text import (
    build_value_char_ids,
    clean_attribute_value,
    sample_negative_attribute_triples,
    select_char_vocabulary,
)
from linkingtk.algorithms.ea._attre_torch import compose_value_embeddings
from linkingtk.blocking.base import BlockingStrategy
from linkingtk.core.entity import Entity
from linkingtk.eval import Evaluator
from linkingtk.exceptions import LinkingTKError

# Two isomorphic 4-node chains ("next"-linked), same fixture shape as
# test_iptranse.py's/test_jape.py's -- a pipeline-correctness check, not a
# generalization benchmark. See test_attre_benchmark.py for held-out
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

# Each entity gets a short attribute value; matching values across KGs give
# the character/attribute half real cross-lingual signal.
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
    def test_recovers_seeded_alignment(self) -> None:
        linker = AttrELinker(embedding_dim=16, num_epochs=100, batch_size=32, literal_len=5)
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
        linker = AttrELinker(
            embedding_dim=16, num_epochs=100, batch_size=32, literal_len=5, device="cuda"
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

    def test_partial_seed_generalizes_to_unseeded_pairs(self) -> None:
        # Seed only 2 of 4 pairs -- shared-id structural + attribute/character
        # training together should still recover the rest (mirrors
        # IPTransE's/JAPE's equivalent tests).
        partial_ground_truth = [("kg1:a", "kg2:w"), ("kg1:c", "kg2:y")]
        linker = AttrELinker(
            embedding_dim=16,
            num_epochs=300,
            batch_size=32,
            learning_rate=0.5,
            literal_len=5,
        )
        linker.fit(
            _KG1,
            _KG2,
            partial_ground_truth,
            graph=_GRAPH,
            attribute_triples1=_ATTR1,
            attribute_triples2=_ATTR2,
            random_state=0,
        )

        results = linker.link(_KG1, _KG2, blocking=_AllPairs())
        predictions = [(r.source_id, r.target_id) for r in results]

        report = Evaluator.evaluate(predictions=predictions, ground_truth=_GROUND_TRUTH)
        assert report.metrics["precision@1"] == 1.0

    def test_survives_more_distinct_attribute_values_than_entities(self) -> None:
        # Regression test: entity_pool for the character/attribute (CE) side
        # must be built from the entity column alone. build_kg_context's own
        # entity_pool is `union(column 0, column 2)`, correct for relation
        # triples where both endpoints are entities but wrong for attribute
        # triples (column 2 is a *value* id, a completely different id
        # space). With few entities and many more distinct values than
        # entities, an unfixed entity_pool would include out-of-range value
        # ids and crash entity-embedding lookups during negative sampling --
        # this only surfaced at real-dataset scale (30000 entities, 50000+
        # values) despite passing on every other, smaller toy fixture here.
        many_values_attr1 = [
            (entity.id, "http://ex.org/tag", f"kg1-{entity.id}-tag{i}")
            for entity in _KG1
            for i in range(3)
        ]
        many_values_attr2 = [
            (entity.id, "http://ex.org/tag", f"kg2-{entity.id}-tag{i}")
            for entity in _KG2
            for i in range(3)
        ]
        linker = AttrELinker(embedding_dim=8, num_epochs=5, batch_size=32, literal_len=5)

        linker.fit(
            _KG1,
            _KG2,
            _GROUND_TRUTH,
            graph=_GRAPH,
            attribute_triples1=many_values_attr1,
            attribute_triples2=many_values_attr2,
            random_state=0,
        )

        assert linker._fitted

    def test_early_stopping_runs_without_error(self) -> None:
        linker = AttrELinker(embedding_dim=16, num_epochs=100, batch_size=32)
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
        linker = AttrELinker()
        with pytest.raises(LinkingTKError, match="before fit"):
            linker.link(_KG1, _KG2, blocking=_AllPairs())

    def test_fit_with_no_ground_truth_ids_in_graph_raises(self) -> None:
        linker = AttrELinker()
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

    def test_fit_with_no_attribute_triples_raises(self) -> None:
        linker = AttrELinker()
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

    def test_link_with_entity_absent_from_training_graph_raises(self) -> None:
        linker = AttrELinker(embedding_dim=8, num_epochs=5, batch_size=32)
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
    def test_strips_parenthetical_content(self) -> None:
        assert clean_attribute_value("Paris (city)") == "Paris"

    def test_strips_light_punctuation(self) -> None:
        assert clean_attribute_value("U.S.A., Inc.") == "USA Inc"

    def test_replaces_underscore_and_hyphen_with_space(self) -> None:
        assert clean_attribute_value("New_York-City") == "New York City"


class TestSelectCharVocabulary:
    def test_keeps_frequent_characters(self) -> None:
        vocab = select_char_vocabulary(["aaa", "aab"], frequency_threshold=0.1)

        assert set(vocab) == {"a", "b"}

    def test_drops_rare_characters_below_threshold(self) -> None:
        # "z" occurs once out of many characters -- well below a high threshold.
        values = ["aaaaaaaaaa", "z"]

        vocab = select_char_vocabulary(values, frequency_threshold=0.5)

        assert "a" in vocab
        assert "z" not in vocab

    def test_ids_start_at_one(self) -> None:
        vocab = select_char_vocabulary(["ab"], frequency_threshold=0.0)

        assert set(vocab.values()) == {1, 2}

    def test_empty_values_returns_empty_vocabulary(self) -> None:
        assert select_char_vocabulary([], frequency_threshold=0.0001) == {}


class TestBuildValueCharIds:
    def test_pads_short_values(self) -> None:
        char_to_id = {"a": 1, "b": 2}

        result = build_value_char_ids(["ab"], char_to_id, literal_len=4)

        assert np.array_equal(result["ab"], np.array([1, 2, 0, 0]))

    def test_truncates_long_values(self) -> None:
        char_to_id = {"a": 1, "b": 2, "c": 3, "d": 4}

        result = build_value_char_ids(["abcd"], char_to_id, literal_len=2)

        assert np.array_equal(result["abcd"], np.array([1, 2]))

    def test_out_of_vocabulary_character_maps_to_zero(self) -> None:
        result = build_value_char_ids(["z"], {}, literal_len=1)

        assert np.array_equal(result["z"], np.array([0]))


class TestSampleNegativeAttributeTriples:
    def test_only_corrupts_entity_column(self) -> None:
        positives = np.array([[0, 5, 9]], dtype=np.int64)
        entity_pool = np.array([1, 2, 3], dtype=np.int64)
        rng = np.random.default_rng(0)

        negatives = sample_negative_attribute_triples(positives, entity_pool, set(), rng)

        assert negatives[0, 1] == 5
        assert negatives[0, 2] == 9
        assert negatives[0, 0] in entity_pool

    def test_avoids_reproducing_a_real_triple_when_possible(self) -> None:
        positives = np.array([[0, 5, 9]], dtype=np.int64)
        entity_pool = np.array([0, 1], dtype=np.int64)
        real_triples = {(0, 5, 9)}
        rng = np.random.default_rng(0)

        negatives = sample_negative_attribute_triples(positives, entity_pool, real_triples, rng)

        assert tuple(negatives[0]) != (0, 5, 9)


class TestComposeValueEmbeddings:
    def test_matches_hand_computed_prefix_mean_sum(self) -> None:
        # 3 characters, dim 2: c0=[1,0], c1=[0,1], c2=[2,2].
        # prefix means: mean(c0)=[1,0]; mean(c0,c1)=[.5,.5]; mean(c0,c1,c2)=[1,1].
        # sum of prefix means = [2.5, 1.5].
        chars = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]]])

        result = compose_value_embeddings(chars)

        assert torch.allclose(result, torch.tensor([[2.5, 1.5]]))

    def test_batches_independently(self) -> None:
        chars = torch.tensor(
            [
                [[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]],
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            ]
        )

        result = compose_value_embeddings(chars)

        assert torch.allclose(result[0], torch.tensor([2.5, 1.5]))
        assert torch.allclose(result[1], torch.tensor([0.0, 0.0]))

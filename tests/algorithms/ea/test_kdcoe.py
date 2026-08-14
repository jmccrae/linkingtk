import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch

from linkingtk.algorithms.ea import KDCoELinker
from linkingtk.algorithms.ea._kdcoe_text import (
    build_word_ids,
    extract_description_text,
    load_fasttext_vectors,
    tokenize_description,
)
from linkingtk.blocking.base import BlockingStrategy
from linkingtk.core.entity import Entity
from linkingtk.eval import Evaluator
from linkingtk.exceptions import LinkingTKError

# Two isomorphic 4-node chains ("next"-linked), same fixture shape as
# test_mtranse.py's/test_iptranse.py's/test_jape.py's -- a pipeline-correctness
# check, not a generalization benchmark. See test_kdcoe_benchmark.py for
# held-out generalization at real-dataset shape.
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

# Every entity gets a real description triple with matching text across KGs
# (different predicate spellings per KG, exercising the "escription" substring
# match across dc:/dbo:-style predicates) -- this is what gives description-
# based bootstrapping real signal for the unseeded pairs in
# test_partial_seed_bootstraps_to_unseeded_pairs below (their single-letter
# *labels* carry no such signal; label-fallback text is covered separately by
# TestExtractDescriptionText and test_works_without_attribute_triples).
_ATTR1 = [
    ("kg1:a", "http://purl.org/dc/elements/1.1/description", "red small thing"),
    ("kg1:b", "http://purl.org/dc/elements/1.1/description", "blue medium item"),
    ("kg1:c", "http://dbpedia.org/ontology/description", "green big object"),
    ("kg1:d", "http://dbpedia.org/ontology/description", "yellow huge shape"),
]
_ATTR2 = [
    ("kg2:w", "http://purl.org/dc/terms/description", "red small thing"),
    ("kg2:x", "http://purl.org/dc/terms/description", "blue medium item"),
    ("kg2:y", "http://dbpedia.org/ontology/depictionDescription", "green big object"),
    ("kg2:z", "http://dbpedia.org/ontology/depictionDescription", "yellow huge shape"),
]

_WV_DIM = 8
_VOCAB = [
    "a",
    "b",
    "c",
    "d",
    "w",
    "x",
    "y",
    "z",
    "red",
    "small",
    "thing",
    "blue",
    "medium",
    "item",
    "green",
    "big",
    "object",
    "yellow",
    "huge",
    "shape",
]


def _fasttext_zip_url(tmp_path: Path) -> str:
    """A tiny, deterministic fastText-format ``.vec.zip`` covering ``_VOCAB``."""
    rng = np.random.default_rng(0)
    vec_path = tmp_path / "vectors.vec"
    with vec_path.open("w") as f:
        f.write(f"{len(_VOCAB)} {_WV_DIM}\n")
        for word in _VOCAB:
            values = " ".join(f"{v:.4f}" for v in rng.uniform(-1, 1, size=_WV_DIM))
            f.write(f"{word} {values}\n")
    zip_path = tmp_path / "vectors.vec.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.write(vec_path, arcname="vectors.vec")
    return f"file://{zip_path}"


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
    def test_recovers_seeded_alignment(self, tmp_path: Path) -> None:
        linker = KDCoELinker(
            embedding_dim=16,
            num_epochs=50,
            batch_size=32,
            wv_dim=_WV_DIM,
            desc_batch_size=4,
            word_embed_url=_fasttext_zip_url(tmp_path),
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

    def test_source_embedding_differs_from_target_embedding(self, tmp_path: Path) -> None:
        # KDCoE projects the source side through the learned mapping
        # matrix but scores the target side raw, same as MTransE --
        # source_embedding/target_embedding must preserve that asymmetry.
        linker = KDCoELinker(
            embedding_dim=16,
            num_epochs=20,
            batch_size=32,
            wv_dim=_WV_DIM,
            desc_batch_size=4,
            word_embed_url=_fasttext_zip_url(tmp_path),
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

        entity_id = _KG1[0].id
        assert not np.allclose(
            linker.source_embedding(entity_id), linker.target_embedding(entity_id)
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
    def test_recovers_seeded_alignment_on_cuda(self, tmp_path: Path) -> None:
        linker = KDCoELinker(
            embedding_dim=16,
            num_epochs=50,
            batch_size=32,
            wv_dim=_WV_DIM,
            desc_batch_size=4,
            word_embed_url=_fasttext_zip_url(tmp_path),
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

    def test_works_without_attribute_triples(self, tmp_path: Path) -> None:
        # attribute_triples1/2 omitted -- description text falls back to labels
        # for every entity, but the pathway must still run end-to-end.
        linker = KDCoELinker(
            embedding_dim=16,
            num_epochs=50,
            batch_size=32,
            wv_dim=_WV_DIM,
            desc_batch_size=4,
            word_embed_url=_fasttext_zip_url(tmp_path),
        )
        linker.fit(_KG1, _KG2, _GROUND_TRUTH, graph=_GRAPH, random_state=0)

        results = linker.link(_KG1, _KG2, blocking=_AllPairs())
        predictions = [(r.source_id, r.target_id) for r in results]

        report = Evaluator.evaluate(predictions=predictions, ground_truth=_GROUND_TRUTH)
        assert report.metrics["precision@1"] == 1.0

    def test_partial_seed_bootstraps_to_unseeded_pairs(self, tmp_path: Path) -> None:
        # Seed only 2 of 4 pairs -- structural + description co-training
        # together should still recover the rest (mirrors IPTransE's/JAPE's
        # equivalent tests).
        partial_ground_truth = [("kg1:a", "kg2:w"), ("kg1:c", "kg2:y")]
        linker = KDCoELinker(
            embedding_dim=16,
            num_epochs=100,
            batch_size=32,
            learning_rate=0.1,
            wv_dim=_WV_DIM,
            desc_batch_size=4,
            max_co_training_iters=3,
            word_embed_url=_fasttext_zip_url(tmp_path),
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

    def test_early_stopping_runs_without_error(self, tmp_path: Path) -> None:
        linker = KDCoELinker(
            embedding_dim=16,
            num_epochs=50,
            batch_size=32,
            wv_dim=_WV_DIM,
            desc_batch_size=4,
            word_embed_url=_fasttext_zip_url(tmp_path),
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
        linker = KDCoELinker()
        with pytest.raises(LinkingTKError, match="before fit"):
            linker.link(_KG1, _KG2, blocking=_AllPairs())

    def test_fit_with_no_ground_truth_ids_in_graph_raises(self, tmp_path: Path) -> None:
        linker = KDCoELinker(word_embed_url=_fasttext_zip_url(tmp_path))
        bogus_ground_truth = [("not:an:id", "also:not:an:id")]
        with pytest.raises(LinkingTKError, match="no seed pairs"):
            linker.fit(_KG1, _KG2, bogus_ground_truth, graph=_GRAPH, random_state=0)

    def test_link_with_entity_absent_from_training_graph_raises(self, tmp_path: Path) -> None:
        linker = KDCoELinker(
            embedding_dim=8,
            num_epochs=5,
            batch_size=32,
            wv_dim=_WV_DIM,
            desc_batch_size=4,
            word_embed_url=_fasttext_zip_url(tmp_path),
        )
        linker.fit(_KG1, _KG2, _GROUND_TRUTH, graph=_GRAPH, random_state=0)

        unseen = Entity(id="kg1:unseen", labels=["unseen"])
        with pytest.raises(LinkingTKError, match="no trained embedding"):
            linker.link([unseen], _KG2, blocking=_AllPairs())


class TestExtractDescriptionText:
    def test_uses_matching_description_predicate(self) -> None:
        entities = [Entity(id="e1", labels=["fallback"])]
        triples = [("e1", "http://dbpedia.org/ontology/description", "real text")]

        result = extract_description_text(entities, triples)

        assert result == {"e1": "real text"}

    def test_falls_back_to_label_when_no_description_triple(self) -> None:
        entities = [Entity(id="e1", labels=["Fallback Label"])]

        result = extract_description_text(entities, [])

        assert result == {"e1": "Fallback Label"}

    def test_ignores_non_description_predicates(self) -> None:
        entities = [Entity(id="e1", labels=["fallback"])]
        triples = [("e1", "http://dbpedia.org/ontology/birthDate", "2000-01-01")]

        result = extract_description_text(entities, triples)

        assert result == {"e1": "fallback"}


class TestTokenizeDescription:
    def test_strips_punctuation_and_lowercases(self) -> None:
        assert tokenize_description("Red, Small! Thing.") == ["red", "small", "thing"]

    def test_empty_text(self) -> None:
        assert tokenize_description("") == []


class TestBuildWordIds:
    def test_pads_short_sequences(self) -> None:
        result = build_word_ids(
            {"e1": ["a", "b"]}, {"a": 0, "b": 1}, default_desc_length=4, oov_id=2
        )

        assert np.array_equal(result["e1"], np.array([0, 1, 2, 2]))

    def test_truncates_long_sequences(self) -> None:
        result = build_word_ids(
            {"e1": ["a", "b", "c", "d", "e"]},
            {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4},
            default_desc_length=3,
            oov_id=5,
        )

        assert np.array_equal(result["e1"], np.array([0, 1, 2]))

    def test_out_of_vocabulary_word_maps_to_oov_id(self) -> None:
        result = build_word_ids({"e1": ["unknown"]}, {}, default_desc_length=2, oov_id=99)

        assert np.array_equal(result["e1"], np.array([99, 99]))


class TestLoadFasttextVectors:
    def test_filters_to_requested_vocabulary(self, tmp_path: Path) -> None:
        url = _fasttext_zip_url(tmp_path)

        vectors = load_fasttext_vectors(url, {"a", "red"})

        assert set(vectors) == {"a", "red"}
        assert vectors["a"].shape == (_WV_DIM,)

    def test_word_outside_file_is_simply_absent(self, tmp_path: Path) -> None:
        url = _fasttext_zip_url(tmp_path)

        vectors = load_fasttext_vectors(url, {"nonexistent-word"})

        assert vectors == {}

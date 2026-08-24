import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch

from linkingtk.algorithms.ea import RDGCNLinker
from linkingtk.blocking.base import BlockingStrategy
from linkingtk.core.entity import Entity
from linkingtk.eval import Evaluator
from linkingtk.exceptions import LinkingTKError

# Two isomorphic 4-node chains ("next"-linked), same fixture shape as
# test_gcn_align.py's/test_mtranse.py's -- a pipeline-correctness check, not
# a generalization benchmark. See test_rdgcn_benchmark.py for held-out
# generalization. Unlike those, RDGCN needs entities with real labels (its
# name-embedding init) -- single letters here, matching test_kdcoe.py's
# convention.
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

_WV_DIM = 8
_VOCAB = ["a", "b", "c", "d", "w", "x", "y", "z"]


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
        linker = RDGCNLinker(
            embedding_dim=_WV_DIM,
            num_epochs=200,
            learning_rate=0.05,
            neg_triple_num=3,
            word_embed_url=_fasttext_zip_url(tmp_path),
        )
        linker.fit(_KG1, _KG2, _GROUND_TRUTH, graph=_GRAPH, random_state=0)

        results = linker.link(_KG1, _KG2, blocking=_AllPairs())
        predictions = [(r.source_id, r.target_id) for r in results]

        report = Evaluator.evaluate(predictions=predictions, ground_truth=_GROUND_TRUTH)
        assert report.metrics["precision@1"] == 1.0

    def test_early_stopping_runs_without_error(self, tmp_path: Path) -> None:
        linker = RDGCNLinker(
            embedding_dim=_WV_DIM,
            num_epochs=100,
            learning_rate=0.05,
            neg_triple_num=3,
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

    def test_source_embedding_equals_target_embedding(self, tmp_path: Path) -> None:
        # RDGCN has no learned mapping matrix -- both sides share the same
        # trained embedding table, like GCN-Align.
        linker = RDGCNLinker(
            embedding_dim=_WV_DIM,
            num_epochs=20,
            learning_rate=0.05,
            neg_triple_num=3,
            word_embed_url=_fasttext_zip_url(tmp_path),
        )
        linker.fit(_KG1, _KG2, _GROUND_TRUTH, graph=_GRAPH, random_state=0)

        entity_id = _KG1[0].id
        assert np.array_equal(
            linker.source_embedding(entity_id), linker.target_embedding(entity_id)
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
    def test_recovers_seeded_alignment_on_cuda(self, tmp_path: Path) -> None:
        linker = RDGCNLinker(
            embedding_dim=_WV_DIM,
            num_epochs=200,
            learning_rate=0.05,
            neg_triple_num=3,
            word_embed_url=_fasttext_zip_url(tmp_path),
            device="cuda",
        )
        linker.fit(_KG1, _KG2, _GROUND_TRUTH, graph=_GRAPH, random_state=0)

        results = linker.link(_KG1, _KG2, blocking=_AllPairs())
        predictions = [(r.source_id, r.target_id) for r in results]

        report = Evaluator.evaluate(predictions=predictions, ground_truth=_GROUND_TRUTH)
        assert report.metrics["precision@1"] == 1.0


class TestErrors:
    def test_link_before_fit_raises(self) -> None:
        linker = RDGCNLinker()
        with pytest.raises(LinkingTKError, match="before fit"):
            linker.link(_KG1, _KG2, blocking=_AllPairs())

    def test_fit_with_no_ground_truth_ids_in_graph_raises(self, tmp_path: Path) -> None:
        linker = RDGCNLinker(embedding_dim=_WV_DIM, word_embed_url=_fasttext_zip_url(tmp_path))
        bogus_ground_truth = [("not:an:id", "also:not:an:id")]
        with pytest.raises(LinkingTKError, match="no seed pairs"):
            linker.fit(_KG1, _KG2, bogus_ground_truth, graph=_GRAPH, random_state=0)

    def test_fit_with_no_labels_raises(self, tmp_path: Path) -> None:
        linker = RDGCNLinker(embedding_dim=_WV_DIM, word_embed_url=_fasttext_zip_url(tmp_path))
        unlabeled1 = [Entity(id=e.id, labels=[]) for e in _KG1]
        unlabeled2 = [Entity(id=e.id, labels=[]) for e in _KG2]
        with pytest.raises(LinkingTKError, match="non-empty `labels`"):
            linker.fit(unlabeled1, unlabeled2, _GROUND_TRUTH, graph=_GRAPH, random_state=0)

    def test_link_with_entity_absent_from_training_graph_raises(self, tmp_path: Path) -> None:
        linker = RDGCNLinker(
            embedding_dim=_WV_DIM,
            num_epochs=5,
            neg_triple_num=3,
            word_embed_url=_fasttext_zip_url(tmp_path),
        )
        linker.fit(_KG1, _KG2, _GROUND_TRUTH, graph=_GRAPH, random_state=0)

        unseen = Entity(id="kg1:unseen", labels=["unseen"])
        with pytest.raises(LinkingTKError, match="no trained embedding"):
            linker.link([unseen], _KG2, blocking=_AllPairs())

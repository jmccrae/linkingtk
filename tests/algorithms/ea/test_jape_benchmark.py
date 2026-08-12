"""Benchmark: JAPELinker trained on an OpenEA-native-format train/test
split, evaluated with linkingtk.eval.Evaluator.evaluate_ranked (Hits@1,
Hits@10, MRR) -- part of the Knowledge Graph Embeddings milestone's
per-method acceptance criteria (#28).

Unlike test_mtranse_benchmark.py/test_iptranse_benchmark.py, this can't
reuse tests/fixtures/kg_zip_kge_benchmark_toy/ (that fixture has no
attribute triples, being in the numeric-id _KGZipDataset format) -- see
tests/fixtures/openea_native_kge_benchmark_toy/ instead, in the
_OpenEANativeDataset format JAPE's attribute pipeline needs. See
examples/jape_benchmark.py for the same methodology at real dataset scale.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import ClassVar

from linkingtk.algorithms.ea import JAPELinker
from linkingtk.blocking.base import BlockingStrategy
from linkingtk.core.entity import Entity
from linkingtk.datasets.openea_native import _OpenEANativeDataset
from linkingtk.eval import Evaluator
from linkingtk.utils.graph import to_triples

_FIXTURES_DIR = Path(__file__).parent.parent.parent / "fixtures" / "openea_native_kge_benchmark_toy"


class _BenchmarkDataset(_OpenEANativeDataset):
    """Test-only concrete subclass pointed at the (shared) benchmark fixture."""

    _dataset_name: ClassVar[str] = "jape_benchmark_toy"


def _zip_url(tmp_path: Path) -> str:
    zip_path = tmp_path / "data.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        for file in _FIXTURES_DIR.iterdir():
            archive.write(file, arcname=file.name)
    return f"file://{zip_path}"


class _AllPairs(BlockingStrategy):
    """Lets every pair through -- see test_jape.py's identical helper."""

    def candidate_pairs(
        self, dataset1: list[Entity], dataset2: list[Entity]
    ) -> list[tuple[Entity, Entity]]:
        return [(e1, e2) for e1 in dataset1 for e2 in dataset2]


def test_jape_linker_reports_ranked_metrics_on_held_out_split(tmp_path: Path) -> None:
    dataset = _BenchmarkDataset(zip_url=_zip_url(tmp_path))
    entities1, entities2, _ = dataset.load()
    train_pairs, test_pairs, _ = dataset.load_splits()
    graph1, graph2 = dataset.load_graphs()
    attrs1, attrs2 = dataset.load_attribute_triples()
    graph = to_triples(graph1) + to_triples(graph2)

    linker = JAPELinker(
        embedding_dim=16,
        num_epochs=100,
        batch_size=64,
        learning_rate=0.05,
        attr_max_epoch=20,
        sub_mat_size=2,
    )
    linker.fit(
        entities1,
        entities2,
        ground_truth=train_pairs,
        graph=graph,
        random_state=0,
        attribute_triples1=attrs1,
        attribute_triples2=attrs2,
    )
    results = linker.link(entities1, entities2, blocking=_AllPairs())

    ranked_predictions = [(r.source_id, [r.target_id, *r.alternatives]) for r in results]
    report = Evaluator.evaluate_ranked(ranked_predictions, ground_truth=test_pairs, top_k=[1, 10])

    assert set(report.metrics) == {"Hits@1", "Hits@10", "MRR"}
    for value in report.metrics.values():
        assert 0.0 <= value <= 1.0
    assert report.metrics["Hits@10"] >= report.metrics["Hits@1"]
    assert report.metrics["MRR"] >= report.metrics["Hits@1"]

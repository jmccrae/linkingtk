"""Benchmark: MultiKELinker trained on an OpenEA-native-format train/test
split, evaluated with linkingtk.eval.Evaluator.evaluate_ranked (Hits@1,
Hits@10, MRR) -- part of the Knowledge Graph Embeddings milestone's
per-method acceptance criteria (#35).

Reuses tests/fixtures/openea_native_kge_benchmark_toy/ (built for JAPE's
benchmark test, #28, also reused by KDCoE/AttrE/IMUSE) -- structural-
sanity-only assertions, same rationale as the rest of the family's
benchmark tests. Ranking is exhaustive (via linkingtk.eval.rank_exhaustive,
no blocking), matching OpenEA's own evaluation methodology. See
examples/multike_benchmark.py for the same methodology at real dataset
scale.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import ClassVar

from linkingtk.algorithms.ea import MultiKELinker
from linkingtk.datasets.openea_native import _OpenEANativeDataset
from linkingtk.eval import Evaluator, rank_exhaustive
from linkingtk.utils.graph import to_triples

_FIXTURES_DIR = Path(__file__).parent.parent.parent / "fixtures" / "openea_native_kge_benchmark_toy"
_TINY_MODEL = "hf-internal-testing/tiny-random-DistilBertModel"


class _BenchmarkDataset(_OpenEANativeDataset):
    """Test-only concrete subclass pointed at the (shared) benchmark fixture."""

    _dataset_name: ClassVar[str] = "multike_benchmark_toy"


def _zip_url(tmp_path: Path) -> str:
    zip_path = tmp_path / "data.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        for file in _FIXTURES_DIR.iterdir():
            archive.write(file, arcname=file.name)
    return f"file://{zip_path}"


def test_multike_linker_reports_ranked_metrics_on_held_out_split(tmp_path: Path) -> None:
    dataset = _BenchmarkDataset(zip_url=_zip_url(tmp_path))
    entities1, entities2, _ = dataset.load()
    train_pairs, test_pairs, _ = dataset.load_splits()
    graph1, graph2 = dataset.load_graphs()
    attrs1, attrs2 = dataset.load_attribute_triples()
    graph = to_triples(graph1) + to_triples(graph2)

    linker = MultiKELinker(
        embedding_dim=16,
        num_epochs=50,
        batch_size=64,
        entity_batch_size=64,
        attribute_batch_size=64,
        learning_rate=0.1,
        common_space_learning_rate=0.1,
        min_predicate_triple_count=1,
        literal_encoder_model=_TINY_MODEL,
    )
    linker.fit(
        entities1,
        entities2,
        ground_truth=train_pairs,
        graph=graph,
        attribute_triples1=attrs1,
        attribute_triples2=attrs2,
        random_state=0,
    )
    test_source_ids = {s for s, _ in test_pairs}
    test_target_ids = {t for _, t in test_pairs}
    ranked_predictions = rank_exhaustive(
        linker,
        [e for e in entities1 if e.id in test_source_ids],
        [e for e in entities2 if e.id in test_target_ids],
    )
    report = Evaluator.evaluate_ranked(ranked_predictions, ground_truth=test_pairs, top_k=[1, 10])

    assert set(report.metrics) == {"Hits@1", "Hits@10", "MRR"}
    for value in report.metrics.values():
        assert 0.0 <= value <= 1.0
    assert report.metrics["Hits@10"] >= report.metrics["Hits@1"]
    assert report.metrics["MRR"] >= report.metrics["Hits@1"]

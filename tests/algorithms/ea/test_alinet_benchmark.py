"""Benchmark: AliNetLinker trained on a DBP15K/OpenEA-style native
train/test split, evaluated with linkingtk.eval.Evaluator.evaluate_ranked
(Hits@1, Hits@10, MRR) -- part of #44's acceptance criteria.

Structural-sanity-only assertions, same rationale as
test_gcn_align_benchmark.py's/test_rdgcn_benchmark.py's: unlike GCN-Align
and RDGCN, AliNet has no published EN-FR-15K-V1 number to compare against
at all (see the module docstring on linkingtk.algorithms.ea.alinet), so
this is the most literal case of "no numeric target" among the three GNN
linkers. Ranking is exhaustive (via linkingtk.eval.rank_exhaustive, no
blocking), matching OpenEA's own evaluation methodology. See
examples/alinet_benchmark.py for the same methodology at real dataset
scale.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import ClassVar

from linkingtk.algorithms.ea import AliNetLinker
from linkingtk.datasets.kg_zip import _KGZipDataset
from linkingtk.eval import Evaluator, rank_exhaustive
from linkingtk.utils.graph import to_triples

_FIXTURES_DIR = Path(__file__).parent.parent.parent / "fixtures" / "kg_zip_kge_benchmark_toy"


class _BenchmarkDataset(_KGZipDataset):
    """Test-only concrete subclass pointed at the (shared) benchmark fixture."""

    _folder = "kge_benchmark_toy"
    _ground_truth_files: ClassVar[tuple[str, ...]] = ("sup_ent_ids", "ref_ent_ids")
    _train_ground_truth_file = "sup_ent_ids"
    _test_ground_truth_file = "ref_ent_ids"


def _zip_url(tmp_path: Path) -> str:
    zip_path = tmp_path / "data.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        for file in _FIXTURES_DIR.iterdir():
            archive.write(file, arcname=f"kge_benchmark_toy/{file.name}")
    return f"file://{zip_path}"


def test_alinet_linker_reports_ranked_metrics_on_held_out_split(tmp_path: Path) -> None:
    dataset = _BenchmarkDataset(zip_url=_zip_url(tmp_path))
    entities1, entities2, _ = dataset.load()
    train_pairs, test_pairs, _ = dataset.load_splits()
    graph1, graph2 = dataset.load_graphs()
    graph = to_triples(graph1) + to_triples(graph2)

    linker = AliNetLinker(
        layer_dims=[16, 16, 8],
        num_epochs=150,
        learning_rate=0.05,
        batch_size=16,
        neg_triple_num=3,
    )
    linker.fit(entities1, entities2, ground_truth=train_pairs, graph=graph, random_state=0)

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

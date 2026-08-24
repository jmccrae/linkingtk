"""Trains GCNAlignLinker on OpenEA's EN-FR-15K-V1 dataset using its native
train/test split and reports Hits@1, Hits@10, MRR via
linkingtk.eval.Evaluator.evaluate_ranked.

Same methodology as kge_benchmark.py/bootea_benchmark.py: train pairs seed
the margin loss, test pairs are held out for ranked evaluation via
linkingtk.eval.rank_exhaustive (no blocking), matching OpenEA's own
evaluation methodology.

Note: this port only implements GCN-Align's structural (``se``) branch, not
the attribute (``ae``) branch OpenEA's own published EN-FR-15K-V1 numbers
use (Hits@1=0.338, Hits@10=0.680, MRR=0.451, ``test_method: "sa"``,
``beta=0.9`` -- see the module docstring on
linkingtk.algorithms.ea.gcn_align for details). Expect a lower Hits@1 here
than that published number -- this is a documented, deliberate scope
reduction, not a bug.

Requires the `kge` optional dependency group (for `torch`) — install with
`uv sync --extra kge`. Fetches a zip over the network the first time it's
run; cached under ~/.cache/linkingtk/downloads after that.

Run with: `uv run python examples/gcn_align_benchmark.py`
"""

from __future__ import annotations

from linkingtk.algorithms.ea import GCNAlignLinker
from linkingtk.datasets import EnFr15KDataset
from linkingtk.eval import Evaluator, rank_exhaustive
from linkingtk.utils.graph import to_triples


def main() -> None:
    dataset = EnFr15KDataset()
    entities1, entities2, _ = dataset.load()
    train_pairs, test_pairs, val_pairs = dataset.load_splits()
    graph1, graph2 = dataset.load_graphs()
    graph = to_triples(graph1) + to_triples(graph2)

    linker = GCNAlignLinker(num_epochs=500, device="cuda")
    linker.fit(
        entities1,
        entities2,
        ground_truth=train_pairs,
        graph=graph,
        random_state=0,
        val_ground_truth=val_pairs,
    )

    test_source_ids = {s for s, _ in test_pairs}
    test_target_ids = {t for _, t in test_pairs}
    ranked_predictions = rank_exhaustive(
        linker,
        [e for e in entities1 if e.id in test_source_ids],
        [e for e in entities2 if e.id in test_target_ids],
    )
    report = Evaluator.evaluate_ranked(ranked_predictions, ground_truth=test_pairs, top_k=[1, 10])
    print(f"{len(train_pairs)} train / {len(test_pairs)} test pairs")
    print("Metrics:", report.metrics)
    print("Published OpenEA GCN-Align (se+ae) EN-FR-15K-V1: Hits@1=0.338, Hits@10=0.680, MRR=0.451")
    print("(structural-only port here -- not directly comparable, see module docstring)")


if __name__ == "__main__":
    main()

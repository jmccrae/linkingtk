"""Trains AliNetLinker on OpenEA's EN-FR-15K-V1 dataset using its native
train/test split and reports Hits@1, Hits@10, MRR via
linkingtk.eval.Evaluator.evaluate_ranked.

Same methodology as gcn_align_benchmark.py/rdgcn_benchmark.py: train pairs
seed the margin loss, test pairs are held out for ranked evaluation via
linkingtk.eval.rank_exhaustive (no blocking), matching OpenEA's own
evaluation methodology.

Unlike GCN-Align and RDGCN, **no published EN-FR-15K-V1 number exists for
AliNet** to compare against -- it was added to OpenEA after
docs/detailed_results_current_approaches_15K.csv's last update; its own
numbers live only in an external Google Sheet linked from OpenEA's README,
not reliably fetchable. This script exists for manual sanity-checking only
(metrics land in a plausible range, comparable to or better than
GCN-Align's own structural-only numbers given AliNet is a newer, generally
stronger method per its paper -- a loose informal expectation, not a
target).

AliNet is structural-only (no attribute-triple dependency), so this uses
EnFr15KDataset like gcn_align_benchmark.py, not EnFr15KAttrDataset.

Requires the `kge` optional dependency group (for `torch`) — install with
`uv sync --extra kge`. Fetches a zip over the network the first time it's
run; cached under ~/.cache/linkingtk/downloads after that.

Run with: `uv run python examples/alinet_benchmark.py`
"""

from __future__ import annotations

from linkingtk.algorithms.ea import AliNetLinker
from linkingtk.datasets import EnFr15KDataset
from linkingtk.eval import Evaluator, rank_exhaustive
from linkingtk.utils.graph import to_triples


def main() -> None:
    dataset = EnFr15KDataset()
    entities1, entities2, _ = dataset.load()
    train_pairs, test_pairs, val_pairs = dataset.load_splits()
    graph1, graph2 = dataset.load_graphs()
    graph = to_triples(graph1) + to_triples(graph2)

    linker = AliNetLinker(num_epochs=500, device="cuda")  # layer_dims=[500, 400, 300]
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
    print("No published OpenEA AliNet EN-FR-15K-V1 number exists -- see module docstring.")


if __name__ == "__main__":
    main()

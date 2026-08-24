"""Trains GCNAlignLinker (structural + attribute branches) on OpenEA's
EN-FR-15K-V1 dataset (native format, with real attribute triples) using
its own train/test split, and reports Hits@1, Hits@10, MRR via
linkingtk.eval.Evaluator.evaluate_ranked.

Like kdcoe_benchmark.py/rdgcn_benchmark.py, this sources data from
EnFr15KAttrDataset, not EnFr15KDataset -- the attribute (``ae``) branch
needs attribute triples, and EnFr15KDataset's numeric-id rehost has none
at all. Train pairs seed both branches' margin loss; test pairs are held
out for ranked evaluation via linkingtk.eval.rank_exhaustive (no
blocking). Scoring uses Manhattan distance + CSLS(k=10), matching OpenEA's
own configured evaluation methodology for this method
(``eval_metric: "manhattan"``, ``eval_norm: false``, ``csls: 10`` in
``run/args/gcnalign_args_15K.json``) -- see the module docstring on
linkingtk.algorithms.ea.gcn_align for why cosine similarity (this
package's usual default) undershoots the published number.

Requires the `kge` optional dependency group (for `torch`) — install with
`uv sync --extra kge`. Fetches a zip over the network the first time it's
run; cached under ~/.cache/linkingtk/downloads after that.

Run with: `uv run python examples/gcn_align_benchmark.py`
"""

from __future__ import annotations

from linkingtk.algorithms.ea import GCNAlignLinker
from linkingtk.datasets import EnFr15KAttrDataset
from linkingtk.eval import Evaluator, rank_exhaustive
from linkingtk.utils.graph import to_triples


def main() -> None:
    dataset = EnFr15KAttrDataset()
    entities1, entities2, _ = dataset.load()
    train_pairs, test_pairs, val_pairs = dataset.load_splits()
    graph1, graph2 = dataset.load_graphs()
    graph = to_triples(graph1) + to_triples(graph2)
    attribute_triples1, attribute_triples2 = dataset.load_attribute_triples()

    linker = GCNAlignLinker(num_epochs=500, use_attributes=True, device="cuda")
    linker.fit(
        entities1,
        entities2,
        ground_truth=train_pairs,
        graph=graph,
        random_state=0,
        val_ground_truth=val_pairs,
        attribute_triples1=attribute_triples1,
        attribute_triples2=attribute_triples2,
    )

    test_source_ids = {s for s, _ in test_pairs}
    test_target_ids = {t for _, t in test_pairs}
    ranked_predictions = rank_exhaustive(
        linker,
        [e for e in entities1 if e.id in test_source_ids],
        [e for e in entities2 if e.id in test_target_ids],
        metric="manhattan",
        csls_k=10,
    )
    report = Evaluator.evaluate_ranked(ranked_predictions, ground_truth=test_pairs, top_k=[1, 10])
    print(f"{len(train_pairs)} train / {len(test_pairs)} test pairs")
    print("Metrics (manhattan + csls=10, matching OpenEA's own methodology):", report.metrics)
    print("Published OpenEA GCN-Align (se+ae) EN-FR-15K-V1: Hits@1=0.338, Hits@10=0.680, MRR=0.451")


if __name__ == "__main__":
    main()

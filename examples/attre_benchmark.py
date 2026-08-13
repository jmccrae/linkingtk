"""Trains AttrELinker on OpenEA's EN-FR-15K-V1 dataset (native format, with
attribute triples) using its own train/test/validation split, and reports
Hits@1, Hits@10, MRR via linkingtk.eval.Evaluator.evaluate_ranked.

Like jape_benchmark.py/kdcoe_benchmark.py, this sources data from
EnFr15KAttrDataset, not EnFr15KDataset as issue #31's text suggests --
AttrE's character/attribute-embedding half needs attribute triples, and
EnFr15KDataset's rehost has none at all. Train pairs seed the shared
embedding tables (both the structural and character/attribute spaces);
validation pairs drive early stopping; test pairs are held out and only
used to score ranked predictions.

Requires the `kge` optional dependency group (for `torch`) — install
with `uv sync --extra kge`. Fetches a multi-MB zip over the network the
first time it's run; cached under ~/.cache/linkingtk/downloads after that.

Run with: `uv run python examples/attre_benchmark.py`
"""

from __future__ import annotations

from linkingtk.algorithms.ea import AttrELinker
from linkingtk.blocking import LabelOverlap
from linkingtk.datasets import EnFr15KAttrDataset
from linkingtk.eval import Evaluator
from linkingtk.utils.graph import to_triples


def main() -> None:
    dataset = EnFr15KAttrDataset()
    entities1, entities2, _ = dataset.load()
    train_pairs, test_pairs, val_pairs = dataset.load_splits()
    graph1, graph2 = dataset.load_graphs()
    graph = to_triples(graph1) + to_triples(graph2)
    attribute_triples1, attribute_triples2 = dataset.load_attribute_triples()

    linker = AttrELinker(num_epochs=500)  # embedding_dim=100, batch_size=5000
    linker.fit(
        entities1,
        entities2,
        ground_truth=train_pairs,
        graph=graph,
        attribute_triples1=attribute_triples1,
        attribute_triples2=attribute_triples2,
        random_state=0,
        val_ground_truth=val_pairs,
        patience=5,
        eval_every=10,
    )
    results = linker.link(entities1, entities2, blocking=LabelOverlap(max_matches=10))

    ranked_predictions = [(r.source_id, [r.target_id, *r.alternatives]) for r in results]
    report = Evaluator.evaluate_ranked(ranked_predictions, ground_truth=test_pairs, top_k=[1, 10])
    print(f"{len(train_pairs)} train / {len(val_pairs)} val / {len(test_pairs)} test pairs")
    print("Metrics:", report.metrics)


if __name__ == "__main__":
    main()

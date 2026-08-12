"""Trains JAPELinker on OpenEA's EN-FR-15K-V1 dataset (native format, with
attribute triples) using its own train/test/validation split, and reports
Hits@1, Hits@10, MRR via linkingtk.eval.Evaluator.evaluate_ranked.

Unlike mtranse_benchmark.py/kge_benchmark.py/iptranse_benchmark.py, this
sources data from EnFr15KAttrDataset, not EnFr15KDataset -- JAPE's whole
point is the attribute-correlation signal, and EnFr15KDataset's rehost
(github.com/DexterZeng/EntMatcher) has no attribute triples at all.
EnFr15KAttrDataset (Hugging Face's matchbench/openea-en-fr-15k-v1) is a
different, independently-sampled cut of "EN-FR-15K" with the same official
split-size ratios but a different entity roster -- see
docs/datasets/real_world_ea.md and linkingtk.datasets.openea_native's
module docstring for why. Train pairs seed the shared embedding table and
merge attribute vocabularies for cross-lingual correlation training;
validation pairs drive early stopping; test pairs are held out and only
used to score ranked predictions.

Requires the `kge` optional dependency group (for `torch`) — install
with `uv sync --extra kge`. Fetches a multi-MB zip over the network the
first time it's run; cached under ~/.cache/linkingtk/downloads after that.

Run with: `uv run python examples/jape_benchmark.py`
"""

from __future__ import annotations

from linkingtk.algorithms.ea import JAPELinker
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

    linker = JAPELinker(num_epochs=500)  # embedding_dim=100, batch_size=5000
    linker.fit(
        entities1,
        entities2,
        ground_truth=train_pairs,
        graph=graph,
        random_state=0,
        val_ground_truth=val_pairs,
        patience=5,
        eval_every=10,
        attribute_triples1=attribute_triples1,
        attribute_triples2=attribute_triples2,
    )
    results = linker.link(entities1, entities2, blocking=LabelOverlap(max_matches=10))

    ranked_predictions = [(r.source_id, [r.target_id, *r.alternatives]) for r in results]
    report = Evaluator.evaluate_ranked(ranked_predictions, ground_truth=test_pairs, top_k=[1, 10])
    print(f"{len(train_pairs)} train / {len(val_pairs)} val / {len(test_pairs)} test pairs")
    print("Metrics:", report.metrics)


if __name__ == "__main__":
    main()

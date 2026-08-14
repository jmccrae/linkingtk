"""Trains KDCoELinker on OpenEA's EN-FR-15K-V1 dataset (native format, with
attribute triples) using its own train/test/validation split, and reports
Hits@1, Hits@10, MRR via linkingtk.eval.Evaluator.evaluate_ranked.

Like jape_benchmark.py, this sources data from EnFr15KAttrDataset, not
EnFr15KDataset -- KDCoE's description-encoder pathway needs attribute
triples (for entities that have a real ``.../description``-predicate
triple; entities without one fall back to their own label, see
linkingtk.algorithms.ea.kdcoe's module docstring) and EnFr15KDataset's
rehost has none at all. Train pairs seed both the structural+mapping phase
and the description phase; validation pairs drive early stopping in both;
test pairs are held out and only used to score ranked predictions. Ranking
is exhaustive (every test-source entity against every test-target entity,
via linkingtk.eval.rank_exhaustive, no blocking/candidate restriction),
matching OpenEA's own evaluation methodology.

Requires the `kge` optional dependency group (for `torch`) — install with
`uv sync --extra kge`. Fetches OpenEA's dataset zip (multi-MB) *and*
fastText's pretrained word vectors (`wiki-news-300d-1M.vec.zip`, ~681MB)
over the network the first time it's run -- both cached under
~/.cache/linkingtk/downloads after that. This benchmark takes noticeably
longer to start than the others in this milestone because of that second,
much larger download plus the co-training loop's two alternating training
phases per iteration.

Run with: `uv run python examples/kdcoe_benchmark.py`
"""

from __future__ import annotations

from linkingtk.algorithms.ea import KDCoELinker
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

    linker = KDCoELinker(num_epochs=500)  # embedding_dim=100, batch_size=5000
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
    test_source_ids = {s for s, _ in test_pairs}
    test_target_ids = {t for _, t in test_pairs}
    ranked_predictions = rank_exhaustive(
        linker,
        [e for e in entities1 if e.id in test_source_ids],
        [e for e in entities2 if e.id in test_target_ids],
    )
    report = Evaluator.evaluate_ranked(ranked_predictions, ground_truth=test_pairs, top_k=[1, 10])
    print(f"{len(train_pairs)} train / {len(val_pairs)} val / {len(test_pairs)} test pairs")
    print("Metrics:", report.metrics)


if __name__ == "__main__":
    main()

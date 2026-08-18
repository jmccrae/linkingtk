"""Trains SEALinker on OpenEA's EN-FR-15K-V1 dataset using its native
train/test/validation split and reports Hits@1, Hits@10, MRR via
linkingtk.eval.Evaluator.evaluate_ranked.

Train pairs seed the supervised half of the mapping loss; validation pairs
drive early stopping; test pairs are held out and only used to score ranked
predictions -- this measures genuine generalization to entities the model
wasn't directly told to align. Unlike MTransE, SEA also trains a
semi-supervised cycle-consistency loss over *unlabeled* entities (every
entity not in the train split, both KG1 and KG2 side, sampled independently
-- no pairing needed for that term). Ranking is exhaustive (every
test-source entity against every test-target entity, via
linkingtk.eval.rank_exhaustive, no blocking/candidate restriction),
matching OpenEA's own evaluation methodology -- see
docs/examples/ea_kge_benchmarks.md for methodology details and
mtranse_benchmark.py for SEA's single-mapping-matrix predecessor.

Requires the `kge` optional dependency group (for `torch`) — install
with `uv sync --extra kge`. Fetches a multi-MB zip over the network the
first time it's run; cached under ~/.cache/linkingtk/downloads after that.

Run with: `uv run python examples/sea_benchmark.py`
"""

from __future__ import annotations

from linkingtk.algorithms.ea import SEALinker
from linkingtk.datasets import EnFr15KAttrDataset
from linkingtk.eval import Evaluator, rank_exhaustive
from linkingtk.utils.graph import to_triples


def main() -> None:
    dataset = EnFr15KAttrDataset()
    entities1, entities2, _ = dataset.load()
    train_pairs, test_pairs, val_pairs = dataset.load_splits()
    graph1, graph2 = dataset.load_graphs()
    graph = to_triples(graph1) + to_triples(graph2)

    linker = SEALinker(num_epochs=500)  # embedding_dim=100, batch_size=5000
    linker.fit(
        entities1,
        entities2,
        ground_truth=train_pairs,
        graph=graph,
        random_state=0,
        val_ground_truth=val_pairs,
        patience=5,
        eval_every=10,
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

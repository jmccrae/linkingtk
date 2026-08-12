"""Trains IPTransELinker on OpenEA's EN-FR-15K-V1 dataset using its native
train/test/validation split and reports Hits@1, Hits@10, MRR via
linkingtk.eval.Evaluator.evaluate_ranked.

Train pairs seed the shared embedding table (each pair's target entity is
aliased to its source's id); validation pairs drive early stopping; test
pairs are held out and only used to score ranked predictions -- this
measures genuine generalization to entities the model wasn't directly told
to align. Every bootstrap_every epochs, an unsupervised self-training round
finds additional high-confidence matches by structural embedding similarity
alone and trains on their pseudo-triples too. See
docs/examples/ea_kge_benchmarks.md for methodology details and
mtranse_benchmark.py for the (bootstrapping-free) MTransE equivalent.

Requires the `kge` optional dependency group (for `torch`) — install
with `uv sync --extra kge`. Fetches a ~28MB zip over the network the
first time it's run (shared by all DBP15K/OpenEA datasets); cached under
~/.cache/linkingtk/downloads after that.

Run with: `uv run python examples/iptranse_benchmark.py`
"""

from __future__ import annotations

from linkingtk.algorithms.ea import IPTransELinker
from linkingtk.blocking import LabelOverlap
from linkingtk.datasets import EnFr15KDataset
from linkingtk.eval import Evaluator
from linkingtk.utils.graph import to_triples


def main() -> None:
    dataset = EnFr15KDataset()
    entities1, entities2, _ = dataset.load()
    train_pairs, test_pairs, val_pairs = dataset.load_splits()
    graph1, graph2 = dataset.load_graphs()
    graph = to_triples(graph1) + to_triples(graph2)

    linker = IPTransELinker(num_epochs=500)  # embedding_dim=100, batch_size=5000
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
    results = linker.link(entities1, entities2, blocking=LabelOverlap(max_matches=10))

    ranked_predictions = [(r.source_id, [r.target_id, *r.alternatives]) for r in results]
    report = Evaluator.evaluate_ranked(ranked_predictions, ground_truth=test_pairs, top_k=[1, 10])
    print(f"{len(train_pairs)} train / {len(val_pairs)} val / {len(test_pairs)} test pairs")
    print("Metrics:", report.metrics)


if __name__ == "__main__":
    main()

"""Trains RSN4EALinker on OpenEA's EN-FR-15K-V1 dataset using its native
train/test/validation split and reports Hits@1, Hits@10, MRR via
linkingtk.eval.Evaluator.evaluate_ranked.

Unlike every other linker in this milestone, RSN4EA has no separate
mapping/alignment loss channel at all: train pairs are used to build an
alias-expanded graph (every seed pair's two ids become mutually
substitutable throughout the walkable graph -- see
linkingtk.algorithms.ea.rsn4ea's module docstring), then biased random
walks are sampled over that graph and a Recurrent Skipping Network is
trained to predict each walk's next entity/relation token. Validation pairs
drive early stopping; test pairs are held out and only used to score ranked
predictions. Ranking is exhaustive (every test-source entity against every
test-target entity, via linkingtk.eval.rank_exhaustive, no blocking/
candidate restriction), matching OpenEA's own evaluation methodology -- see
docs/examples/ea_kge_benchmarks.md for methodology details.

Requires the `kge` optional dependency group (for `torch`) — install
with `uv sync --extra kge`. Fetches a multi-MB zip over the network the
first time it's run; cached under ~/.cache/linkingtk/downloads after that.

Path sampling is CPU-bound single-threaded Python regardless of GPU and can
take a while at real dataset scale -- see `max_paths` on RSN4EALinker if
this needs capping.

Run with: `uv run python examples/rsn4ea_benchmark.py`
"""

from __future__ import annotations

from linkingtk.algorithms.ea import RSN4EALinker
from linkingtk.datasets import EnFr15KAttrDataset
from linkingtk.eval import Evaluator, rank_exhaustive
from linkingtk.utils.graph import to_triples


def main() -> None:
    dataset = EnFr15KAttrDataset()
    entities1, entities2, _ = dataset.load()
    train_pairs, test_pairs, val_pairs = dataset.load_splits()
    graph1, graph2 = dataset.load_graphs()
    graph = to_triples(graph1) + to_triples(graph2)

    linker = RSN4EALinker()  # hidden_size=100, num_epochs=30, batch_size=512
    linker.fit(
        entities1,
        entities2,
        ground_truth=train_pairs,
        graph=graph,
        random_state=0,
        val_ground_truth=val_pairs,
        patience=5,
        eval_every=3,
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

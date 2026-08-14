"""Trains KGELinker on OpenEA's EN-FR-15K-V1 dataset using its native
train/test split and reports Hits@1, Hits@10, MRR via
linkingtk.eval.Evaluator.evaluate_ranked.

Train pairs are added to fit() as seed alignment triples (bridging the
English and French DBpedia graphs); test pairs are held out and only used
to score ranked predictions -- unlike kge_ea.py's fully-seeded
pipeline-correctness demo, this measures genuine generalization to
entities the model wasn't directly told to align. Ranking is exhaustive
(every test-source entity against every test-target entity, via
linkingtk.eval.rank_exhaustive, no blocking/candidate restriction),
matching OpenEA's own evaluation methodology -- see
docs/examples/ea_kge_benchmarks.md for methodology details and how this
compares to other knowledge-graph-embedding EA linkers.

Requires the `kge` optional dependency group (for `pykeen`) — install
with `uv sync --extra kge`. Fetches a ~28MB zip over the network the
first time it's run (shared by all DBP15K/OpenEA datasets); cached under
~/.cache/linkingtk/downloads after that.

Run with: `uv run python examples/kge_benchmark.py`
"""

from __future__ import annotations

from linkingtk.algorithms.ea import KGELinker
from linkingtk.datasets import EnFr15KDataset
from linkingtk.eval import Evaluator, rank_exhaustive
from linkingtk.utils.graph import to_triples


def main() -> None:
    dataset = EnFr15KDataset()
    entities1, entities2, _ = dataset.load()
    train_pairs, test_pairs, _val_pairs = dataset.load_splits()
    graph1, graph2 = dataset.load_graphs()
    graph = to_triples(graph1) + to_triples(graph2)

    linker = KGELinker(num_epochs=20)  # TransE, embedding_dim=50
    linker.fit(entities1, entities2, ground_truth=train_pairs, graph=graph, random_state=0)

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


if __name__ == "__main__":
    main()

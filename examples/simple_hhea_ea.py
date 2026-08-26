"""Trains SimpleHHEALinker on ICEWS-WIKI (a heterogeneous event-KG-to-
Wikipedia EA benchmark) using its native train/test split and reports
Hits@1/Hits@10/MRR via linkingtk.eval.Evaluator.evaluate_ranked, alongside
the paper's own published Simple-HHEA number for the same dataset (see
docs/examples/simple_hhea_ea.md).

Train pairs seed both the margin-ranking training signal and the
node2vec structural node-merge (see
linkingtk.algorithms.ea._simple_hhea_structure); test pairs are held out
and only used to score ranked predictions. Ranking is exhaustive (every
test-source entity against every test-target entity, via
linkingtk.eval.rank_exhaustive, no blocking/candidate restriction) with
Cross-domain Similarity Local Scaling (metric="cosine", csls_k=10) --
matching the reference's own CSLS_.py evaluation, which L2-normalizes
both embedding matrices *before* its `sim_handler`'s raw `matmul`
(`main_SimpleHHEA.py`'s `evaluate()`), making that raw matmul equivalent
to cosine similarity, not an unnormalized inner product.

Requires the `kge` optional dependency group (for `node2vec`/`gensim`) --
install with `uv sync --extra kge`. Fetches ICEWS-WIKI's zip over the
network the first time it's run (~75MB, includes ~3.5M ICEWS event
triples); cached under ~/.cache/linkingtk/downloads after that. Trains on
GPU (device="cuda") -- this is a large, heterogeneous dataset (~27K
entities combined), unlike the toy/15K-scale fixtures most other EA
benchmarks in this repo use.

ICEWS-WIKI's combined graph has an extreme power-law degree distribution
(median node degree 18, one node with 273,317 edges -- measured directly),
which makes node2vec's per-edge alias-table precompute (`O(degree)` each,
`O(degree^2)` for a hub node) intractable without
`SimpleHHEALinker`'s `max_degree` cap (default 1000, randomly subsamples
any node's excess edges -- see
linkingtk.algorithms.ea._simple_hhea_structure.cap_node_degree). Even
capped, the precompute step alone takes several minutes and is
single-threaded (not parallelizable); `structure_workers` (default `1`)
would speed up the separate walk-simulation step via multiprocessing, but
is left at its default here -- raising it spawns `joblib`/`loky` worker
processes, which this sandbox's environment doesn't tolerate well (killed
outright on a real attempt).

Run with: `uv run python examples/simple_hhea_ea.py`
"""

from __future__ import annotations

from linkingtk.algorithms.ea import SimpleHHEALinker
from linkingtk.datasets.icews import IcewsWikiDataset
from linkingtk.eval import Evaluator, rank_exhaustive
from linkingtk.utils.graph import to_triples


def main() -> None:
    dataset = IcewsWikiDataset()
    entities1, entities2, _ = dataset.load()
    train_pairs, test_pairs, _val_pairs = dataset.load_splits()
    graph1, graph2 = dataset.load_graphs()
    graph = to_triples(graph1) + to_triples(graph2)
    temporal1, temporal2 = dataset.load_temporal_graphs()

    linker = SimpleHHEALinker(device="cuda")
    linker.fit(
        entities1,
        entities2,
        ground_truth=train_pairs,
        graph=graph,
        temporal_triples=temporal1 + temporal2,
        random_state=0,
    )

    test_source_ids = {s for s, _ in test_pairs}
    test_target_ids = {t for _, t in test_pairs}
    ranked_predictions = rank_exhaustive(
        linker,
        [e for e in entities1 if e.id in test_source_ids],
        [e for e in entities2 if e.id in test_target_ids],
        metric="cosine",
        csls_k=10,
    )
    report = Evaluator.evaluate_ranked(ranked_predictions, ground_truth=test_pairs, top_k=[1, 10])
    print(f"{len(train_pairs)} train / {len(test_pairs)} test pairs")
    print("Metrics:", report.metrics)


if __name__ == "__main__":
    main()

"""Trains MultiKELinker on OpenEA's EN-FR-15K-V1 dataset (native format,
with attribute triples) using its own train/test/validation split, and
reports Hits@1, Hits@10, MRR via linkingtk.eval.Evaluator.evaluate_ranked.

Like jape_benchmark.py/kdcoe_benchmark.py/attre_benchmark.py/
imuse_benchmark.py, this sources data from EnFr15KAttrDataset, not
EnFr15KDataset as issue #35's text suggests -- MultiKE's attribute view
and literal encoder both need attribute triples, and EnFr15KDataset's
rehost has none at all.

Unlike IMUSE, MultiKE is supervised: `train_pairs` seeds its cross-KG
entity inference loss, its only real supervision channel. Validation
pairs drive early stopping; test pairs are held out and only used to
score ranked predictions. Ranking is exhaustive (every test-source entity
against every test-target entity, via linkingtk.eval.rank_exhaustive, no
blocking/candidate restriction), matching OpenEA's own evaluation
methodology.

Requires the `kge` optional dependency group (for `torch`) -- install
with `uv sync --extra kge`. `transformers` (used for the literal encoder)
is already a base dependency. First run downloads both a multi-MB zip
(the dataset, cached under ~/.cache/linkingtk/downloads) and
`distilbert-base-multilingual-cased` (~540MB, cached by `transformers`
itself) over the network.

See linkingtk.algorithms.ea.multike's module docstring for the full
architecture writeup and its three documented scope cuts (space mapping
is dead code in OpenEA's own published run; predicate soft-alignment is
computed once rather than periodically re-estimated; negative sampling
follows this family's established uniform-sampling precedent rather than
OpenEA's truncated/nearest-neighbor optimization).

Run with: `uv run python examples/multike_benchmark.py`
"""

from __future__ import annotations

from linkingtk.algorithms.ea import MultiKELinker
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

    linker = MultiKELinker(num_epochs=2000)  # embedding_dim=100, batch_size=5000 (defaults)
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

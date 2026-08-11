"""Entity Alignment with a knowledge-graph-embedding linker.

Aligns two small, isomorphic knowledge graphs -- two 4-entity chains
linked by a `next` relation -- by training a TransE embedding
(`KGELinker`) jointly over both graphs plus the known alignment pairs
added as extra "seed" triples, then scoring unaligned candidates by
cosine similarity of their trained embeddings. Unlike
`feature_classifier_ea.py`, matching here comes entirely from graph
*structure*, not entity labels/text.

This is a pipeline-correctness demo, not a generalization benchmark: every
ground-truth pair is given to `fit()` as a seed, so it demonstrates the
KGE training/scoring pipeline works correctly, not how well it aligns
entities it wasn't directly told about.

Run with: `uv run python examples/kge_ea.py`
"""

from linkingtk.algorithms.ea import KGELinker
from linkingtk.blocking.base import BlockingStrategy
from linkingtk.core.entity import Entity
from linkingtk.eval import Evaluator


class AllPairs(BlockingStrategy):
    """Lets every pair through -- entities here don't share labels across
    KGs, so ExactMatch's default blocking would find nothing.
    """

    def candidate_pairs(
        self, dataset1: list[Entity], dataset2: list[Entity]
    ) -> list[tuple[Entity, Entity]]:
        return [(e1, e2) for e1 in dataset1 for e2 in dataset2]


def main() -> None:
    kg1 = [Entity(id=f"kg1:{c}", labels=[c]) for c in "abcd"]
    kg2 = [Entity(id=f"kg2:{c}", labels=[c]) for c in "wxyz"]
    graph = [
        ("kg1:a", "next", "kg1:b"),
        ("kg1:b", "next", "kg1:c"),
        ("kg1:c", "next", "kg1:d"),
        ("kg2:w", "next", "kg2:x"),
        ("kg2:x", "next", "kg2:y"),
        ("kg2:y", "next", "kg2:z"),
    ]
    ground_truth = [("kg1:a", "kg2:w"), ("kg1:b", "kg2:x"), ("kg1:c", "kg2:y"), ("kg1:d", "kg2:z")]

    linker = KGELinker(embedding_dim=16, num_epochs=100)
    linker.fit(kg1, kg2, ground_truth, graph=graph, random_state=0)
    results = linker.link(kg1, kg2, blocking=AllPairs())

    for result in results:
        print(f"{result.source_id} -> {result.target_id} (score={result.score:.2f})")

    predictions = [(r.source_id, r.target_id) for r in results]
    report = Evaluator.evaluate(predictions=predictions, ground_truth=ground_truth)
    print("Metrics:", report.metrics)


if __name__ == "__main__":
    main()

"""Entity Alignment baseline with StringSimilarityLinker.

Aligns two small knowledge graphs describing the same three cities, but
with slightly different labels (kg2 appends the country name), so an
exact label match would miss all three pairs. LabelOverlap blocking
(character n-gram overlap) finds the candidates despite the mismatch,
and StringSimilarityLinker picks the best one by Jaccard token overlap.

Run with: `uv run python examples/string_similarity_ea.py`
"""

from linkingtk.algorithms.string_similarity import StringSimilarityLinker
from linkingtk.blocking import LabelOverlap
from linkingtk.core import Entity
from linkingtk.eval import Evaluator


def main() -> None:
    kg1 = [
        Entity(id="kg1:paris", labels=["Paris"]),
        Entity(id="kg1:berlin", labels=["Berlin"]),
        Entity(id="kg1:rome", labels=["Rome"]),
    ]
    kg2 = [
        Entity(id="kg2:paris", labels=["Paris, France"]),
        Entity(id="kg2:berlin", labels=["Berlin, Germany"]),
        Entity(id="kg2:rome", labels=["Rome, Italy"]),
    ]

    linker = StringSimilarityLinker(source_field="label", target_field="label", metric="jaccard")
    results = linker.link(kg1, kg2, blocking=LabelOverlap(max_matches=3))

    for result in results:
        print(f"{result.source_id} -> {result.target_id} (score={result.score:.2f})")

    predictions = [(r.source_id, r.target_id) for r in results]
    ground_truth = [
        ("kg1:paris", "kg2:paris"),
        ("kg1:berlin", "kg2:berlin"),
        ("kg1:rome", "kg2:rome"),
    ]
    report = Evaluator.evaluate(predictions=predictions, ground_truth=ground_truth)
    print("Metrics:", report.metrics)


if __name__ == "__main__":
    main()

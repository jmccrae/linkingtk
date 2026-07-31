"""Minimal end-to-end example: block candidate pairs and evaluate them.

Run with: `uv run python examples/basic_exact_match.py`
"""

from linkingtk.blocking import ExactMatch
from linkingtk.core import Entity
from linkingtk.eval import Evaluator


def main() -> None:
    dataset1 = [
        Entity(id="s1", labels=["cat"]),
        Entity(id="s2", labels=["dog"]),
    ]
    dataset2 = [
        Entity(id="t1", labels=[("cat", "en")]),
        Entity(id="t2", labels=["fish"]),
    ]

    pairs = ExactMatch().candidate_pairs(dataset1, dataset2)
    predictions = [(e1.id, e2.id) for e1, e2 in pairs]
    print("Candidate pairs:", predictions)

    ground_truth = [("s1", "t1"), ("s2", "t2")]
    report = Evaluator.evaluate(predictions=predictions, ground_truth=ground_truth)
    print("Metrics:", report.metrics)

    blocking_report = Evaluator.evaluate_blocking(
        candidate_pairs=predictions,
        ground_truth=ground_truth,
        dataset1_size=len(dataset1),
        dataset2_size=len(dataset2),
    )
    print("Blocking metrics:", blocking_report.metrics)


if __name__ == "__main__":
    main()

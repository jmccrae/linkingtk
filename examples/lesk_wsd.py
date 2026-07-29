"""Word Sense Disambiguation with the Lesk algorithm.

Disambiguates the classic ambiguous word "bank" between its financial and
riverbank senses, by comparing the mention's context against each
candidate sense's gloss.

Run with: `uv run python examples/lesk_wsd.py`
"""

from linkingtk.algorithms.wsd import LeskLinker
from linkingtk.core import Entity
from linkingtk.eval import Evaluator


def main() -> None:
    mentions = [
        Entity(
            id="m1",
            labels=["bank"],
            context="I deposited money at the bank yesterday",
        ),
    ]
    senses = [
        Entity(
            id="bank.n.01",
            labels=["bank"],
            description="a financial institution that accepts deposits of money",
        ),
        Entity(
            id="bank.n.02",
            labels=["bank"],
            description="the land alongside a river",
        ),
    ]

    results = LeskLinker().link(mentions, senses)
    for result in results:
        print(f"{result.source_id} -> {result.target_id} (score={result.score})")
        print(f"  alternatives: {result.alternatives}")

    predictions = [(r.source_id, r.target_id) for r in results]
    ground_truth = [("m1", "bank.n.01")]
    report = Evaluator.evaluate(predictions=predictions, ground_truth=ground_truth)
    print("Metrics:", report.metrics)


if __name__ == "__main__":
    main()

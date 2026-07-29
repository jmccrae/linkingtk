"""Running the bundled toy datasets end-to-end.

ToyWSDDataset and ToyELDataset need no network access, so they're a quick
way to sanity-check a linker without fetching anything external.

Run with: `uv run python examples/toy_datasets.py`
"""

from linkingtk.algorithms.string_similarity import StringSimilarityLinker
from linkingtk.algorithms.wsd import LeskLinker
from linkingtk.datasets import ToyELDataset, ToyWSDDataset
from linkingtk.eval import Evaluator


def main() -> None:
    mentions, senses, wsd_truth = ToyWSDDataset().load()
    wsd_results = LeskLinker().link(mentions, senses)
    wsd_predictions = [(r.source_id, r.target_id) for r in wsd_results]
    print("WSD:", wsd_predictions)
    print(
        "Metrics:", Evaluator.evaluate(predictions=wsd_predictions, ground_truth=wsd_truth).metrics
    )

    mentions, kb, el_truth = ToyELDataset().load()
    el_linker = StringSimilarityLinker(
        source_field="context", target_field="description", metric="word_overlap"
    )
    el_results = el_linker.link(mentions, kb)
    el_predictions = [(r.source_id, r.target_id) for r in el_results]
    print("EL:", el_predictions)
    print("Metrics:", Evaluator.evaluate(predictions=el_predictions, ground_truth=el_truth).metrics)


if __name__ == "__main__":
    main()

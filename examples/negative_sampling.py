"""Mining hard negatives to train a feature-based classifier.

`sample_hard_negatives` mines, per source entity, the top-K most
confusable non-matching candidates that `LabelOverlap` blocking surfaced --
more informative training signal for `FeatureClassifierLinker.fit` than
negatives sampled uniformly at random from the full candidate pool, since
blocking already considered these plausible enough to return.

`LabelOverlap(ngram_size=1, ...)` is used here (rather than the
`max_matches=3` used in `string_similarity_ea.py`) so that blocking
surfaces both correct and incorrect candidate pairs to mine negatives
from -- on a dataset this small, tighter blocking finds only the true
matches, leaving nothing to mine.

Run with: `uv run python examples/negative_sampling.py`
"""

from linkingtk.algorithms.feature_classifier import FeatureClassifierLinker
from linkingtk.blocking import LabelOverlap, sample_hard_negatives
from linkingtk.datasets import ToyEADataset
from linkingtk.eval import Evaluator


def main() -> None:
    kg1, kg2, ground_truth = ToyEADataset().load()
    blocking = LabelOverlap(ngram_size=1, max_matches=10)

    negatives = sample_hard_negatives(kg1, kg2, ground_truth, blocking, top_k=2)
    print("Hard negatives:")
    for entity1, entity2 in negatives:
        print(f"  {entity1.id} -/- {entity2.id}")

    linker = FeatureClassifierLinker().fit(
        kg1, kg2, ground_truth, blocking=blocking, negatives=negatives, random_state=0
    )
    results = linker.link(kg1, kg2, blocking=blocking)
    predictions = [(r.source_id, r.target_id) for r in results]

    report = Evaluator.evaluate(predictions=predictions, ground_truth=ground_truth)
    print("Metrics:", report.metrics)


if __name__ == "__main__":
    main()

"""Entity Alignment with a feature-based classifier, EntMatcher-style.

Aligns the same two toy knowledge graphs as `string_similarity_ea.py`, but
scores each candidate pair with a classical ML classifier (logistic
regression, by default) trained on several hand-crafted similarity
features rather than a single fixed metric, then resolves the final
one-to-one mapping via a globally optimal assignment
(`EntMatcherLinker`) rather than independent per-source argmax.

`LabelOverlap(ngram_size=1, ...)` is used for blocking here (rather than
the `max_matches=3` used elsewhere) so that `fit()` sees both correct and
incorrect candidate pairs to learn from -- on a dataset this small,
tighter blocking finds only the true matches, leaving nothing to
discriminate against.

This is a pipeline-correctness demo, not a generalization benchmark: the
toy dataset has only 3 ground-truth pairs, far too few to draw conclusions
about how well the classifier generalizes to unseen entities.

Run with: `uv run python examples/feature_classifier_ea.py`
"""

from linkingtk.algorithms.ea import EntMatcherLinker
from linkingtk.blocking import LabelOverlap
from linkingtk.datasets import ToyEADataset
from linkingtk.eval import Evaluator


def main() -> None:
    kg1, kg2, ground_truth = ToyEADataset().load()
    blocking = LabelOverlap(ngram_size=1, max_matches=10)

    linker = EntMatcherLinker().fit(kg1, kg2, ground_truth, blocking=blocking, random_state=0)
    results = linker.link(kg1, kg2, blocking=blocking)

    for result in results:
        print(f"{result.source_id} -> {result.target_id} (score={result.score:.2f})")

    predictions = [(r.source_id, r.target_id) for r in results]
    report = Evaluator.evaluate(predictions=predictions, ground_truth=ground_truth)
    print("Metrics:", report.metrics)


if __name__ == "__main__":
    main()

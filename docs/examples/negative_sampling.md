# Hard-negative sampling for classifier training

[`FeatureClassifierLinker.fit`](feature_classifier_ea.md) needs negative
(non-matching) examples alongside the ground-truth positives, and by
default samples them uniformly at random from the candidate pool. But a
blocking strategy has already decided which non-matches are *plausible*
enough to surface as candidates at all — those are more informative
training signal than a uniform random draw, because they're the pairs a
classifier is most likely to get wrong.
[`sample_hard_negatives`](../reference/blocking.md) mines exactly those:
for each ground-truth source entity, the top-K most confusable
non-matching candidates from
[`LabelOverlap`](../reference/blocking.md) blocking, in the best-first
order the strategy already returns them in.

As in [the feature-classifier example](feature_classifier_ea.md),
`LabelOverlap(ngram_size=1, ...)` is used here rather than a tighter
`max_matches=3`, so that blocking surfaces both correct and incorrect
candidates for `sample_hard_negatives` to mine from — tighter blocking on
a dataset this small finds only the true matches, leaving nothing to mine.

```python
--8<-- "examples/negative_sampling.py"
```

Run with:

```bash
uv run python examples/negative_sampling.py
```

```text
Hard negatives:
  kg1:paris -/- kg2:rome
  kg1:paris -/- kg2:berlin
  kg1:berlin -/- kg2:rome
  kg1:berlin -/- kg2:paris
  kg1:rome -/- kg2:berlin
  kg1:rome -/- kg2:paris
Metrics: {'precision@1': 1.0, 'recall': 1.0, 'f1': 1.0}
```

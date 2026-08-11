# Entity Alignment with a feature-based classifier (EntMatcher-style)

This aligns the same two toy knowledge graphs, but scores each candidate
pair with a classical ML classifier
([`FeatureClassifierLinker`](../reference/algorithms.md), logistic regression
by default) trained on several hand-crafted similarity features — several
string-similarity metrics plus TF-IDF cosine similarity — rather than a
single fixed metric, then resolves the final one-to-one mapping via a
globally optimal assignment (the Hungarian algorithm) rather than
independent per-source argmax.
[`EntMatcherLinker`](../reference/algorithms.md) is a preconfigured instance
of `FeatureClassifierLinker` (`matching=OptimalMatcher()`), named after
[EntMatcher](https://github.com/DexterZeng/EntMatcher) — not a dependency
of this project, but the source of the "a globally optimal assignment can
outperform independent per-source matching" idea reused here. Blocking
here uses `LabelOverlap(ngram_size=1, ...)` rather than the `max_matches=3`
used in [the StringSimilarityLinker example](string_similarity_ea.md), so
that `fit()` sees some incorrect candidate pairs to learn to reject, not
just the true matches.

This is a pipeline-correctness demo, not a generalization benchmark — the
toy dataset has only 3 ground-truth pairs, far too few to demonstrate real
generalization.

```python
--8<-- "examples/feature_classifier_ea.py"
```

Run with:

```bash
uv run python examples/feature_classifier_ea.py
```

```text
kg1:paris -> kg2:paris (score=0.91)
kg1:berlin -> kg2:berlin (score=0.91)
kg1:rome -> kg2:rome (score=0.90)
Metrics: {'precision@1': 1.0, 'recall': 1.0, 'f1': 1.0}
```

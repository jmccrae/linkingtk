# Examples

Runnable versions of everything on this page live under
[`examples/`](https://github.com/jmccrae/linkingtk/tree/main/examples) in the
repository.

## Word Sense Disambiguation with Lesk

This disambiguates the classic ambiguous word *"bank"* between its financial
and riverbank senses. A single mention carries the ambiguous label and a
sentence of context; two candidate senses carry the same label and a gloss.
[`LeskLinker`](reference/algorithms.md) scores each candidate by how many
words its gloss shares with the mention's context, so the "deposited money"
context should pull the link toward the financial-institution sense.

```python
--8<-- "examples/lesk_wsd.py"
```

Run with:

```bash
uv run python examples/lesk_wsd.py
```

```text
m1 -> bank.n.01 (score=1.0)
  alternatives: ['bank.n.02']
Metrics: {'precision@1': 1.0, 'recall': 1.0, 'f1': 1.0}
```

## Entity Alignment with StringSimilarityLinker

This aligns two small knowledge graphs from [`ToyEADataset`](datasets.md#toy-datasets)
describing the same three cities, but `kg2`'s labels append the country name
(`"Paris"` vs. `"Paris, France"`), so an exact label match would miss every
pair. [`LabelOverlap`](reference/blocking.md) blocking (character n-gram
overlap) finds the candidates despite the mismatch, and
[`StringSimilarityLinker`](reference/algorithms.md) picks the best one per
source entity by Jaccard token overlap on the `label` field.
[`LeskLinker`](reference/algorithms.md) above is a preconfigured instance
of this same class (`source_field="context"`, `target_field="description"`,
`metric="word_overlap"`).

```python
--8<-- "examples/string_similarity_ea.py"
```

Run with:

```bash
uv run python examples/string_similarity_ea.py
```

```text
kg1:paris -> kg2:paris (score=0.50)
kg1:berlin -> kg2:berlin (score=0.50)
kg1:rome -> kg2:rome (score=0.50)
Metrics: {'precision@1': 1.0, 'recall': 1.0, 'f1': 1.0}
```

## Entity Alignment with a feature-based classifier (EntMatcher-style)

This aligns the same two toy knowledge graphs, but scores each candidate
pair with a classical ML classifier
([`FeatureClassifierLinker`](reference/algorithms.md), logistic regression
by default) trained on several hand-crafted similarity features — several
string-similarity metrics plus TF-IDF cosine similarity — rather than a
single fixed metric, then resolves the final one-to-one mapping via a
globally optimal assignment (the Hungarian algorithm) rather than
independent per-source argmax.
[`EntMatcherLinker`](reference/algorithms.md) is a preconfigured instance
of `FeatureClassifierLinker` (`matching=OptimalMatcher()`), named after
[EntMatcher](https://github.com/DexterZeng/EntMatcher) — not a dependency
of this project, but the source of the "a globally optimal assignment can
outperform independent per-source matching" idea reused here. Blocking
here uses `LabelOverlap(ngram_size=1, ...)` rather than the `max_matches=3`
used above, so that `fit()` sees some incorrect candidate pairs to learn
to reject, not just the true matches.

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

## Blocking and evaluation with ExactMatch

A minimal end-to-end example showing
[`ExactMatch`](reference/blocking.md) blocking followed by
[`Evaluator.evaluate`](reference/eval.md) against known ground truth, plus
[`Evaluator.evaluate_blocking`](reference/eval.md) to assess the blocking
step itself — independent of any downstream linker — via Pair Completeness
(the fraction of true matches the blocking pass kept) and Reduction Ratio
(the fraction of the full cross-product it eliminated).

```python
--8<-- "examples/basic_exact_match.py"
```

Run with:

```bash
uv run python examples/basic_exact_match.py
```

```text
Candidate pairs: [('s1', 't1')]
Metrics: {'precision@1': 1.0, 'recall': 0.5, 'f1': 0.6666666666666666}
Blocking metrics: {'pair_completeness': 0.5, 'reduction_ratio': 0.75}
```

## MVP milestone acceptance demo

This is the MVP milestone's acceptance demo: the same `StringSimilarityLinker`
baseline (`LeskLinker` is a preconfigured instance of it) run against a
[bundled toy dataset](datasets.md#toy-datasets) for each of the four tasks —
Entity Alignment, Entity Linking, Word Sense Disambiguation and Word Sense
Alignment — each scored with [`Evaluator.evaluate`](reference/eval.md).

```python
--8<-- "examples/mvp_benchmark.py"
```

Run with:

```bash
uv run python examples/mvp_benchmark.py
```

```text
EA: {'precision@1': 1.0, 'recall': 1.0, 'f1': 1.0}
EL: {'precision@1': 1.0, 'recall': 1.0, 'f1': 1.0}
WSD: {'precision@1': 1.0, 'recall': 1.0, 'f1': 1.0}
WSA: {'precision@1': 1.0, 'recall': 1.0, 'f1': 1.0}
```

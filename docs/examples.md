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

This aligns two small knowledge graphs describing the same three cities,
but `kg2`'s labels append the country name (`"Paris"` vs.
`"Paris, France"`), so an exact label match would miss every pair.
[`LabelOverlap`](reference/blocking.md) blocking (character n-gram
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

## Blocking and evaluation with ExactMatch

A minimal end-to-end example showing
[`ExactMatch`](reference/blocking.md) blocking followed by
[`Evaluator.evaluate`](reference/eval.md) against known ground truth.

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
```

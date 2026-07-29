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

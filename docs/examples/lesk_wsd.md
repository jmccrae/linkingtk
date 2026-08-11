# Word Sense Disambiguation with Lesk

This disambiguates the classic ambiguous word *"bank"* between its financial
and riverbank senses. A single mention carries the ambiguous label and a
sentence of context; two candidate senses carry the same label and a gloss.
[`LeskLinker`](../reference/algorithms.md) scores each candidate by how many
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

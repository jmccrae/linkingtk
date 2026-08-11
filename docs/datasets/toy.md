# Toy datasets

`ToyWSDDataset` and `ToyELDataset` are small, hand-curated, and bundled
directly in the code — no network access required. Each picks classically
ambiguous words/mentions with two candidates apiece, so a context-blind
baseline (e.g. always picking the first candidate) would score no better
than chance:

- **`ToyEADataset`** — 3 same-city entities across two knowledge graphs,
  where the second graph's labels append the country name (e.g.
  *"Paris, France"*), so exact label matching misses every pair.
- **`ToyELDataset`** — 6 mentions of *"Paris"*, *"Mercury"* and *"Amazon"*
  against two knowledge-base entries apiece.
- **`ToyWSDDataset`** — 4 mentions of *"bass"* and *"crane"* against their
  two senses apiece (fish/music, machine/bird). The classic *"bank"*
  example is deliberately left out here since it's already the walkthrough
  in [the Lesk example](../examples/lesk_wsd.md).
- **`ToyWSADataset`** — 4 senses of *"mouse"* and *"bat"*, each glossed
  differently across two dictionaries.

```python
--8<-- "examples/toy_datasets.py"
```

Run with:

```bash
uv run python examples/toy_datasets.py
```

```text
WSD: [('m1', 'bass.n.01'), ('m2', 'bass.n.02'), ('m3', 'crane.n.01'), ('m4', 'crane.n.02')]
Metrics: {'precision@1': 1.0, 'recall': 1.0, 'f1': 1.0}
EL: [('m1', 'Q90'), ('m2', 'Q830149'), ('m3', 'Q308'), ('m4', 'Q925'), ('m5', 'Q3733'), ('m6', 'Q3884')]
Metrics: {'precision@1': 1.0, 'recall': 1.0, 'f1': 1.0}
```

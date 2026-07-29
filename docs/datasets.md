# Datasets

`linkingtk.datasets` provides ready-made [`DatasetLoader`](reference/datasets.md)
implementations. Each one's `load()` returns a `(dataset1, dataset2,
ground_truth)` tuple, ready to hand straight to a linker's `link()` method
and then to [`Evaluator.evaluate`](reference/eval.md).

## Toy datasets

`ToyWSDDataset` and `ToyELDataset` are small, hand-curated, and bundled
directly in the code — no network access required. Each picks classically
ambiguous words/mentions with two candidates apiece, so a context-blind
baseline (e.g. always picking the first candidate) would score no better
than chance:

- **`ToyWSDDataset`** — 4 mentions of *"bass"* and *"crane"* against their
  two senses apiece (fish/music, machine/bird). The classic *"bank"*
  example is deliberately left out here since it's already the walkthrough
  in [the Lesk example](examples.md#word-sense-disambiguation-with-lesk).
- **`ToyELDataset`** — 6 mentions of *"Paris"*, *"Mercury"* and *"Amazon"*
  against two knowledge-base entries apiece.

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

## Naisc ontology-matching datasets

`ConferenceDataset` and `AnatomyDataset` load OWL ontology pairs (plus a
reference alignment) from the
[Naisc](https://github.com/insight-centre/naisc/tree/master/datasets)
project's OAEI benchmarks. Unlike the toy datasets above, these fetch
their `left.rdf`/`right.rdf`/`align.rdf` files over the network on every
`load()` call, and require the `graph` optional dependency group (for
`rdflib`) — install with `uv sync --extra graph`. Only `owl:Class`
entities are extracted; object/data properties are dropped.

- **`ConferenceDataset`** — academic-conference ontologies, well under 100
  classes per side. Small enough for quick, exhaustive Entity Alignment /
  Word Sense Alignment testing.
- **`AnatomyDataset`** — mouse-anatomy vs. NCI Thesaurus ontologies.
  Despite being listed alongside `ConferenceDataset` as a "toy" dataset in
  Naisc, this is the full anatomy track (thousands of classes per side) —
  prefer `ConferenceDataset` unless you specifically need a larger
  benchmark.

```python
from linkingtk.datasets import ConferenceDataset

left, right, ground_truth = ConferenceDataset().load()
```

Both accept a `base_url` override (a URL or `file://` path) if you want to
point them at a local copy instead of GitHub, which is how
[`tests/datasets/test_naisc.py`](https://github.com/jmccrae/linkingtk/blob/main/tests/datasets/test_naisc.py)
exercises them against fixtures without a network call.

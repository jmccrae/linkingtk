# Naisc ontology-matching datasets

`ConferenceDataset` and `AnatomyDataset` load OWL ontology pairs (plus a
reference alignment) from the
[Naisc](https://github.com/insight-centre/naisc/tree/master/datasets)
project's OAEI benchmarks. Unlike the toy datasets, these fetch
their `left.rdf`/`right.rdf`/`align.rdf` files over the network (cached
under `~/.cache/linkingtk/downloads/` after the first `load()` call), and
require the `graph` optional dependency group (for `rdflib`) — install
with `uv sync --extra graph`. Only `owl:Class` entities are extracted;
object/data properties are dropped.

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

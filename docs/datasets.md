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

- **`ToyEADataset`** — 3 same-city entities across two knowledge graphs,
  where the second graph's labels append the country name (e.g.
  *"Paris, France"*), so exact label matching misses every pair.
- **`ToyELDataset`** — 6 mentions of *"Paris"*, *"Mercury"* and *"Amazon"*
  against two knowledge-base entries apiece.
- **`ToyWSDDataset`** — 4 mentions of *"bass"* and *"crane"* against their
  two senses apiece (fish/music, machine/bird). The classic *"bank"*
  example is deliberately left out here since it's already the walkthrough
  in [the Lesk example](examples/lesk_wsd.md).
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

## Naisc ontology-matching datasets

`ConferenceDataset` and `AnatomyDataset` load OWL ontology pairs (plus a
reference alignment) from the
[Naisc](https://github.com/insight-centre/naisc/tree/master/datasets)
project's OAEI benchmarks. Unlike the toy datasets above, these fetch
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

## Real-world Entity Alignment datasets

Unlike the datasets above, these are graph-heavy: each
[`GraphDatasetLoader`](reference/datasets.md) exposes a `load_graphs()`
method alongside `load()`, returning the two knowledge graphs' relation
triples (`list[tuple[subject_id, predicate_id, object_id]]`) -- the
training data a KGE-based linker needs, not just entity labels. They fetch
a multi-MB archive on first use and cache it under
`~/.cache/linkingtk/downloads/` (skip this section if you just want a
quick, offline example -- see the toy/Naisc datasets above instead).

- **`DBP15KZhEnDataset`/`DBP15KJaEnDataset`/`DBP15KFrEnDataset`** — the
  classic cross-lingual DBpedia benchmark (~15K entities/side).
- **`EnFr15KDataset`/`EnDe15KDataset`** — OpenEA's multilingual DBpedia
  pairs (~15K entities/side).
- **`DbpediaWikidata15KDataset`/`DbpediaYago15KDataset`** — OpenEA's
  homogeneous DBpedia-Wikidata/DBpedia-YAGO pairs (~15K entities/side).
- **`IcewsWikiDataset`** — a heterogeneous event-KG-to-Wikipedia pair.
- **`WordNetWikidataLanguagesDataset`/`WordNetWikidataLocationsDataset`/`WordNetWikidataOrganismsDataset`/`WordNetWikidataOrganismsHardDataset`**
  — WordNet synsets aligned to Wikidata items, by topic.

```python
from linkingtk.datasets import DBP15KZhEnDataset

dataset = DBP15KZhEnDataset()
zh_entities, en_entities, ground_truth = dataset.load()
zh_graph, en_graph = dataset.load_graphs()
```

DBP15K/OpenEA/ICEWS share a `zip_url`/`cache_dir` constructor override;
WordNet-Wikidata (fetched per-file, no zip) has the same `base_url`/
`cache_dir` shape as `ConferenceDataset`/`AnatomyDataset` above. None of
DESIGN.md's originally-named hosts (Google Drive for DBP15K, Dropbox/
Figshare for OpenEA) are stable, programmatically fetchable URLs, so these
loaders instead point at GitHub-hosted rehosts of the same data in the
same format — see each module's docstring for the exact source and the
tradeoffs that come with it (e.g. only the 15K/V1 OpenEA sizes are
available this way, and ICEWS's `icews_yago` variant isn't implemented).

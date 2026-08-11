# Real-world Entity Alignment datasets

Unlike the toy and Naisc datasets, these are graph-heavy: each
[`GraphDatasetLoader`](../reference/datasets.md) exposes a `load_graphs()`
method alongside `load()`, returning the two knowledge graphs' relation
triples (`list[tuple[subject_id, predicate_id, object_id]]`) — the
training data a KGE-based linker needs, not just entity labels. They fetch
a multi-MB archive on first use and cache it under
`~/.cache/linkingtk/downloads/` (skip this page if you just want a quick,
offline example — see the [toy](toy.md)/[Naisc](naisc.md) datasets instead).

None of DESIGN.md's originally-named hosts (Google Drive for DBP15K,
Dropbox/Figshare for OpenEA) are stable, programmatically fetchable URLs,
so these loaders instead point at GitHub-hosted rehosts of the same data
in the same format — see each module's docstring for the exact source and
the tradeoffs that come with it (e.g. only the 15K/V1 OpenEA sizes are
available this way, and ICEWS's `icews_yago` variant isn't implemented).

## DBP15K

The classic cross-lingual DBpedia benchmark:

- **`DBP15KZhEnDataset`** — Chinese-English pair (~15K entities/side).
- **`DBP15KJaEnDataset`** — Japanese-English pair (~15K entities/side).
- **`DBP15KFrEnDataset`** — French-English pair (~15K entities/side).

```python
from linkingtk.datasets import DBP15KZhEnDataset

dataset = DBP15KZhEnDataset()
zh_entities, en_entities, ground_truth = dataset.load()
zh_graph, en_graph = dataset.load_graphs()
```

## OpenEA

Multilingual and homogeneous DBpedia pairs, sharing DBP15K's `zip_url`/
`cache_dir` constructor override shape and source archive:

- **`EnFr15KDataset`/`EnDe15KDataset`** — multilingual DBpedia pairs
  (~15K entities/side).
- **`DbpediaWikidata15KDataset`/`DbpediaYago15KDataset`** — homogeneous
  DBpedia-Wikidata/DBpedia-YAGO pairs (~15K entities/side).

```python
from linkingtk.datasets import EnFr15KDataset

dataset = EnFr15KDataset()
en_entities, fr_entities, ground_truth = dataset.load()
en_graph, fr_graph = dataset.load_graphs()
```

## ICEWS

A heterogeneous event-KG-to-Wikipedia pair, sharing DBP15K/OpenEA's
`zip_url`/`cache_dir` constructor override shape (though it's a separate
archive):

- **`IcewsWikiDataset`**

```python
from linkingtk.datasets import IcewsWikiDataset

dataset = IcewsWikiDataset()
icews_entities, wiki_entities, ground_truth = dataset.load()
icews_graph, wiki_graph = dataset.load_graphs()
```

`icews_yago` (the other dataset from the same source) isn't implemented —
see `IcewsWikiDataset`'s docstring for why.

## WordNet-Wikidata

WordNet synsets aligned to Wikidata items, by topic. Fetched per-file (no
zip), with the same `base_url`/`cache_dir` constructor override shape as
[`ConferenceDataset`/`AnatomyDataset`](naisc.md):

- **`WordNetWikidataLanguagesDataset`** — language senses/items (the
  smallest subset — a good default).
- **`WordNetWikidataLocationsDataset`** — place-name senses/items.
- **`WordNetWikidataOrganismsDataset`** — species/taxon senses/items,
  keyed on binomial (scientific) names.
- **`WordNetWikidataOrganismsHardDataset`** — a harder variant of
  `WordNetWikidataOrganismsDataset`.

```python
from linkingtk.datasets import WordNetWikidataLanguagesDataset

dataset = WordNetWikidataLanguagesDataset()
wordnet_entities, wikidata_entities, ground_truth = dataset.load()
wordnet_graph, wikidata_graph = dataset.load_graphs()
```

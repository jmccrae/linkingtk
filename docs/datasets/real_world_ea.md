# Entity alignment datasets

Unlike the [toy datasets](toy.md) (which also cover the Naisc
ontology-matching benchmarks), these are graph-heavy: each
[`GraphDatasetLoader`](../reference/datasets.md) exposes a `load_graphs()`
method alongside `load()`, returning the two knowledge graphs' relation
triples (`list[tuple[subject_id, predicate_id, object_id]]`) — the
training data a KGE-based linker needs, not just entity labels. They fetch
a multi-MB archive on first use and cache it under
`~/.cache/linkingtk/downloads/` (skip this page if you just want a quick,
offline example — see the [toy datasets](toy.md) instead).

None of DESIGN.md's originally-named hosts (Google Drive for DBP15K,
Dropbox/Figshare for OpenEA) are stable, programmatically fetchable URLs,
so these loaders instead point at GitHub-hosted rehosts of the same data
in the same format — see each module's docstring for the exact source and
the tradeoffs that come with it (e.g. only the 15K/V1 OpenEA sizes are
available this way, and ICEWS's `icews_yago` variant isn't implemented).

DBP15K, OpenEA and ICEWS also ship a native supervised train/test split of
their ground truth (OpenEA/DBP15K additionally have a validation split;
ICEWS doesn't) — `load_splits()` exposes it as `(train_pairs, test_pairs,
val_pairs)`, separate from `load()`'s single concatenated `ground_truth`.
This is what a real KGE-EA benchmark needs: the train pairs are added to
the training graph as seed alignment triples, and the test pairs are held
out for ranked evaluation — see the [KGE method
benchmarks](../examples/ea_kge_benchmarks.md) example. WordNet-Wikidata has
no native split, so it doesn't implement `load_splits()`.

```python
from linkingtk.datasets import EnFr15KDataset

train_pairs, test_pairs, val_pairs = EnFr15KDataset().load_splits()
```

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

## OpenEA (native format, with attributes)

`EnFr15KDataset` above (and its siblings) come from a rehost with **no
attribute triples at all** -- fine for structural methods (KGELinker,
MTransE, IPTransE), but not for methods that use attribute signal (JAPE,
and later AttrE/IMUSE/MultiKE). `EnFr15KAttrDataset` and its siblings load
OpenEA's own native per-triple format instead (raw URIs, including
`attr_triples_1`/`attr_triples_2`), rehosted on the Hugging Face Hub by the
matchbench project.

**Not a drop-in replacement** for the loaders above: cross-checked while
building JAPE's benchmark (#28) by downloading both zips and comparing
entity URI sets -- only ~10% of `EnFr15KDataset`'s entity URIs coincide
with `EnFr15KAttrDataset`'s. They're two independently-sampled cuts of
"EN-FR-15K" that happen to share the same official split-size ratios
(20/10/70%), not the same entity roster. Use this family specifically when
attribute triples are needed; use the loaders above otherwise (already
used/verified by MTransE and IPTransE).

- **`EnFr15KAttrDataset`/`EnDe15KAttrDataset`** — multilingual DBpedia
  pairs, with attributes.
- **`DbpediaWikidata15KAttrDataset`/`DbpediaYago15KAttrDataset`** —
  homogeneous pairs, with attributes.

```python
from linkingtk.datasets import EnFr15KAttrDataset

dataset = EnFr15KAttrDataset()
en_entities, fr_entities, ground_truth = dataset.load()
en_graph, fr_graph = dataset.load_graphs()
en_attrs, fr_attrs = dataset.load_attribute_triples()
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

`load_graphs()` drops each triple's two trailing timestamp-id columns
(matching every other `_KGZipDataset`, none of which have temporal facts
at all). `load_temporal_graphs()` keeps them instead, resolved to real
`"YYYY-MM"` labels via the archive's own `time_id` file, for
[`SimpleHHEALinker`][linkingtk.algorithms.ea.simple_hhea.SimpleHHEALinker]'s
Time2Vec branch:

```python
icews_temporal, wiki_temporal = dataset.load_temporal_graphs()
# [(subject_id, relation_id, object_id, start_label, end_label), ...]
# start_label/end_label are "YYYY-MM" strings, or None if unresolvable.
```

## WordNet-Wikidata

WordNet synsets aligned to Wikidata items, by topic. Fetched per-file (no
zip), with the same `base_url`/`cache_dir` constructor override shape as
[`ConferenceDataset`/`AnatomyDataset`](toy.md#naisc-ontology-matching-datasets):

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

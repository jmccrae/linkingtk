# Entity Linking datasets

These loaders all follow the same shape as
[`algorithms.el`][linkingtk.algorithms.el]'s convention: `dataset1`
(mentions) carries `context=(text, start, end)` and no description,
`dataset2` (the KB) carries a label and, where the source provides one, a
description. A mention with no resolvable gold entity (NIL) is skipped
entirely rather than kept with an empty target — the same convention used
by every loader on this page.

## AIDA-CoNLL

[`AidaConllDataset`](../reference/datasets.md) links named-entity mentions
in Reuters newswire text (the CoNLL 2003 NER corpus) to Wikidata QIDs, via
the Hugging Face Hub republish `cyanic-selkie/aida-conll-yago-wikidata`
(the original release's licensing is murky; this republish is
cc-by-sa-3.0). KB entities' descriptions are fetched live from Wikipedia's
MediaWiki API and disk-cached. Has a native train/validation/test split:

```python
from linkingtk.datasets import AidaConllDataset

dataset = AidaConllDataset()
mentions, kb, ground_truth = dataset.load()
train_pairs, test_pairs, val_pairs = dataset.load_splits()
```

## Zeshel

[`ZeshelDataset`](../reference/datasets.md) is BLINK's own zero-shot EL
benchmark (Logeswaran et al. 2019): mentions inside a Wikia (fandom)
page's text, linked to *other* Wikia pages, with train/validation/test
domains kept disjoint so the model must resolve entities from description
text alone. Loaded from the Hugging Face Hub republish `naist-nlp/zeshel`.
See [`BlinkLinker`](../reference/algorithms.md). Has a native
train/validation/test split; `load_splits()` skips the ~1.2GB entity
dictionary download entirely:

```python
from linkingtk.datasets import ZeshelDataset

dataset = ZeshelDataset()
mentions, kb, ground_truth = dataset.load()
train_pairs, test_pairs, val_pairs = dataset.load_splits()
```

## DaMuEL

[`DamuelDataset`](../reference/datasets.md) links Wikipedia mentions
across 53 languages to Wikidata QIDs (Kubeša & Straka, 2023), published on
LINDAT/CLARIAH-CZ as one large tar per language (278MB-26.3GB). This
loader streams the tar and stops after `max_parts` of its 500 shuffled
`.xz` shards, so it never materializes a full language in memory:

```python
from linkingtk.datasets import DamuelDataset

dataset = DamuelDataset(language="en", max_parts=2)
mentions, kb, ground_truth = dataset.load()
```

Raise `max_parts` for denser ground truth, especially on lower-resource
languages. DaMuEL publishes no native train/dev/test split, so there's no
`load_splits()` here.

## LCQuAD 2.0

[`Lcquad2Dataset`](../reference/datasets.md) derives entity mentions from
a large-scale KGQA benchmark (~30,000 crowd-sourced questions, each paired
with a SPARQL query over Wikidata) that itself carries no mention-span
annotations: every `wd:Qxxx` referenced in a row's `sparql_wikidata` is a
gold entity, and its text span is recovered by matching the entity's live
Wikidata label against the question text (~81% resolve this way; the rest
are skipped, same NIL-skip convention as the other loaders on this page).
KB labels/descriptions are fetched live via Wikidata's `wbgetentities`
API. Has a native train/test split (no validation split):

```python
from linkingtk.datasets import Lcquad2Dataset

dataset = Lcquad2Dataset()
mentions, kb, ground_truth = dataset.load()
train_pairs, test_pairs, val_pairs = dataset.load_splits()  # val_pairs is always []
```

Unlike the other loaders here, `load_splits()` can't skip the network
label lookup — resolving a mention's span requires each candidate
entity's Wikidata label, not just its id — so it only skips building the
full KB `Entity` list.

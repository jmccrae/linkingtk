# Word Sense Disambiguation datasets

Unlike [`ToyWSDDataset`](toy.md) (two hand-picked candidate senses per
mention), these are real sense-annotated corpora tagged against a full
dictionary — there's no fixed candidate-sense list to enumerate, since in
principle any open-class word can be any synset in WordNet. So `dataset2`
here isn't a materialized `list[Entity]`: it's a
[`WnEntitySource`](../reference/sources.md), a query-driven
[`EntitySource`](../reference/core.md) wrapping the
[`wn`](https://github.com/goodmami/wn) library — `dataset1` (the mentions)
still loads eagerly, but the candidate side is only ever queried per
mention via [`ExactMatch`](../reference/blocking.md) blocking, exactly
like the [`WnEntitySource` WSD example](../examples/wn_wsd.md). Requires
the `wn` optional dependency and a one-time download of whichever lexicon
`dataset2` queries (see each section below for the right one):

```bash
uv pip install linkingtk[wn]
python -m wn download <lexicon>
```

## SemCor

[`SemCorDataset`](../reference/datasets.md) loads
[SemCor 2026](https://github.com/globalwordnet/semcor), a fork and
reannotation of the classic Brown-Corpus SemCor corpus kept aligned with
the current Open English WordNet release. It fetches every one of the
corpus's 352 documents (~46MB of YAML, cached under
`~/.cache/linkingtk/downloads/` after the first `load()` call) — not a
toy subset — and returns one mention per sense-tagged content word, with
`context=(sentence_text, start, end)`. `dataset2` defaults to
``"oewn:2025+"`` (not `WnEntitySource`'s own standalone default of
``"oewn:2021"``), matching the release the corpus's `oewn_key` layer is
currently generated against:

```bash
python -m wn download oewn:2025+
```

```python
from linkingtk.datasets import SemCorDataset

mentions, senses, ground_truth = SemCorDataset().load()
```

Pass `categories=[...]` (e.g. `["press_reportage"]`) to load only some of
the corpus's Brown Corpus genre subdirectories instead of the full thing.

## UFSAC

[`UfsacDataset`](../reference/datasets.md) parses the shared XML schema
behind [UFSAC](https://github.com/getalp/UFSAC) (Vial et al., 2018), which
unifies 16+ WSD corpora (SemCor, WordNet Gloss Tagged, MASC, OMSTI, the
SensEval/SemEval all-words and lexical-sample tasks, ...) under one
format, all tagged with WordNet 3.0 sense keys.

Unlike every other loader on this page, `UfsacDataset` **doesn't fetch its
own data**: UFSAC's entire collection is distributed as a single Google
Drive archive bundling every corpus's XML together (see UFSAC's README),
not a stable, programmatically fetchable per-corpus URL the way this
package's other loaders' sources are. Download and extract the corpus you
want from that archive yourself, then point the loader at the resulting
file (`.xml` or UFSAC's own `.xml.xz` compression, both accepted directly):

```python
from linkingtk.datasets import UfsacDataset

mentions, senses, ground_truth = UfsacDataset("~/ufsac-public-2.1/semcor.xml").load()
```

Since the parser handles UFSAC's one shared schema, this works for any of
the 16+ corpora in the archive, not just `semcor.xml`.

UFSAC's WordNet 3.0 sense keys (e.g. `"group%1:03:00::"`) aren't `wn` ids,
so `ground_truth` here is resolved through
[`sensekey_to_synset_id`](../reference/sources.md) against `dataset2`'s
lexicon (`"omw-en:1.4"` by default — OMW's English WordNet, built directly
from WordNet 3.0) before it's returned; a mention whose sense key doesn't
resolve is dropped entirely, the same NIL-mention convention
[`AidaConllDataset`](../reference/datasets.md) uses for EL.

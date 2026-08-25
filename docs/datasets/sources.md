# Entity Sources

Every loader on the other pages in this section returns a fully
materialized `list[Entity]` via `load()`. That doesn't work for a target
that's too large to enumerate up front, or that already lives in some
external, queryable system (a locally-installed dictionary, a live web
API) — [`EntitySource`](../reference/core.md) is the query-driven
alternative: instead of `load()`ing everything, it exposes `search(query,
top_k)` and `get(entity_id)`, and can be passed directly as `dataset2` to a
[`BlockingStrategy`](../reference/blocking.md) or a
[linker](../reference/algorithms.md)'s `link()` in place of a `list[Entity]`.

`linkingtk.sources` ships three concrete `EntitySource`s, plus a fourth
that wraps *any* of them (or your own) in a local cache. All the examples
below use [`ExactMatch`](../reference/blocking.md) blocking + a
[`StringSimilarityLinker`](../reference/algorithms.md) to disambiguate —
see [the full runnable scripts](../examples/wikipedia_el.md) for complete,
verified output.

## WordNet (`WnEntitySource`)

[`WnEntitySource`](../reference/sources.md) queries a locally-installed
[`wn`](https://github.com/goodmami/wn) lexicon (e.g. Open English WordNet)
without loading every synset into memory first — the query-driven
counterpart to [`ToyWSDDataset`](toy.md) for real WSD/WSA work:

```python
from linkingtk.sources import WnEntitySource

wordnet = WnEntitySource(lexicon="oewn:2021")
candidates = wordnet.search("bass", top_k=10)
sense = wordnet.get("oewn-02001858-n")
```

Requires the `wn` optional dependency (`uv sync --extra wn`) and a
one-time lexicon download (`python -m wn download oewn:2021`); see the
module docstring for details.

## Wikipedia (`WikipediaEntitySource`)

[`WikipediaEntitySource`](../reference/sources.md) queries the public
MediaWiki search API live — no dump download or materialization:

```python
from linkingtk.sources import WikipediaEntitySource

wikipedia = WikipediaEntitySource(lang="en")
hits = wikipedia.search("Paris", top_k=10)
page = wikipedia.get("Paris (mythology)")
```

Wikipedia disambiguates same-named articles via a parenthetical title
suffix (e.g. `"Paris (mythology)"`) rather than a distinct label —
`WikipediaEntitySource` strips that suffix out of the returned `Entity`'s
label and prepends it to the description instead, so `ExactMatch`'s
exact-label blocking can still reach a disambiguated page for a bare
mention. See [the full example](../examples/wikipedia_el.md) for why that
matters and what it looks like end to end. Requires the `wikipedia`
optional dependency (`uv sync --extra wikipedia`).

## Wikidata (`WikidataEntitySource`)

[`WikidataEntitySource`](../reference/sources.md) is the equivalent for
Wikidata's action API — entity ids are QIDs, and a `P31` ("instance of")
claim, when present, is surfaced into `Entity.properties["instance_of"]`.
It supports two backends, selected by whether you pass `vector_index`.

### Without a vector index — live queries

The default: every `search`/`get` call hits `wbsearchentities`/
`wbgetentities` directly.

```python
from linkingtk.sources import WikidataEntitySource

wikidata = WikidataEntitySource(lang="en")
hits = wikidata.search("Paris", top_k=10)   # -> Q90, Q830149, Q162121, ...
item = wikidata.get("Q90")
```

Unlike Wikipedia, Wikidata doesn't fold disambiguation into the label —
every one of those "Paris" hits' label is plainly `"Paris"` — so no
label-stripping is needed here. Fine for a handful of mentions; see [the
full example](../examples/wikidata_el.md). Requires the `wikipedia`
optional dependency (the HTTP client, shared with `WikipediaEntitySource`).

### With a vector index — fully offline

Live per-mention queries don't scale to linking a whole corpus against
Wikidata. [`VectorIndexEntitySource`](../reference/sources.md) is a local
FAISS approximate-nearest-neighbor index built once (from a bulk label
dump, or any `Iterable[Entity]`) and reused with no network calls at all —
generalized from the one-off FAISS index built for
[`wn-wd-entity-align`](https://github.com/jmccrae/wn-wd-entity-align).
Pass it to `WikidataEntitySource` as `vector_index`, and `search`/`get` go
local: `get` falls back to a live call only on an index miss (the index
may be a partial or sampled build).

```python
from pathlib import Path

from sentence_transformers import SentenceTransformer

from linkingtk.core import Entity
from linkingtk.sources import VectorIndexEntitySource, WikidataEntitySource

items = [
    Entity(id="Q90", labels=["Paris"], description="capital of France"),
    Entity(id="Q830149", labels=["Paris"], description="city in Texas, USA"),
]
embedder = SentenceTransformer("all-MiniLM-L6-v2")

VectorIndexEntitySource.build(items, embedder, Path("wikidata_index"))

# Later, or in another process -- no re-embedding, no network:
index = VectorIndexEntitySource.load(Path("wikidata_index"), embedder)
wikidata = WikidataEntitySource(vector_index=index)
wikidata.search("Paris")  # served entirely from the local index
```

`build` takes a pluggable `Embedder` (any object with a batch
`encode(list[str]) -> np.ndarray` method — a real `SentenceTransformer`
satisfies this directly), and by default projects embeddings down via a
truncated SVD to shrink the index. See [the full offline
example](../examples/vector_index_el.md), which also `save()`s/`load()`s
the index to demonstrate the round trip. Requires the `vector-index`
optional dependency (`faiss-cpu` + `sentence-transformers`):
`uv sync --extra vector-index`.

`VectorIndexEntitySource` isn't Wikidata-specific — it's built from a
plain `Iterable[Entity]`, so it works as a local index over any large
target (a big `WnEntitySource` export, a custom corpus), not just Wikidata.

## Caching wrapper

Any `EntitySource` — including your own — can be wrapped in
[`CachingEntitySource`](../reference/core.md) to memoize `search`/`get`
calls, worthwhile for the live HTTP-backed ones above where the same
mention text tends to recur within a run:

```python
from linkingtk.core import CachingEntitySource
from linkingtk.sources import WikipediaEntitySource

wikipedia = CachingEntitySource(WikipediaEntitySource())
```

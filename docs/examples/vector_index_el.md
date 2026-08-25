# Fully-offline Entity Linking via a local vector index

Live per-mention queries against a huge external source (Wikidata,
Wikipedia, ...) don't scale -- [the live Wikidata example](wikidata_el.md)
is fine for a handful of mentions, but not for linking a whole corpus. This
example instead builds a
[`VectorIndexEntitySource`](../reference/sources.md) once, from a small
in-memory batch of entities (standing in for a bulk-downloaded label dump --
the same role
[`wn-wd-entity-align`](https://github.com/jmccrae/wn-wd-entity-align)'s
FAISS index plays over a full Wikidata export), `save()`s it to disk,
`load()`s it back (as a separate process would), and wires the reloaded
index into `WikidataEntitySource(vector_index=...)` -- so `search`/`get`
never touch the network at all.

A real large-scale build follows the exact same `build`/`save`/`load` API,
just over many more entities.

Requires the `faiss` and `sentence-transformers` optional dependencies (the
latter downloads a small model the first time it runs):

```bash
uv pip install linkingtk[faiss,sentence-transformers]
```

```python
--8<-- "examples/vector_index_el.py"
```

Run with:

```bash
uv run python examples/vector_index_el.py
```

```text
m1 -> Q90 (score=5.0)
  alternatives: ['Q830149', 'Q162121', 'Q18331346']
  description: capital and most populous city of France
```

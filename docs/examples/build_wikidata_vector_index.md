# Building a VectorIndexEntitySource from a real Wikidata dump

[`vector_index_el.py`](vector_index_el.md) demonstrates
[`VectorIndexEntitySource`](../reference/sources.md) end to end, but only
against a handful of in-memory entities. This is the operational
counterpart: a real, standalone script for building a real index from
Wikidata's own
[dump](https://www.wikidata.org/wiki/Wikidata:Database_download)
(`latest-all.json.gz`) via
[`WikidataDumpEntities`](../reference/sources.md) -- streamed directly
from a local path or an `http(s)://` URL, no full download required
first.

This can run for hours over the full dump, so it shows a `tqdm` progress
bar by default (`--no-progress` to disable it). With SVD dimensionality
reduction on (the default -- see `--reduced-dim`/`--no-reduce`), it makes
two streaming passes over the dump: one to fit the SVD sample, one to
actually index, so you'll see two progress bars.

Requires the `vector-index` optional dependency:

```bash
uv pip install linkingtk[vector-index]
```

```python
--8<-- "examples/build_wikidata_vector_index.py"
```

Run with (from the repo root):

```bash
# Real, full-scale build (hours):
uv run python examples/build_wikidata_vector_index.py \
    --dump https://dumps.wikimedia.org/wikidatawiki/entities/latest-all.json.gz \
    --out ./wikidata_index

# Quick smoke test against a local dump, capped to 1000 entities:
uv run python examples/build_wikidata_vector_index.py \
    --dump ./my-local-dump.json.gz --out ./wikidata_index_smoke --limit 1000
```

Verified against a small local dump fixture (not the real multi-hundred-GB
one, which isn't practical to fetch here) -- both progress bars render,
`--no-reduce`/`--limit`/`--no-progress` all work, and the built index is
immediately searchable:

```text
Built index at ./wikidata_index_smoke
Sanity-check query 'Paris' -> ['Q830149', 'Q90']
```

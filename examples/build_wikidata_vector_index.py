"""Builds a local VectorIndexEntitySource from a real Wikidata dump.

This is the operational counterpart to `vector_index_el.py`'s small
in-memory demo: a real build, from a real
[Wikidata dump](https://www.wikidata.org/wiki/Wikidata:Database_download)
(``latest-all.json.gz``, several hundred GB uncompressed), via
[`WikidataDumpEntities`](../reference/sources.md) streaming straight from
either a local path or an ``http(s)://`` URL -- no full download required
first. Expect this to run for hours over the full dump; pass `--limit` for
a quick smoke test against a prefix of it instead.

Building with `--reduced-dim` set (the default, matching
[`VectorIndexEntitySource.build`](../reference/sources.md)'s own default)
makes two passes over the dump -- fitting the SVD sample, then indexing --
so you'll see two progress bars, one per pass. Pass `--no-reduce` to index
full-dimensional embeddings in a single pass instead.

Requires the `vector-index` optional dependency:

    uv pip install linkingtk[vector-index]

Run with (from the repo root):
```
# Real, full-scale build (hours):
uv run python examples/build_wikidata_vector_index.py \\
    --dump https://dumps.wikimedia.org/wikidatawiki/entities/latest-all.json.gz \\
    --out ./wikidata_index

# Quick smoke test against a local dump, capped to 1000 entities:
uv run python examples/build_wikidata_vector_index.py \\
    --dump ./my-local-dump.json.gz --out ./wikidata_index_smoke --limit 1000
```
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sentence_transformers import SentenceTransformer

from linkingtk.sources import VectorIndexEntitySource, WikidataDumpEntities


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dump",
        required=True,
        help="Path or http(s):// URL to a Wikidata JSON dump (e.g. latest-all.json.gz).",
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="Directory to write the index bundle to."
    )
    parser.add_argument("--lang", default="en", help="Which label/description language to index.")
    parser.add_argument(
        "--model", default="all-MiniLM-L6-v2", help="sentence-transformers model name."
    )
    parser.add_argument("--reduced-dim", type=int, default=28)
    parser.add_argument(
        "--no-reduce",
        action="store_true",
        help="Skip SVD dimensionality reduction (indexes full-dimensional embeddings, one pass).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100_000,
        help="Entities reservoir-sampled to fit the SVD projection. Ignored with --no-reduce.",
    )
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Stop after this many entities -- for a quick smoke test, not a real build.",
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable the progress bar.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    embedder = SentenceTransformer(args.model)
    entities = WikidataDumpEntities(
        args.dump, lang=args.lang, limit=args.limit, progress=not args.no_progress
    )

    index = VectorIndexEntitySource.build(
        entities,
        embedder,
        args.out,
        reduced_dim=None if args.no_reduce else args.reduced_dim,
        sample_size=args.sample_size,
        batch_size=args.batch_size,
    )
    print(f"Built index at {args.out}")

    results = index.search("Paris", top_k=3)
    print("Sanity-check query 'Paris' ->", [e.id for e in results])


if __name__ == "__main__":
    main()

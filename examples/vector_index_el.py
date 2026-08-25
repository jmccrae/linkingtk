"""Fully-offline Entity Linking via a local vector index.

Live per-mention queries against a huge external source (Wikidata,
Wikipedia, ...) don't scale -- `wikidata_el.py`'s live
[`WikidataEntitySource`](../reference/sources.md) is fine for a handful of
mentions, but not for linking a whole corpus. This example instead builds a
[`VectorIndexEntitySource`](../reference/sources.md) once, from a small
in-memory batch of entities (standing in for a bulk-downloaded label dump --
the same role `wn-wd-entity-align`'s FAISS index plays over a full Wikidata
export, https://github.com/jmccrae/wn-wd-entity-align), `save()`s it to
disk, `load()`s it back (as a separate process would), and wires the
reloaded index into `WikidataEntitySource(vector_index=...)` -- so `search`/
`get` never touch the network at all.

A real large-scale build follows the exact same `build`/`save`/`load` API,
just over many more entities.

Requires the `faiss` and `sentence-transformers` optional dependencies (the
latter downloads a small model the first time it runs):

    uv pip install linkingtk[faiss,sentence-transformers]

Run with: `uv run python examples/vector_index_el.py`
"""

import tempfile
from pathlib import Path

from sentence_transformers import SentenceTransformer

from linkingtk.algorithms.string_similarity import StringSimilarityLinker
from linkingtk.blocking import ExactMatch
from linkingtk.core import Entity
from linkingtk.sources import VectorIndexEntitySource, WikidataEntitySource

_ITEMS = [
    Entity(id="Q90", labels=["Paris"], description="capital and most populous city of France"),
    Entity(id="Q830149", labels=["Paris"], description="city in Lamar County, Texas, USA"),
    Entity(id="Q18331346", labels=["Paris"], description="a family name"),
    Entity(id="Q162121", labels=["Paris"], description="genus of flowering plants"),
]


def main() -> None:
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    with tempfile.TemporaryDirectory() as tmp_dir:
        index_path = Path(tmp_dir) / "wikidata_paris_index"
        VectorIndexEntitySource.build(_ITEMS, embedder, index_path, reduced_dim=8)

        # Reload as a separate process would -- no network, no re-embedding.
        index = VectorIndexEntitySource.load(index_path, embedder)
        items = WikidataEntitySource(vector_index=index)

        mentions = [
            Entity(
                id="m1",
                labels=["Paris"],
                context="capital and most populous city of France",
            ),
        ]
        linker = StringSimilarityLinker(
            source_field="context", target_field="description", metric="word_overlap"
        )
        results = linker.link(mentions, items, blocking=ExactMatch())
        for result in results:
            print(f"{result.source_id} -> {result.target_id} (score={result.score})")
            print(f"  alternatives: {result.alternatives}")

        target = items.get(results[0].target_id)
        assert target is not None
        print(f"  description: {target.description}")


if __name__ == "__main__":
    main()

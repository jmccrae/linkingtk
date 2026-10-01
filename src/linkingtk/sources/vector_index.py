"""A local, FAISS-backed EntitySource -- generalized from a project-specific index.

Querying a huge external source (e.g. Wikidata) live, per mention, doesn't
scale: [`wn-wd-entity-align`](https://github.com/jmccrae/wn-wd-entity-align)
hit exactly this and built a one-off FAISS label-similarity index over a bulk
Wikidata label dump to search entirely offline instead. This module
generalizes that approach into a reusable
[EntitySource][linkingtk.core.source.EntitySource]: build a
[VectorIndexEntitySource][linkingtk.sources.vector_index.VectorIndexEntitySource]
once, from any `Iterable[Entity]` (a Wikidata dump, a large `WnEntitySource`
export, anything), `save()` it, and `search()`/`get()` it later with no
network calls at all -- e.g. as
[WikidataEntitySource][linkingtk.sources.wikidata.WikidataEntitySource]'s
`vector_index` argument.
"""

from __future__ import annotations

import dbm
import itertools
import json
import random
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from linkingtk.core.entity import ContextWithSpan, Entity, LabelWithLang, label_texts
from linkingtk.core.source import EntitySource
from linkingtk.core.text import Field, resolve_field
from linkingtk.exceptions import OptionalDependencyError

if TYPE_CHECKING:
    import faiss
    import numpy.typing as npt


class Embedder(Protocol):
    """Structural type for a batch text encoder, e.g.
    `sentence_transformers.SentenceTransformer`'s own `encode(list[str],
    batch_size=...) -> np.ndarray` (with its default `convert_to_numpy=True`)
    matches this directly -- no wrapper class needed, the same way
    [Vectorizer][linkingtk.blocking.embedding.Vectorizer] matches
    scikit-learn's `fit_transform`/`transform` interface.

    `batch_size` must be accepted (even if a concrete `Embedder` ignores
    it): `VectorIndexEntitySource.build` always passes it explicitly,
    since `SentenceTransformer.encode` otherwise silently re-chunks
    whatever's handed to it into its own default sub-batches of 32 for the
    actual forward passes -- regardless of how many texts the call itself
    received. That single silent default was confirmed (profiling a real
    GPU) to cap throughput at ~7k texts/s no matter the caller's batch
    size, vs. ~39k texts/s once a sane batch size (512+) is actually
    threaded through -- a >5x slowdown, and the real reason a full
    Wikidata-dump build was taking over two days (issue #68), not the low
    CPU utilization that made it look like a parallelism problem.
    """

    def encode(
        self, texts: list[str], /, batch_size: int = 32
    ) -> npt.NDArray[np.floating[Any]]: ...


def _build_or_load_offsets(path: Path) -> npt.NDArray[np.int64]:
    """Byte offset of each line in `path`, cached to `<path>.offsets.npy`.

    Ported from `wn-wd-entity-align`'s `faiss_lemma_query.py`: at index
    sizes where the sidecar file itself is too big to parse fully into
    Python objects, a query only needs to seek to and read the handful of
    lines it actually matched.
    """
    cache_path = path.with_suffix(path.suffix + ".offsets.npy")
    if not (cache_path.exists() and cache_path.stat().st_mtime >= path.stat().st_mtime):
        offsets = []
        pos = 0
        with path.open("rb") as f:
            for line in f:
                offsets.append(pos)
                pos += len(line)
        np.save(cache_path, np.array(offsets, dtype=np.int64))
    return np.asarray(np.load(cache_path, mmap_mode="r"))


def _entity_to_json(entity: Entity) -> str:
    return json.dumps(
        {
            "id": entity.id,
            "labels": [
                list(label) if isinstance(label, tuple) else label for label in entity.labels
            ],
            "description": (
                list(entity.description)
                if isinstance(entity.description, tuple)
                else entity.description
            ),
            "context": (
                list(entity.context) if isinstance(entity.context, tuple) else entity.context
            ),
            "properties": entity.properties,
        }
    )


def _entity_from_json(raw: str) -> Entity:
    data = json.loads(raw)
    description: str | LabelWithLang | None = data["description"]
    if isinstance(description, list):
        description = (description[0], description[1])
    context: str | ContextWithSpan | None = data["context"]
    if isinstance(context, list):
        context = (context[0], context[1], context[2])
    return Entity(
        id=data["id"],
        labels=[tuple(label) if isinstance(label, list) else label for label in data["labels"]],
        description=description,
        context=context,
        properties=data["properties"],
    )


def _row_texts(field: Field) -> Callable[[Entity], list[str]]:
    """Resolve `field` into the texts an entity gets one index row each for.

    ``"label"`` gives one row per distinct non-empty label, so a query
    matching any one label exactly isn't diluted by the entity's other
    labels and aliases (as embedding them space-joined would be). Every
    other field gives exactly one row per entity.
    """
    if field == "label":
        return lambda entity: [text for text in dict.fromkeys(label_texts(entity)) if text]
    extract = resolve_field(field)
    return lambda entity: [extract(entity)]


def _reservoir_sample_texts(
    entities: Iterable[Entity],
    k: int,
    extract: Callable[[Entity], list[str]],
    seed: int = 42,
    scan_limit: int | None = None,
) -> list[str]:
    """Uniformly sample up to `k` texts from a single streamed pass over `entities`.

    Ported from `wn-wd-entity-align`'s `reservoir_sample_labels`: unlike
    just taking the first `k`, this isn't biased toward whatever `entities`
    happens to yield first (e.g. a dump sorted by id). If `scan_limit` is
    set, only the first `scan_limit` entities are read -- the sample is then
    uniform over that prefix, not over all of `entities`.
    """
    rng = random.Random(seed)
    reservoir: list[str] = []
    texts = (text for entity in itertools.islice(entities, scan_limit) for text in extract(entity))
    for i, text in enumerate(texts):
        if i < k:
            reservoir.append(text)
        else:
            j = rng.randint(0, i)
            if j < k:
                reservoir[j] = text
    return reservoir


def _project_and_normalize(
    vectors: npt.NDArray[np.floating[Any]], vh: npt.NDArray[np.floating[Any]] | None
) -> npt.NDArray[np.floating[Any]]:
    if vh is not None:
        vectors = vectors @ vh
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return np.asarray((vectors / norms).astype(np.float32))


class VectorIndexEntitySource(EntitySource):
    """A local FAISS approximate-nearest-neighbor index over embedded entity
    text, queried as an [EntitySource][linkingtk.core.source.EntitySource].

    Build once via [build][linkingtk.sources.vector_index.VectorIndexEntitySource.build]
    (optionally [save][linkingtk.sources.vector_index.VectorIndexEntitySource.save]/
    [load][linkingtk.sources.vector_index.VectorIndexEntitySource.load] to
    reuse across processes); `search`/`get` afterwards touch no network and
    no external service.

    Raises:
        OptionalDependencyError: If `faiss` isn't installed.
    """

    def __init__(
        self,
        index: faiss.Index,
        embedder: Embedder,
        ids_path: Path,
        entities_db_path: Path,
        vh: npt.NDArray[np.floating[Any]] | None,
    ) -> None:
        self._index = index
        self._embedder = embedder
        self._ids_path = ids_path
        self._offsets = _build_or_load_offsets(ids_path)
        self._entities_db_path = entities_db_path
        self._entities_db = dbm.open(str(entities_db_path), "r")
        self._vh = vh

    @classmethod
    def build(
        cls,
        entities: Iterable[Entity],
        embedder: Embedder,
        path: Path,
        field: Field = "label",
        reduced_dim: int | None = 28,
        sample_size: int = 100_000,
        sample_scan_limit: int | None = None,
        batch_size: int = 4096,
        encode_batch_size: int = 32,
    ) -> VectorIndexEntitySource:
        """Build a fresh index over `entities`, persisted under `path`.

        Streams `entities` rather than materializing it, so it scales to a
        source too large to fit in memory at once -- e.g.
        [WikidataDumpEntities][linkingtk.sources.wikidata.WikidataDumpEntities],
        reading directly from a downloaded Wikidata dump.

        Args:
            entities: The entities to index, embedded on the text(s) `field`
                resolves (default: one index row per label).
                Must be re-iterable (e.g. a `list`, or an object like
                `WikidataDumpEntities` whose `__iter__` starts fresh each
                call) when `reduced_dim` is set, since fitting the SVD
                projection and indexing are then two separate passes over
                `entities` -- a single-use iterator/generator only works
                with `reduced_dim=None` (one pass).
            embedder: A batch text encoder (see `Embedder`), e.g. a real
                `sentence_transformers.SentenceTransformer(...)`.
            path: Directory to write the index bundle to (created if
                missing) -- see `save` for its contents.
            field: Which entity text to embed: ``"label"`` (each distinct
                label embedded as its own index row, all mapping back to
                the same entity -- so an exact match on any one label isn't
                diluted by the others), ``"description"``, or
                ``"context"`` (one row per entity). A callable taking an
                ``Entity`` and returning ``str`` may be passed instead, for
                fields not covered above (one row per entity) -- e.g.
                ``lambda e: " ".join(label_texts(e))`` for the old
                space-joined-labels behavior. An entity with no text for
                `field` gets no rows (so is never a search hit), but is
                still stored for `get`.
            reduced_dim: If set (the default, 28, matching
                `wn-wd-entity-align`'s own build), embeddings are projected
                down to this many dimensions via a truncated SVD fit on a
                sample of `entities`' texts, shrinking the index -- pass
                `None` to index full-dimensional embeddings instead.
            sample_size: Max texts sampled (reservoir sampling, so not
                biased toward `entities`' start) to fit the SVD projection.
                Ignored if `reduced_dim` is `None`.
            sample_scan_limit: If set, the SVD-fitting pass stops after
                reading this many entities, reservoir-sampling
                `sample_size` texts from just that prefix rather than from
                all of `entities` -- e.g. to avoid streaming a whole
                Wikidata dump twice when the first few million entities
                already give a representative sample. `None` (the
                default) scans everything. Ignored if `reduced_dim` is
                `None`.
            batch_size: Texts buffered per `embedder.encode` call (and per
                FAISS add).
            encode_batch_size: The `batch_size` passed to `embedder.encode`
                itself -- for `SentenceTransformer`, what sizes each GPU
                forward pass, and so the GPU memory a build holds. The
                default of 32 keeps that to roughly the model's own weights
                (LaBSE: ~1.9 GB reserved at 32 vs. ~6.4 GB at 4096) while
                still embedding thousands of labels a second -- far faster
                than streaming a Wikidata dump over the network supplies
                them, which is the real bottleneck there. Raise it (e.g.
                512) only when embedding is the bottleneck, such as
                building from a local file (issue #68).

        Raises:
            OptionalDependencyError: If `faiss` isn't installed.
            TypeError: If `reduced_dim` is set and `entities` is a
                single-use iterator rather than a re-iterable object.
        """
        try:
            import faiss
        except ImportError as exc:
            raise OptionalDependencyError("VectorIndexEntitySource", "vector-index") from exc

        extract = _row_texts(field)

        vh: npt.NDArray[np.floating[Any]] | None = None
        if reduced_dim is not None:
            if iter(entities) is entities:
                raise TypeError(
                    "entities must be re-iterable when reduced_dim is set -- fitting "
                    "the SVD projection and indexing are two separate passes over "
                    "entities. Pass a list, or an object like WikidataDumpEntities "
                    "whose __iter__ starts fresh each call, not a single-use "
                    "iterator/generator -- or pass reduced_dim=None for a one-shot "
                    "iterator."
                )
            sample = _reservoir_sample_texts(
                entities, sample_size, extract, scan_limit=sample_scan_limit
            )
            if sample:
                sample_vectors = np.asarray(embedder.encode(sample, batch_size=encode_batch_size))
                _, _, vh_full = np.linalg.svd(sample_vectors, full_matrices=False)
                # A too-small/low-rank sample (fewer distinct texts than
                # reduced_dim) can't fit a projection of the requested size --
                # silently slicing past vh_full's actual row count would instead
                # produce a smaller-than-recorded matrix, corrupting save/load.
                effective_dim = min(reduced_dim, vh_full.shape[0])
                vh = vh_full[:effective_dim, :].T.astype(np.float32)
                reduced_dim = effective_dim

        index: faiss.Index | None = None

        path.mkdir(parents=True, exist_ok=True)
        ids_path = path / "ids.txt"
        entities_db_path = path / "entities.db"

        with (
            ids_path.open("w", encoding="utf-8") as ids_file,
            dbm.open(str(entities_db_path), "n") as entities_db,
        ):
            # One entry per index row: an entity contributes one row per text.
            batch_ids: list[str] = []
            batch_texts: list[str] = []

            def flush_batch() -> None:
                nonlocal index
                if not batch_texts:
                    return
                vectors = _project_and_normalize(
                    np.asarray(embedder.encode(batch_texts, batch_size=encode_batch_size)), vh
                )
                if index is None:
                    index = faiss.IndexFlatIP(vectors.shape[1])
                index.add(vectors)
                for entity_id in batch_ids:
                    ids_file.write(f"{entity_id}\n")

            for entity in entities:
                entities_db[entity.id] = _entity_to_json(entity)
                for text in extract(entity):
                    batch_ids.append(entity.id)
                    batch_texts.append(text)
                if len(batch_texts) >= batch_size:
                    flush_batch()
                    batch_ids.clear()
                    batch_texts.clear()
            flush_batch()

        if index is None:
            # No entities given -- still need an index of the right dimensionality.
            probe = _project_and_normalize(np.asarray(embedder.encode([""])), vh)
            index = faiss.IndexFlatIP(probe.shape[1])

        meta = {"reduced_dim": reduced_dim}
        (path / "meta.json").write_text(json.dumps(meta))
        faiss.write_index(index, str(path / "index.faiss"))
        if vh is not None:
            vh.tofile(path / "vh.bin")

        return cls(
            index=index,
            embedder=embedder,
            ids_path=ids_path,
            entities_db_path=entities_db_path,
            vh=vh,
        )

    def save(self, path: Path) -> None:
        """Write this index's bundle to `path` (a directory, created if missing).

        Writes `index.faiss` (the FAISS index itself), `vh.bin` (the SVD
        projection matrix, only if one was used), `ids.txt` (+ a cached
        `ids.txt.offsets.npy`, the entity id of each index row, one per line
        in row order -- an id repeats once per label it was indexed under),
        `entities.db` (a `dbm` store, entity id -> JSON), and `meta.json`.
        """
        try:
            import faiss
        except ImportError as exc:
            raise OptionalDependencyError("VectorIndexEntitySource", "vector-index") from exc

        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path / "index.faiss"))
        if self._vh is not None:
            self._vh.tofile(path / "vh.bin")
        reduced_dim = self._vh.shape[1] if self._vh is not None else None
        (path / "meta.json").write_text(json.dumps({"reduced_dim": reduced_dim}))

        if (path / "ids.txt").resolve() != self._ids_path.resolve():
            ids_text = self._ids_path.read_text(encoding="utf-8")
            (path / "ids.txt").write_text(ids_text, encoding="utf-8")
        if (path / "entities.db").resolve() != self._entities_db_path.resolve():
            with dbm.open(str(path / "entities.db"), "n") as dst:
                for key in self._entities_db.keys():  # noqa: SIM118
                    dst[key] = self._entities_db[key]

    @classmethod
    def load(cls, path: Path, embedder: Embedder) -> VectorIndexEntitySource:
        """Reload an index bundle written by `build`/`save` from `path`.

        `embedder` must be re-supplied (it isn't serialized) and should be
        the same one -- or one producing embeddings in the same space --
        used to build the index.
        """
        try:
            import faiss
        except ImportError as exc:
            raise OptionalDependencyError("VectorIndexEntitySource", "vector-index") from exc

        index = faiss.read_index(str(path / "index.faiss"))
        meta = json.loads((path / "meta.json").read_text())
        vh: npt.NDArray[np.floating[Any]] | None = None
        vh_path = path / "vh.bin"
        if meta.get("reduced_dim") is not None and vh_path.exists():
            vh = np.fromfile(vh_path, dtype=np.float32).reshape(-1, meta["reduced_dim"])

        return cls(
            index=index,
            embedder=embedder,
            ids_path=path / "ids.txt",
            entities_db_path=path / "entities.db",
            vh=vh,
        )

    def _id_at(self, row: int) -> str:
        with self._ids_path.open("rb") as f:
            f.seek(int(self._offsets[row]))
            return f.readline().decode("utf-8").rstrip("\n")

    def search(self, query: str, top_k: int = 10) -> list[Entity]:
        """Nearest entities to `query` by embedding cosine similarity, best match first."""
        return self.search_batch([query], top_k)[0]

    def search_batch(self, queries: list[str], top_k: int = 10) -> list[list[Entity]]:
        """Real batched search: one FAISS query for all of `queries` at once.

        Several index rows can map to the same entity (one per label), so
        rows are deduplicated by entity -- each ranked by its best-matching
        row -- and the search is retried with a doubled row count until
        every query has `top_k` distinct entities or the index is exhausted.
        """
        if not queries:
            return []
        vectors = _project_and_normalize(np.asarray(self._embedder.encode(queries)), self._vh)
        total_rows = int(self._index.ntotal)
        k = top_k
        while True:
            _distances, indices = self._index.search(vectors, min(k, total_rows) or top_k)
            ranked_ids = [self._distinct_ids(row_indices, top_k) for row_indices in indices]
            if k >= total_rows or all(len(ids) >= top_k for ids in ranked_ids):
                break
            k *= 2
        results = []
        for ids in ranked_ids:
            hits = [entity for entity_id in ids if (entity := self.get(entity_id)) is not None]
            results.append(hits)
        return results

    def _distinct_ids(self, row_indices: npt.NDArray[np.int64], limit: int) -> list[str]:
        """The first `limit` distinct entity ids among `row_indices`, in order."""
        ids: dict[str, None] = {}
        for row in row_indices:
            if row < 0:
                continue
            ids.setdefault(self._id_at(int(row)), None)
            if len(ids) >= limit:
                break
        return list(ids)

    def get(self, entity_id: str) -> Entity | None:
        """Look up an indexed entity by id, or `None` if it isn't in the index."""
        raw = self._entities_db.get(entity_id)
        if raw is None:
            return None
        return _entity_from_json(raw.decode("utf-8") if isinstance(raw, bytes) else raw)

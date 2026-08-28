from __future__ import annotations

import pickle
import sys
import types
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from linkingtk.core.entity import Entity
from linkingtk.exceptions import OptionalDependencyError
from linkingtk.sources.vector_index import VectorIndexEntitySource


class _FakeEmbedder:
    """Deterministic bag-of-letters encoder -- no real sentence-transformers needed."""

    def __init__(self, dim: int = 26) -> None:
        self.dim = dim

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for char in text.lower():
                if "a" <= char <= "z":
                    vectors[row, ord(char) - ord("a")] += 1.0
        return vectors


class _RecordingEmbedder(_FakeEmbedder):
    """Records every `batch_size` it was called with.

    Regression coverage for issue #68: `build` must pass its own
    `batch_size` through to `embedder.encode` explicitly -- a
    `SentenceTransformer`-backed embedder otherwise silently re-chunks
    into its own default sub-batches of 32 for the actual forward passes,
    regardless of how many texts one `encode` call receives (confirmed on
    a real GPU to cap throughput over 5x below what a real batch size
    reaches).
    """

    def __init__(self, dim: int = 26) -> None:
        super().__init__(dim)
        self.batch_sizes: list[int] = []

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        self.batch_sizes.append(batch_size)
        return super().encode(texts, batch_size)


class _FakeIndexFlatIP:
    """Brute-force cosine-similarity search, standing in for `faiss.IndexFlatIP`."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.vectors: list[np.ndarray] = []

    def add(self, vectors: np.ndarray) -> None:
        self.vectors.extend(vectors)

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        n_queries = queries.shape[0]
        if not self.vectors:
            return (
                np.zeros((n_queries, k), dtype=np.float32),
                -np.ones((n_queries, k), dtype=np.int64),
            )
        matrix = np.stack(self.vectors)
        scores = queries @ matrix.T
        k_eff = min(k, matrix.shape[0])
        order = np.argsort(-scores, axis=1)[:, :k_eff]
        top_scores = np.take_along_axis(scores, order, axis=1)
        if k_eff < k:
            pad = k - k_eff
            order = np.concatenate([order, -np.ones((n_queries, pad), dtype=np.int64)], axis=1)
            top_scores = np.concatenate(
                [top_scores, np.zeros((n_queries, pad), dtype=np.float32)], axis=1
            )
        return top_scores.astype(np.float32), order.astype(np.int64)


def _fake_write_index(index: _FakeIndexFlatIP, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump({"dim": index.dim, "vectors": index.vectors}, f)


def _fake_read_index(path: str) -> _FakeIndexFlatIP:
    with open(path, "rb") as f:
        data = pickle.load(f)
    index = _FakeIndexFlatIP(data["dim"])
    index.vectors = data["vectors"]
    return index


def _fake_faiss_module() -> types.ModuleType:
    module = types.ModuleType("faiss")
    module.IndexFlatIP = _FakeIndexFlatIP  # type: ignore[attr-defined]
    module.write_index = _fake_write_index  # type: ignore[attr-defined]
    module.read_index = _fake_read_index  # type: ignore[attr-defined]
    return module


@pytest.fixture(autouse=True)
def fake_faiss(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    module = _fake_faiss_module()
    monkeypatch.setitem(sys.modules, "faiss", module)
    return module


_ENTITIES = [
    Entity(id="Q1", labels=["Paris"], description="capital of France"),
    Entity(id="Q2", labels=["Paris (mythology)"], description="Trojan prince"),
    Entity(id="Q3", labels=["London"], description="capital of UK"),
]


class TestBuildAndSearch:
    def test_search_ranks_nearest_first(self, tmp_path: Path) -> None:
        source = VectorIndexEntitySource.build(
            _ENTITIES, _FakeEmbedder(), tmp_path / "idx", reduced_dim=None
        )

        results = source.search("London", top_k=3)

        assert [e.id for e in results][0] == "Q3"

    def test_top_k_limits_results(self, tmp_path: Path) -> None:
        source = VectorIndexEntitySource.build(
            _ENTITIES, _FakeEmbedder(), tmp_path / "idx", reduced_dim=None
        )

        assert len(source.search("Paris", top_k=1)) == 1

    def test_reduced_dim_projects_before_indexing(self, tmp_path: Path) -> None:
        source = VectorIndexEntitySource.build(
            _ENTITIES, _FakeEmbedder(), tmp_path / "idx", reduced_dim=2
        )

        results = source.search("London", top_k=3)

        assert results[0].id == "Q3"
        assert source._vh is not None
        assert source._vh.shape[1] == 2

    def test_reduced_dim_none_skips_projection(self, tmp_path: Path) -> None:
        source = VectorIndexEntitySource.build(
            _ENTITIES, _FakeEmbedder(), tmp_path / "idx", reduced_dim=None
        )

        assert source._vh is None
        assert not (tmp_path / "idx" / "vh.bin").exists()

    def test_reduced_dim_larger_than_sample_rank_is_clamped(self, tmp_path: Path) -> None:
        # Only 3 distinct texts -> SVD rank <= 3, even though reduced_dim asks for 10.
        source = VectorIndexEntitySource.build(
            _ENTITIES, _FakeEmbedder(), tmp_path / "idx", reduced_dim=10
        )

        assert source._vh is not None
        assert source._vh.shape[1] <= 3
        # Round-trips without a shape mismatch.
        reloaded = VectorIndexEntitySource.load(tmp_path / "idx", _FakeEmbedder())
        assert reloaded._vh is not None
        assert reloaded._vh.shape == source._vh.shape

    def test_build_from_empty_entities(self, tmp_path: Path) -> None:
        source = VectorIndexEntitySource.build([], _FakeEmbedder(), tmp_path / "idx")

        assert source.search("anything") == []

    def test_batch_size_smaller_than_entity_count_still_indexes_everything(
        self, tmp_path: Path
    ) -> None:
        # Forces multiple flush_batch() calls inside build() (streaming path).
        source = VectorIndexEntitySource.build(
            _ENTITIES, _FakeEmbedder(), tmp_path / "idx", reduced_dim=None, batch_size=1
        )

        assert {e.id for e in source.search("Paris", top_k=3)} >= {"Q1", "Q2"}
        assert source.get("Q3") is not None

    def test_batch_size_is_forwarded_to_embedder_encode(self, tmp_path: Path) -> None:
        embedder = _RecordingEmbedder()

        VectorIndexEntitySource.build(
            _ENTITIES, embedder, tmp_path / "idx", reduced_dim=2, sample_size=2, batch_size=2
        )

        # One call fitting the SVD sample, one+ flushing entities -- every
        # one of them must carry the caller's own batch_size (2), not
        # whatever the embedder's own encode() defaults to.
        assert embedder.batch_sizes
        assert all(size == 2 for size in embedder.batch_sizes)


class _ReiterableEntities:
    """A re-iterable Entity source (fresh generator each __iter__ call) --
    stands in for `WikidataDumpEntities`, without needing a real file.
    """

    def __init__(self, entities: list[Entity]) -> None:
        self._entities = entities

    def __iter__(self) -> Iterator[Entity]:
        yield from self._entities


class TestReiterableSource:
    def test_build_from_reiterable_object_with_reduced_dim(self, tmp_path: Path) -> None:
        source = VectorIndexEntitySource.build(
            _ReiterableEntities(_ENTITIES), _FakeEmbedder(), tmp_path / "idx", reduced_dim=2
        )

        assert [e.id for e in source.search("London", top_k=1)] == ["Q3"]

    def test_build_from_one_shot_generator_with_reduced_dim_none_works(
        self, tmp_path: Path
    ) -> None:
        source = VectorIndexEntitySource.build(
            (e for e in _ENTITIES), _FakeEmbedder(), tmp_path / "idx", reduced_dim=None
        )

        assert [e.id for e in source.search("London", top_k=1)] == ["Q3"]

    def test_build_from_one_shot_generator_with_reduced_dim_set_raises(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(TypeError, match="re-iterable"):
            VectorIndexEntitySource.build(
                (e for e in _ENTITIES), _FakeEmbedder(), tmp_path / "idx", reduced_dim=2
            )


class TestGet:
    def test_returns_indexed_entity(self, tmp_path: Path) -> None:
        source = VectorIndexEntitySource.build(
            _ENTITIES, _FakeEmbedder(), tmp_path / "idx", reduced_dim=None
        )

        entity = source.get("Q2")

        assert entity is not None
        assert entity.labels == ["Paris (mythology)"]
        assert entity.description == "Trojan prince"

    def test_missing_id_returns_none(self, tmp_path: Path) -> None:
        source = VectorIndexEntitySource.build(
            _ENTITIES, _FakeEmbedder(), tmp_path / "idx", reduced_dim=None
        )

        assert source.get("Q999") is None


class TestSearchBatch:
    def test_batches_multiple_queries(self, tmp_path: Path) -> None:
        source = VectorIndexEntitySource.build(
            _ENTITIES, _FakeEmbedder(), tmp_path / "idx", reduced_dim=None
        )

        results = source.search_batch(["Paris", "London"], top_k=1)

        assert [r[0].id for r in results] == ["Q1", "Q3"]

    def test_empty_queries_returns_empty_list(self, tmp_path: Path) -> None:
        source = VectorIndexEntitySource.build(
            _ENTITIES, _FakeEmbedder(), tmp_path / "idx", reduced_dim=None
        )

        assert source.search_batch([]) == []


class TestSaveLoad:
    def test_round_trips_search_and_get(self, tmp_path: Path) -> None:
        source = VectorIndexEntitySource.build(
            _ENTITIES, _FakeEmbedder(), tmp_path / "built", reduced_dim=None
        )

        source.save(tmp_path / "saved")
        reloaded = VectorIndexEntitySource.load(tmp_path / "saved", _FakeEmbedder())

        assert [e.id for e in reloaded.search("London", top_k=1)] == ["Q3"]
        assert reloaded.get("Q2") is not None
        assert reloaded.get("Q2").description == "Trojan prince"  # type: ignore[union-attr]

    def test_load_writes_reusable_offsets_cache(self, tmp_path: Path) -> None:
        VectorIndexEntitySource.build(
            _ENTITIES, _FakeEmbedder(), tmp_path / "idx", reduced_dim=None
        )

        assert (tmp_path / "idx" / "ids.txt.offsets.npy").exists()

        # Reloading reuses the cache rather than rebuilding it (mtime unchanged).
        cache_path = tmp_path / "idx" / "ids.txt.offsets.npy"
        mtime_before = cache_path.stat().st_mtime
        VectorIndexEntitySource.load(tmp_path / "idx", _FakeEmbedder())
        assert cache_path.stat().st_mtime == mtime_before


class TestOptionalDependency:
    def test_build_without_faiss_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "faiss", None)

        with pytest.raises(OptionalDependencyError):
            VectorIndexEntitySource.build(_ENTITIES, _FakeEmbedder(), tmp_path / "idx")

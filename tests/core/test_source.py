from linkingtk.core.entity import Entity
from linkingtk.core.source import CachingEntitySource, EntitySource


class _FakeSource(EntitySource):
    """In-memory EntitySource backed by a small dict, with call counters."""

    def __init__(self, entities: list[Entity]) -> None:
        self._entities = entities
        self.search_calls = 0
        self.get_calls = 0

    def search(self, query: str, top_k: int = 10) -> list[Entity]:
        self.search_calls += 1
        matches = [entity for entity in self._entities if query in entity.labels]
        return matches[:top_k]

    def get(self, entity_id: str) -> Entity | None:
        self.get_calls += 1
        for entity in self._entities:
            if entity.id == entity_id:
                return entity
        return None


def _source() -> _FakeSource:
    return _FakeSource(
        [
            Entity(id="b1", labels=["cat"]),
            Entity(id="b2", labels=["dog"]),
        ]
    )


def test_search_finds_matching_entity() -> None:
    source = _source()
    assert [e.id for e in source.search("cat")] == ["b1"]


def test_get_returns_none_for_missing_id() -> None:
    assert _source().get("missing") is None


def test_search_batch_defaults_to_looping_over_search() -> None:
    source = _source()
    results = source.search_batch(["cat", "dog", "missing"])
    assert [[e.id for e in r] for r in results] == [["b1"], ["b2"], []]
    assert source.search_calls == 3


class TestCachingEntitySource:
    def test_repeated_search_hits_wrapped_source_once(self) -> None:
        inner = _source()
        cached = CachingEntitySource(inner)

        first = cached.search("cat")
        second = cached.search("cat")

        assert [e.id for e in first] == [e.id for e in second] == ["b1"]
        assert inner.search_calls == 1

    def test_repeated_get_hits_wrapped_source_once(self) -> None:
        inner = _source()
        cached = CachingEntitySource(inner)

        first = cached.get("b1")
        second = cached.get("b1")

        assert first is second
        assert inner.get_calls == 1

    def test_different_top_k_is_cached_separately(self) -> None:
        inner = _FakeSource([Entity(id=f"b{i}", labels=["cat"]) for i in range(5)])
        cached = CachingEntitySource(inner)

        cached.search("cat", top_k=2)
        cached.search("cat", top_k=3)

        assert inner.search_calls == 2

    def test_get_caches_a_missing_id_too(self) -> None:
        inner = _source()
        cached = CachingEntitySource(inner)

        assert cached.get("missing") is None
        assert cached.get("missing") is None
        assert inner.get_calls == 1

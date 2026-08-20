from linkingtk.blocking.exact import ExactMatch
from linkingtk.core.entity import Entity, label_texts
from linkingtk.core.source import EntitySource


class _FakeSource(EntitySource):
    """In-memory EntitySource backed by a small dict, with a call counter."""

    def __init__(self, entities: list[Entity]) -> None:
        self._entities = entities
        self.search_calls = 0

    def search(self, query: str, top_k: int = 10) -> list[Entity]:
        self.search_calls += 1
        matches = [entity for entity in self._entities if query in label_texts(entity)]
        return matches[:top_k]

    def get(self, entity_id: str) -> Entity | None:
        for entity in self._entities:
            if entity.id == entity_id:
                return entity
        return None


def test_exact_match_finds_shared_label() -> None:
    dataset1 = [Entity(id="a1", labels=["cat"]), Entity(id="a2", labels=["dog"])]
    dataset2 = [Entity(id="b1", labels=[("cat", "en")]), Entity(id="b2", labels=["fish"])]

    pairs = ExactMatch().candidate_pairs(dataset1, dataset2)

    assert [(e1.id, e2.id) for e1, e2 in pairs] == [("a1", "b1")]


def test_exact_match_no_candidates() -> None:
    dataset1 = [Entity(id="a1", labels=["cat"])]
    dataset2 = [Entity(id="b1", labels=["dog"])]

    assert ExactMatch().candidate_pairs(dataset1, dataset2) == []


def test_exact_match_deduplicates_pairs() -> None:
    dataset1 = [Entity(id="a1", labels=["cat", "kitty"])]
    dataset2 = [Entity(id="b1", labels=["cat", "kitty"])]

    pairs = ExactMatch().candidate_pairs(dataset1, dataset2)

    assert len(pairs) == 1


def test_exact_match_is_not_ranked() -> None:
    assert ExactMatch.ranked is False


class TestEntitySource:
    def test_finds_shared_label_via_source(self) -> None:
        dataset1 = [Entity(id="a1", labels=["cat"]), Entity(id="a2", labels=["dog"])]
        source = _FakeSource(
            [Entity(id="b1", labels=[("cat", "en")]), Entity(id="b2", labels=["fish"])]
        )

        pairs = ExactMatch().candidate_pairs(dataset1, source)

        assert [(e1.id, e2.id) for e1, e2 in pairs] == [("a1", "b1")]

    def test_drops_non_exact_search_results(self) -> None:
        # search() may return loosely-related candidates (e.g. an HTTP
        # search API); only labels that actually match exactly should
        # survive into the candidate pairs.
        dataset1 = [Entity(id="a1", labels=["cat"])]

        class _LooseSource(EntitySource):
            def search(self, query: str, top_k: int = 10) -> list[Entity]:
                return [Entity(id="b1", labels=["catfish"])]

            def get(self, entity_id: str) -> Entity | None:
                return None

        assert ExactMatch().candidate_pairs(dataset1, _LooseSource()) == []

    def test_underscore_query_matches_space_separated_search_result(self) -> None:
        # A source's search() may itself be underscore/space-insensitive
        # (e.g. `wn`'s omw-en lexicons index phrasal verbs space-separated
        # even though queries commonly use the classic underscore-joined
        # form, so a query for "point_out" finds a "point out" result) --
        # the post-filter shouldn't be stricter than that.
        dataset1 = [Entity(id="a1", labels=["point_out"])]

        class _UnderscoreInsensitiveSource(EntitySource):
            def search(self, query: str, top_k: int = 10) -> list[Entity]:
                return [Entity(id="b1", labels=["point out"])]

            def get(self, entity_id: str) -> Entity | None:
                return None

        pairs = ExactMatch().candidate_pairs(dataset1, _UnderscoreInsensitiveSource())

        assert [(e1.id, e2.id) for e1, e2 in pairs] == [("a1", "b1")]

    def test_deduplicates_pairs_from_source(self) -> None:
        dataset1 = [Entity(id="a1", labels=["cat", "kitty"])]
        source = _FakeSource([Entity(id="b1", labels=["cat", "kitty"])])

        pairs = ExactMatch().candidate_pairs(dataset1, source)

        assert len(pairs) == 1

    def test_no_enumeration_of_the_source(self) -> None:
        dataset1 = [Entity(id="a1", labels=["cat"])]
        source = _FakeSource([Entity(id="b1", labels=["cat"])])

        ExactMatch().candidate_pairs(dataset1, source)

        assert source.search_calls == 1

    def test_top_k_is_forwarded_to_search(self) -> None:
        dataset1 = [Entity(id="a1", labels=["cat"])]
        recorded: list[int] = []

        class _RecordingSource(EntitySource):
            def search(self, query: str, top_k: int = 10) -> list[Entity]:
                recorded.append(top_k)
                return []

            def get(self, entity_id: str) -> Entity | None:
                return None

        ExactMatch(top_k=50).candidate_pairs(dataset1, _RecordingSource())

        assert recorded == [50]

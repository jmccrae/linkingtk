from linkingtk.algorithms.string_similarity import StringSimilarityLinker
from linkingtk.algorithms.wsd import LeskLinker
from linkingtk.blocking.base import BlockingStrategy
from linkingtk.blocking.exact import ExactMatch
from linkingtk.core.entity import Entity
from linkingtk.core.source import EntitySource
from linkingtk.matchers import GreedyMatcher, OptimalMatcher


class _AllPairs(BlockingStrategy):
    """Blocking strategy that lets every pair through, used to isolate metric behavior."""

    def candidate_pairs(
        self, dataset1: list[Entity], dataset2: list[Entity]
    ) -> list[tuple[Entity, Entity]]:
        return [(e1, e2) for e1 in dataset1 for e2 in dataset2]


class _FakeSource(EntitySource):
    """In-memory EntitySource backed by a small dict, no network/library dependency."""

    def __init__(self, entities: list[Entity]) -> None:
        self._entities = entities

    def search(self, query: str, top_k: int = 10) -> list[Entity]:
        matches = [entity for entity in self._entities if query in entity.labels]
        return matches[:top_k]

    def get(self, entity_id: str) -> Entity | None:
        for entity in self._entities:
            if entity.id == entity_id:
                return entity
        return None


class TestEntitySourceEndToEnd:
    def test_links_against_an_entity_source_via_exact_match(self) -> None:
        dataset1 = [Entity(id="e1", labels=["big cat"])]
        source = _FakeSource(
            [
                Entity(id="t1", labels=["big cat"]),
                Entity(id="t2", labels=["small dog"]),
            ]
        )

        linker = StringSimilarityLinker(
            source_field="label", target_field="label", metric="jaccard"
        )
        results = linker.link(dataset1, source, blocking=ExactMatch())

        assert len(results) == 1
        assert results[0].target_id == "t1"
        assert results[0].score == 1.0


class TestFieldsAndMetrics:
    def test_label_jaccard_baseline_for_entity_alignment(self) -> None:
        dataset1 = [Entity(id="e1", labels=["big cat"])]
        dataset2 = [
            Entity(id="t1", labels=["big cat"]),
            Entity(id="t2", labels=["small dog"]),
        ]

        linker = StringSimilarityLinker(
            source_field="label", target_field="label", metric="jaccard"
        )
        results = linker.link(dataset1, dataset2, blocking=ExactMatch())

        assert len(results) == 1
        assert results[0].target_id == "t1"
        assert results[0].score == 1.0

    def test_context_description_word_overlap_for_entity_linking(self) -> None:
        mentions = [Entity(id="m1", labels=["paris"], context="I visited Paris last summer")]
        candidates = [
            Entity(
                id="paris_france",
                labels=["paris"],
                description="capital city of France, visited by millions",
            ),
            Entity(id="paris_texas", labels=["paris"], description="a small town in Texas"),
        ]

        linker = StringSimilarityLinker(
            source_field="context", target_field="description", metric="word_overlap"
        )
        results = linker.link(mentions, candidates)

        assert results[0].target_id == "paris_france"

    def test_description_jaccard_for_word_sense_alignment(self) -> None:
        dataset1 = [Entity(id="s1", labels=["bank"], description="a financial institution")]
        dataset2 = [
            Entity(id="t1", labels=["bank"], description="a financial institution"),
            Entity(id="t2", labels=["bank"], description="the land alongside a river"),
        ]

        linker = StringSimilarityLinker(
            source_field="description", target_field="description", metric="jaccard"
        )
        results = linker.link(dataset1, dataset2)

        assert results[0].target_id == "t1"
        assert results[0].score == 1.0

    def test_levenshtein_favors_near_identical_labels(self) -> None:
        dataset1 = [Entity(id="e1", labels=["colour"])]
        dataset2 = [
            Entity(id="t1", labels=["color"]),
            Entity(id="t2", labels=["banana"]),
        ]

        linker = StringSimilarityLinker(
            source_field="label", target_field="label", metric="levenshtein"
        )
        # ExactMatch wouldn't block "colour" against "color"/"banana" (no
        # shared label), so use a blocking strategy that lets everything
        # through to isolate the metric being tested.
        results = linker.link(dataset1, dataset2, blocking=_AllPairs())

        assert results[0].target_id == "t1"
        assert 0.0 < results[0].score < 1.0

    def test_empty_field_scores_sensibly(self) -> None:
        dataset1 = [Entity(id="e1", labels=["cat"])]
        dataset2 = [Entity(id="t1", labels=["cat"])]

        linker = StringSimilarityLinker(
            source_field="description", target_field="description", metric="jaccard"
        )
        results = linker.link(dataset1, dataset2, blocking=ExactMatch())

        assert results[0].score == 0.0


class TestRanking:
    def test_alternatives_are_ranked_below_best_match(self) -> None:
        dataset1 = [Entity(id="e1", labels=["red apple"])]
        dataset2 = [
            Entity(id="t1", labels=["red apple"]),
            Entity(id="t2", labels=["red car"]),
            Entity(id="t3", labels=["green apple"]),
        ]

        linker = StringSimilarityLinker(
            source_field="label", target_field="label", metric="jaccard"
        )
        results = linker.link(dataset1, dataset2, blocking=_AllPairs())

        assert results[0].target_id == "t1"
        assert set(results[0].alternatives) == {"t2", "t3"}


class TestMatchingStrategy:
    def test_defaults_to_greedy_matcher(self) -> None:
        linker = StringSimilarityLinker()
        assert isinstance(linker.matching, GreedyMatcher)

    def test_accepts_an_optimal_matcher_to_resolve_collisions(self) -> None:
        # e1's and e2's individually-best target is both t1 (word_overlap
        # 5 and 6 respectively), but e2 also has a decent t2 match (4)
        # while e1's t2 match is weak (1) -- the globally optimal
        # assignment (e1->t1, e2->t2, total 9) beats the alternative
        # (e1->t2, e2->t1, total 7), unlike greedy which lets both pick t1.
        dataset1 = [
            Entity(id="e1", labels=["a b c d e g"]),
            Entity(id="e2", labels=["a b c d e f g h i j"]),
        ]
        dataset2 = [
            Entity(id="t1", labels=["a b c d e f"]),
            Entity(id="t2", labels=["g h i j"]),
        ]

        greedy = StringSimilarityLinker(metric="word_overlap", matching=GreedyMatcher())
        greedy_results = greedy.link(dataset1, dataset2, blocking=_AllPairs())
        assert {(r.source_id, r.target_id) for r in greedy_results} == {
            ("e1", "t1"),
            ("e2", "t1"),
        }

        optimal = StringSimilarityLinker(metric="word_overlap", matching=OptimalMatcher())
        optimal_results = optimal.link(dataset1, dataset2, blocking=_AllPairs())
        assert {(r.source_id, r.target_id) for r in optimal_results} == {
            ("e1", "t1"),
            ("e2", "t2"),
        }


class TestLeskIsAStringSimilarityLinker:
    def test_lesk_is_preconfigured_string_similarity_linker(self) -> None:
        lesk = LeskLinker()
        assert isinstance(lesk, StringSimilarityLinker)
        assert lesk.source_field == "context"
        assert lesk.target_field == "description"
        assert lesk.metric == "word_overlap"


class TestCustomFieldsAndMetrics:
    def test_accepts_a_custom_field_extractor(self) -> None:
        dataset1 = [Entity(id="e1", labels=["a"], properties={"code": "xyz"})]
        dataset2 = [Entity(id="t1", labels=["a"], properties={"code": "xyz"})]

        linker = StringSimilarityLinker(
            source_field=lambda e: e.properties.get("code", ""),
            target_field=lambda e: e.properties.get("code", ""),
            metric="jaccard",
        )
        results = linker.link(dataset1, dataset2, blocking=ExactMatch())

        assert results[0].target_id == "t1"
        assert results[0].score == 1.0

    def test_accepts_a_custom_metric(self) -> None:
        dataset1 = [Entity(id="e1", labels=["a"])]
        dataset2 = [Entity(id="t1", labels=["a"])]

        linker = StringSimilarityLinker(metric=lambda t1, t2: 42.0)
        results = linker.link(dataset1, dataset2, blocking=ExactMatch())

        assert results[0].score == 42.0

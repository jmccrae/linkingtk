from linkingtk.algorithms.string_similarity import StringSimilarityLinker
from linkingtk.algorithms.wsd import LeskLinker
from linkingtk.blocking.base import BlockingStrategy
from linkingtk.blocking.exact import ExactMatch
from linkingtk.core.entity import Entity


class _AllPairs(BlockingStrategy):
    """Blocking strategy that lets every pair through, used to isolate metric behavior."""

    def candidate_pairs(
        self, dataset1: list[Entity], dataset2: list[Entity]
    ) -> list[tuple[Entity, Entity]]:
        return [(e1, e2) for e1 in dataset1 for e2 in dataset2]


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

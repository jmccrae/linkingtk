import pytest

from linkingtk.blocking.label_overlap import LabelOverlap
from linkingtk.core.entity import Entity


def _entities(*labels: str) -> list[Entity]:
    return [Entity(id=label, labels=[label]) for label in labels]


class TestNgramMetric:
    def test_finds_similar_labels(self) -> None:
        dataset1 = _entities("colour")
        dataset2 = _entities("color", "banana", "colonel")

        pairs = LabelOverlap(metric="ngram", max_matches=5).candidate_pairs(dataset1, dataset2)

        matched_ids = {e2.id for _, e2 in pairs}
        assert "color" in matched_ids
        assert "banana" not in matched_ids

    def test_max_matches_caps_results_per_source(self) -> None:
        dataset1 = _entities("cataaa")
        dataset2 = _entities("cataaa", "cataab", "cataac", "cataad", "cataae")

        pairs = LabelOverlap(metric="ngram", ngram_size=2, max_matches=2).candidate_pairs(
            dataset1, dataset2
        )

        assert len(pairs) == 2

    def test_lowercase_toggle(self) -> None:
        dataset1 = _entities("COLOR")
        dataset2 = _entities("color")

        case_insensitive = LabelOverlap(metric="ngram", lowercase=True).candidate_pairs(
            dataset1, dataset2
        )
        case_sensitive = LabelOverlap(metric="ngram", lowercase=False).candidate_pairs(
            dataset1, dataset2
        )

        assert len(case_insensitive) == 1
        assert len(case_sensitive) == 0

    def test_threshold_prunes_weak_candidates(self) -> None:
        dataset1 = _entities("color")
        dataset2 = _entities("color", "banana")

        unfiltered = LabelOverlap(metric="ngram", max_matches=5).candidate_pairs(dataset1, dataset2)
        filtered = LabelOverlap(metric="ngram", max_matches=5, threshold=100.0).candidate_pairs(
            dataset1, dataset2
        )

        assert len(unfiltered) >= 1
        assert filtered == []

    def test_rejects_invalid_config(self) -> None:
        with pytest.raises(ValueError):
            LabelOverlap(metric="ngram", max_matches=0)
        with pytest.raises(ValueError):
            LabelOverlap(metric="ngram", ngram_size=0)


class TestLevenshteinMetric:
    def test_finds_similar_labels(self) -> None:
        dataset1 = _entities("colour")
        dataset2 = _entities("color", "banana")

        pairs = LabelOverlap(metric="levenshtein", max_matches=5).candidate_pairs(
            dataset1, dataset2
        )

        matched_ids = {e2.id for _, e2 in pairs}
        assert "color" in matched_ids

    def test_max_matches_caps_results_per_source(self) -> None:
        dataset1 = _entities("cataaa")
        dataset2 = _entities("cataaa", "cataab", "cataac", "cataad", "cataae")

        pairs = LabelOverlap(metric="levenshtein", max_matches=2).candidate_pairs(
            dataset1, dataset2
        )

        assert len(pairs) == 2

    def test_threshold_keeps_only_close_matches(self) -> None:
        dataset1 = _entities("color")
        dataset2 = _entities("color", "banana")

        filtered = LabelOverlap(metric="levenshtein", max_matches=5, threshold=0.1).candidate_pairs(
            dataset1, dataset2
        )

        assert [e2.id for _, e2 in filtered] == ["color"]

    def test_default_queue_max_scales_with_max_matches(self) -> None:
        strategy = LabelOverlap(metric="levenshtein", max_matches=3)
        assert strategy.queue_max == 60

    def test_explicit_queue_max_is_respected(self) -> None:
        strategy = LabelOverlap(metric="levenshtein", max_matches=3, queue_max=7)
        assert strategy.queue_max == 7

import logging

import pytest

from linkingtk.blocking.base import BlockingStrategy
from linkingtk.blocking.label_overlap import LabelOverlap
from linkingtk.blocking.negative_sampling import sample_hard_negatives
from linkingtk.core.entity import Entity
from linkingtk.datasets.toy import ToyEADataset


def _entities(*ids: str) -> list[Entity]:
    return [Entity(id=i, labels=[i]) for i in ids]


class _FixedCandidates(BlockingStrategy):
    """Returns a hand-specified, deliberately unsorted candidate list per source."""

    def __init__(self, candidates_by_source: dict[str, list[str]]) -> None:
        self.candidates_by_source = candidates_by_source

    def candidate_pairs(
        self, dataset1: list[Entity], dataset2: list[Entity]
    ) -> list[tuple[Entity, Entity]]:
        by_id2 = {entity.id: entity for entity in dataset2}
        pairs = []
        for entity1 in dataset1:
            for target_id in self.candidates_by_source.get(entity1.id, []):
                pairs.append((entity1, by_id2[target_id]))
        return pairs


class TestOrderingAndFiltering:
    def test_preserves_blocking_order_without_resorting(self) -> None:
        dataset1 = _entities("A1")
        dataset2 = _entities("T1", "T2", "T3", "T4")
        blocking = _FixedCandidates({"A1": ["T3", "T1", "T2", "T4"]})

        negatives = sample_hard_negatives(
            dataset1, dataset2, ground_truth=[("A1", "T1")], blocking=blocking, top_k=10
        )

        assert [e2.id for _, e2 in negatives] == ["T3", "T2", "T4"]

    def test_true_positive_excluded_even_when_not_last(self) -> None:
        dataset1 = _entities("A1")
        dataset2 = _entities("T1", "T2")
        blocking = _FixedCandidates({"A1": ["T1", "T2"]})

        negatives = sample_hard_negatives(
            dataset1, dataset2, ground_truth=[("A1", "T1")], blocking=blocking, top_k=10
        )

        assert [e2.id for _, e2 in negatives] == ["T2"]

    def test_top_k_caps_negatives_per_source(self) -> None:
        dataset1 = _entities("A1")
        dataset2 = _entities("T1", "T2", "T3", "T4")
        blocking = _FixedCandidates({"A1": ["T3", "T1", "T2", "T4"]})

        negatives = sample_hard_negatives(
            dataset1, dataset2, ground_truth=[("A1", "T1")], blocking=blocking, top_k=2
        )

        assert [e2.id for _, e2 in negatives] == ["T3", "T2"]

    def test_source_with_no_candidates_contributes_nothing(self) -> None:
        dataset1 = _entities("A1", "A2")
        dataset2 = _entities("T1")
        blocking = _FixedCandidates({"A1": ["T1"]})  # A2 absent entirely

        negatives = sample_hard_negatives(
            dataset1,
            dataset2,
            ground_truth=[("A1", "T1"), ("A2", "T1")],
            blocking=blocking,
            top_k=10,
        )

        assert negatives == []

    def test_source_with_multiple_true_targets_mined_once(self) -> None:
        dataset1 = _entities("A1")
        dataset2 = _entities("T1", "T2", "T3", "T4")
        blocking = _FixedCandidates({"A1": ["T1", "T2", "T3", "T4"]})

        negatives = sample_hard_negatives(
            dataset1,
            dataset2,
            ground_truth=[("A1", "T1"), ("A1", "T2")],
            blocking=blocking,
            top_k=10,
        )

        assert [e2.id for _, e2 in negatives] == ["T3", "T4"]


class TestWarnOnEmptyResult:
    def test_warns_when_no_negatives_found_anywhere(self, caplog: pytest.LogCaptureFixture) -> None:
        dataset1 = _entities("A1")
        dataset2 = _entities("T1")
        blocking = _FixedCandidates({"A1": ["T1"]})  # only the true match survives blocking

        with caplog.at_level(logging.WARNING, logger="linkingtk"):
            negatives = sample_hard_negatives(
                dataset1, dataset2, ground_truth=[("A1", "T1")], blocking=blocking, top_k=10
            )

        assert negatives == []
        assert any("zero negative candidates" in record.message for record in caplog.records)


class TestRankedWarning:
    def test_warns_when_blocking_not_marked_ranked(self, caplog: pytest.LogCaptureFixture) -> None:
        dataset1 = _entities("A1")
        dataset2 = _entities("T1", "T2")
        blocking = _FixedCandidates({"A1": ["T1", "T2"]})  # ranked defaults to False

        with caplog.at_level(logging.WARNING, logger="linkingtk"):
            sample_hard_negatives(
                dataset1, dataset2, ground_truth=[("A1", "T1")], blocking=blocking, top_k=10
            )

        assert any(
            "isn't marked as returning best-first" in record.message for record in caplog.records
        )

    def test_no_ranked_warning_for_label_overlap(self, caplog: pytest.LogCaptureFixture) -> None:
        kg1, kg2, ground_truth = ToyEADataset().load()
        blocking = LabelOverlap(ngram_size=1, max_matches=10)

        with caplog.at_level(logging.WARNING, logger="linkingtk"):
            sample_hard_negatives(kg1, kg2, ground_truth, blocking, top_k=2)

        assert not any(
            "isn't marked as returning best-first" in record.message for record in caplog.records
        )


class TestEndToEnd:
    def test_with_real_scored_strategy_on_toy_ea_dataset(self) -> None:
        kg1, kg2, ground_truth = ToyEADataset().load()
        blocking = LabelOverlap(ngram_size=1, max_matches=10)
        candidate_ids = {(e1.id, e2.id) for e1, e2 in blocking.candidate_pairs(kg1, kg2)}
        ground_truth_set = set(ground_truth)

        negatives = sample_hard_negatives(kg1, kg2, ground_truth, blocking, top_k=2)

        assert negatives  # LabelOverlap(ngram_size=1) finds false candidates too
        counts: dict[str, int] = {}
        for entity1, entity2 in negatives:
            assert (entity1.id, entity2.id) in candidate_ids
            assert (entity1.id, entity2.id) not in ground_truth_set
            counts[entity1.id] = counts.get(entity1.id, 0) + 1
        assert all(count <= 2 for count in counts.values())

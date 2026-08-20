import pytest

from linkingtk.algorithms.feature_classifier import DEFAULT_FEATURES, FeatureClassifierLinker
from linkingtk.algorithms.matching import OptimalMatcher
from linkingtk.blocking import LabelOverlap, sample_hard_negatives
from linkingtk.blocking.base import BlockingStrategy
from linkingtk.blocking.exact import ExactMatch
from linkingtk.core.entity import Entity
from linkingtk.core.source import EntitySource
from linkingtk.datasets.toy import ToyEADataset, ToyWSADataset
from linkingtk.eval import Evaluator
from linkingtk.exceptions import LinkingTKError


def _blocking() -> LabelOverlap:
    # ngram_size=1 shares at least one character across every pair on these
    # tiny toy datasets, so candidate_pairs() includes both true matches and
    # false ones -- fit() needs both positive and negative examples.
    return LabelOverlap(ngram_size=1, max_matches=10)


class _AllPairs(BlockingStrategy):
    """Blocking strategy that lets every pair through, used to isolate behavior."""

    def candidate_pairs(
        self, dataset1: list[Entity], dataset2: list[Entity]
    ) -> list[tuple[Entity, Entity]]:
        return [(e1, e2) for e1 in dataset1 for e2 in dataset2]


class TestFitAndLink:
    def test_greedy_matching_on_toy_ea_dataset(self) -> None:
        kg1, kg2, ground_truth = ToyEADataset().load()
        blocking = _blocking()

        linker = FeatureClassifierLinker().fit(
            kg1, kg2, ground_truth, blocking=blocking, random_state=0
        )
        results = linker.link(kg1, kg2, blocking=blocking)
        predictions = [(r.source_id, r.target_id) for r in results]

        report = Evaluator.evaluate(predictions=predictions, ground_truth=ground_truth)
        assert report.metrics["precision@1"] == 1.0

    def test_optimal_matching_on_toy_ea_dataset(self) -> None:
        kg1, kg2, ground_truth = ToyEADataset().load()
        blocking = _blocking()

        linker = FeatureClassifierLinker(matching=OptimalMatcher()).fit(
            kg1, kg2, ground_truth, blocking=blocking, random_state=0
        )
        results = linker.link(kg1, kg2, blocking=blocking)
        predictions = [(r.source_id, r.target_id) for r in results]

        report = Evaluator.evaluate(predictions=predictions, ground_truth=ground_truth)
        assert report.metrics["precision@1"] == 1.0

    def test_works_on_toy_wsa_dataset(self) -> None:
        dict1, dict2, ground_truth = ToyWSADataset().load()
        blocking = _blocking()

        linker = FeatureClassifierLinker().fit(
            dict1, dict2, ground_truth, blocking=blocking, random_state=0
        )
        results = linker.link(dict1, dict2, blocking=blocking)
        predictions = [(r.source_id, r.target_id) for r in results]

        report = Evaluator.evaluate(predictions=predictions, ground_truth=ground_truth)
        assert report.metrics["precision@1"] == 1.0

    def test_fit_returns_self_for_chaining(self) -> None:
        kg1, kg2, ground_truth = ToyEADataset().load()
        linker = FeatureClassifierLinker()

        assert linker.fit(kg1, kg2, ground_truth, blocking=_blocking()) is linker


class _RecordingClassifier:
    """Stub classifier that records the labels it was trained on."""

    def __init__(self) -> None:
        self.fit_y: list[int] | None = None

    def fit(self, X: object, y: list[int]) -> "_RecordingClassifier":
        self.fit_y = list(y)
        return self

    def predict_proba(self, X: list[list[float]]) -> list[list[float]]:
        return [[0.5, 0.5] for _ in X]


class TestExplicitNegatives:
    def test_sample_hard_negatives_integrates_with_fit(self) -> None:
        kg1, kg2, ground_truth = ToyEADataset().load()
        blocking = _blocking()
        negatives = sample_hard_negatives(kg1, kg2, ground_truth, blocking, top_k=2)

        linker = FeatureClassifierLinker().fit(
            kg1, kg2, ground_truth, blocking=blocking, negatives=negatives
        )
        results = linker.link(kg1, kg2, blocking=blocking)
        predictions = [(r.source_id, r.target_id) for r in results]

        report = Evaluator.evaluate(predictions=predictions, ground_truth=ground_truth)
        assert report.metrics["precision@1"] == 1.0

    def test_empty_explicit_negatives_raises(self) -> None:
        kg1, kg2, ground_truth = ToyEADataset().load()
        linker = FeatureClassifierLinker()

        with pytest.raises(LinkingTKError):
            linker.fit(kg1, kg2, ground_truth, blocking=_blocking(), negatives=[])

    def test_negative_ratio_ignored_when_negatives_given(self) -> None:
        kg1, kg2, ground_truth = ToyEADataset().load()
        blocking = _blocking()
        negatives = sample_hard_negatives(kg1, kg2, ground_truth, blocking, top_k=1)
        classifier = _RecordingClassifier()

        FeatureClassifierLinker(classifier=classifier).fit(
            kg1,
            kg2,
            ground_truth,
            blocking=blocking,
            negatives=negatives,
            negative_ratio=100,
        )

        assert classifier.fit_y is not None
        assert classifier.fit_y.count(0) == len(negatives)


class TestGuards:
    def test_link_before_fit_raises(self) -> None:
        kg1, kg2, _ = ToyEADataset().load()
        linker = FeatureClassifierLinker()

        with pytest.raises(LinkingTKError):
            linker.link(kg1, kg2)

    def test_fit_with_no_positives_raises(self) -> None:
        kg1, kg2, _ = ToyEADataset().load()
        linker = FeatureClassifierLinker()

        with pytest.raises(LinkingTKError):
            linker.fit(kg1, kg2, ground_truth=[], blocking=_blocking())

    def test_fit_with_no_negatives_raises(self) -> None:
        kg1, kg2, ground_truth = ToyEADataset().load()
        linker = FeatureClassifierLinker()

        # Default LabelOverlap(max_matches=3) only ever finds each entity's
        # true match on this dataset (no false candidates at all).
        with pytest.raises(LinkingTKError):
            linker.fit(kg1, kg2, ground_truth, blocking=LabelOverlap(max_matches=3))

    def test_link_rejects_entity_source(self) -> None:
        kg1, kg2, ground_truth = ToyEADataset().load()
        linker = FeatureClassifierLinker().fit(
            kg1, kg2, ground_truth, blocking=_blocking(), random_state=0
        )

        class _EmptySource(EntitySource):
            def search(self, query: str, top_k: int = 10) -> list[Entity]:
                return []

            def get(self, entity_id: str) -> Entity | None:
                return None

        # ExactMatch itself supports an EntitySource; the TypeError here
        # must come from FeatureClassifierLinker's own guard.
        with pytest.raises(TypeError, match="EntitySource"):
            linker.link(kg1, _EmptySource(), blocking=ExactMatch())


class TestEmptyVocabularyGuard:
    def test_missing_descriptions_do_not_crash_fit(self) -> None:
        # ToyEADataset entities have no description at all, so the combined
        # label+description text is label-only -- fine -- but a dataset
        # with *no* text on either field at all must not crash TfidfVectorizer.
        dataset1 = [Entity(id="a1", labels=[""]), Entity(id="a2", labels=[""])]
        dataset2 = [Entity(id="b1", labels=[""]), Entity(id="b2", labels=[""])]
        ground_truth = [("a1", "b1")]

        linker = FeatureClassifierLinker()
        linker.fit(dataset1, dataset2, ground_truth, blocking=_AllPairs(), random_state=0)

        assert linker._tfidf_fitted is False


class TestPluggability:
    def test_accepts_custom_features(self) -> None:
        kg1, kg2, ground_truth = ToyEADataset().load()
        custom_features = [DEFAULT_FEATURES[0]]

        linker = FeatureClassifierLinker(features=custom_features).fit(
            kg1, kg2, ground_truth, blocking=_blocking(), random_state=0
        )

        assert linker.features == custom_features

    def test_accepts_a_custom_classifier(self) -> None:
        class _AlwaysPositive:
            def fit(self, X: object, y: object) -> "_AlwaysPositive":
                return self

            def predict_proba(self, X: list[list[float]]) -> list[list[float]]:
                return [[0.0, 1.0] for _ in X]

        kg1, kg2, ground_truth = ToyEADataset().load()
        blocking = _blocking()
        linker = FeatureClassifierLinker(classifier=_AlwaysPositive()).fit(
            kg1, kg2, ground_truth, blocking=blocking, random_state=0
        )
        results = linker.link(kg1, kg2, blocking=blocking)

        assert all(r.score == 1.0 for r in results)

"""Feature-based classifier linker for Entity Alignment and Word Sense Alignment.

Scores each blocked candidate pair with a classical ML classifier trained
on hand-crafted similarity features, rather than a single fixed metric —
in the style of EntMatcher (https://github.com/DexterZeng/EntMatcher, see
DESIGN.md's Entity Alignment references). EntMatcher itself is a research
repo, not a dependency of this project; the idea reused here is that a
*globally optimal* one-to-one assignment
([OptimalMatcher][linkingtk.algorithms.matching.OptimalMatcher]) can outperform
independent per-source argmax matching, regardless of what produces the
underlying similarity score.
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from collections.abc import Callable
from typing import Any, Protocol

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from linkingtk.algorithms.base import DEFAULT_BLOCKING, BaseLinker
from linkingtk.algorithms.matching import DEFAULT_MATCHER, Matcher
from linkingtk.algorithms.string_similarity import jaccard, levenshtein_similarity, word_overlap
from linkingtk.blocking.base import BlockingStrategy
from linkingtk.blocking.embedding import Vectorizer
from linkingtk.core.entity import Entity, description_text
from linkingtk.core.result import AlignmentResult
from linkingtk.core.source import EntitySource
from linkingtk.core.text import Field, resolve_field
from linkingtk.exceptions import LinkingTKError
from linkingtk.utils.graph import Graph

logger = logging.getLogger("linkingtk")

FeatureFn = Callable[[Entity, Entity], float]
TfidfVectors = dict[str, Any]

_label_text = resolve_field("label")


class Classifier(Protocol):
    """Structural type for a fit/predict_proba classifier (e.g. scikit-learn's)."""

    def fit(self, X: Any, y: Any) -> Any: ...
    def predict_proba(self, X: Any) -> Any: ...


def _metric_feature(field: Field, metric: Callable[[str, str], float]) -> FeatureFn:
    extract = resolve_field(field)

    def feature(entity1: Entity, entity2: Entity) -> float:
        return metric(extract(entity1), extract(entity2))

    return feature


def _length_ratio(text1: str, text2: str) -> float:
    longer = max(len(text1), len(text2))
    return min(len(text1), len(text2)) / longer if longer else 1.0


def _combined_text(entity: Entity) -> str:
    return _label_text(entity) + " " + description_text(entity)


DEFAULT_FEATURES: list[tuple[str, FeatureFn]] = [
    ("label_word_overlap", _metric_feature("label", word_overlap)),
    ("label_jaccard", _metric_feature("label", jaccard)),
    ("label_levenshtein", _metric_feature("label", levenshtein_similarity)),
    ("label_length_ratio", _metric_feature("label", _length_ratio)),
    ("description_word_overlap", _metric_feature("description", word_overlap)),
    ("description_jaccard", _metric_feature("description", jaccard)),
    ("description_levenshtein", _metric_feature("description", levenshtein_similarity)),
]


class FeatureClassifierLinker(BaseLinker):
    """Scores candidate pairs with a classical ML classifier over hand-crafted features.

    Unlike [StringSimilarityLinker][linkingtk.algorithms.string_similarity.StringSimilarityLinker]
    (a single fixed metric), this combines several similarity signals —
    ``DEFAULT_FEATURES`` (string-overlap/edit-distance metrics on the
    ``label`` and ``description`` fields, reused from
    [linkingtk.algorithms.string_similarity][]) plus a TF-IDF cosine
    similarity feature over combined label+description text — into a
    feature vector scored by a trainable ``classifier``. Must be
    [fit][linkingtk.algorithms.feature_classifier.FeatureClassifierLinker.fit]
    before [link][linkingtk.algorithms.base.BaseLinker.link] can be called.

    Args:
        features: ``(name, fn)`` pairs, each ``fn`` scoring one
            ``(entity1, entity2)`` pair. Defaults to ``DEFAULT_FEATURES``.
        vectorizer: Any object exposing scikit-learn's
            ``fit_transform``/``transform`` interface, used for the TF-IDF
            cosine similarity feature. Defaults to
            ``TfidfVectorizer(stop_words="english")``. Fit once in
            [fit][linkingtk.algorithms.feature_classifier.FeatureClassifierLinker.fit],
            over both datasets' combined label+description
            text, and reused via ``transform`` in ``link`` — unlike
            [EmbeddingSimilarityBlocker][linkingtk.blocking.embedding.EmbeddingSimilarityBlocker],
            which has no explicit fit step and refits per call.
        classifier: Any object exposing scikit-learn's ``fit``/
            ``predict_proba`` interface, expected to return probabilities
            in ``[0, 1]`` for its positive class. Defaults to
            ``StandardScaler`` + ``LogisticRegression`` in a
            ``sklearn.pipeline.make_pipeline``.
        matching: Strategy used to resolve scored candidates into final
            links. Defaults to
            [GreedyMatcher][linkingtk.algorithms.matching.GreedyMatcher] (each source
            entity's highest-scoring candidate, independently, like
            ``StringSimilarityLinker`` — multiple source entities may map
            to the same target). Pass
            [OptimalMatcher][linkingtk.algorithms.matching.OptimalMatcher] for a globally
            optimal one-to-one assignment instead, which can outperform
            greedy matching when two source entities' individually-best
            candidate is the same target. See
            [EntMatcherLinker][linkingtk.algorithms.ea.entmatcher.EntMatcherLinker]
            for a preconfigured instance using it.
    """

    def __init__(
        self,
        features: list[tuple[str, FeatureFn]] | None = None,
        vectorizer: Vectorizer | None = None,
        classifier: Classifier | None = None,
        matching: Matcher = DEFAULT_MATCHER,
    ) -> None:
        self.features = features if features is not None else DEFAULT_FEATURES
        self.vectorizer: Vectorizer = (
            vectorizer if vectorizer is not None else TfidfVectorizer(stop_words="english")
        )
        self.classifier: Classifier = (
            classifier
            if classifier is not None
            else make_pipeline(StandardScaler(), LogisticRegression())
        )
        self.matching = matching
        self._tfidf_fitted = False
        self._fitted = False

    def fit(
        self,
        dataset1: list[Entity],
        dataset2: list[Entity],
        ground_truth: list[tuple[str, str]],
        blocking: BlockingStrategy = DEFAULT_BLOCKING,
        negatives: list[tuple[Entity, Entity]] | None = None,
        negative_ratio: int = 1,
        random_state: int | None = None,
    ) -> FeatureClassifierLinker:
        """Train the classifier on candidate pairs from ``blocking``.

        Positive examples are the candidate pairs found in
        ``ground_truth``. Negative examples default to a simple,
        self-contained strategy: the remaining candidate pairs, sampled
        uniformly at random (up to ``negative_ratio`` times the number of
        positives). Pass ``negatives`` explicitly — e.g. from
        [sample_hard_negatives][linkingtk.blocking.negative_sampling.sample_hard_negatives]
        — for more deliberate negative mining instead; when given,
        ``negative_ratio`` and ``random_state`` are ignored.

        Args:
            dataset1: Source entities.
            dataset2: Target entities.
            ground_truth: List of ``(source_id, target_id)`` true pairs.
            blocking: Strategy used to generate training candidate pairs.
            negatives: Explicit negative ``(entity1, entity2)`` pairs to
                train on, overriding the default random-sampling strategy.
            negative_ratio: Maximum number of negative examples sampled
                per positive example. Ignored if ``negatives`` is given.
            random_state: Seed for negative-example sampling. Ignored if
                ``negatives`` is given.

        Returns:
            ``self``, for chaining.

        Raises:
            LinkingTKError: If no ``ground_truth`` pair survives
                ``blocking`` (no positive examples to learn from), or if
                ``negatives`` is empty (given explicitly, or after
                sampling) — a classifier can't train on a single class.
        """
        candidates = list(blocking.candidate_pairs(dataset1, dataset2))
        ground_truth_set = set(ground_truth)
        positives = [(e1, e2) for e1, e2 in candidates if (e1.id, e2.id) in ground_truth_set]
        if not positives:
            raise LinkingTKError(
                "No ground-truth pairs survived blocking; fit() has nothing to learn "
                "from. Check that `blocking` actually finds the ground-truth pairs "
                "(see Evaluator.evaluate_blocking) before training a classifier on top "
                "of it."
            )

        if negatives is None:
            negatives = self._sample_random_negatives(
                candidates, ground_truth_set, len(positives), negative_ratio, random_state
            )
        elif not negatives:
            raise LinkingTKError(
                "`negatives` is empty; fit() cannot train a classifier on a single class."
            )

        training_pairs = positives + negatives
        labels = [1] * len(positives) + [0] * len(negatives)

        tfidf1, tfidf2 = self._fit_tfidf(dataset1, dataset2)
        X = self._feature_matrix(training_pairs, tfidf1, tfidf2)
        self.classifier.fit(X, labels)
        self._fitted = True
        return self

    def link(
        self,
        dataset1: list[Entity],
        dataset2: list[Entity] | EntitySource,
        graph: Graph = None,
        blocking: BlockingStrategy = DEFAULT_BLOCKING,
    ) -> list[AlignmentResult]:
        if not self._fitted:
            raise LinkingTKError("FeatureClassifierLinker.link() called before fit().")
        if isinstance(dataset2, EntitySource):
            raise TypeError(
                "FeatureClassifierLinker requires a fully materialized list[Entity] for "
                "dataset2 -- it fits/transforms a TF-IDF vectorizer over the whole target "
                "set, which an EntitySource exists specifically to avoid. Not supported yet."
            )

        pairs = list(blocking.candidate_pairs(dataset1, dataset2))
        if not pairs:
            return []

        tfidf1, tfidf2 = self._tfidf_vectors(dataset1, dataset2)
        X = self._feature_matrix(pairs, tfidf1, tfidf2)
        # Indexed row-by-row rather than with numpy-style `[:, 1]` slicing,
        # so a duck-typed classifier (per the Classifier protocol) can
        # return plain nested lists instead of an ndarray.
        scores = [row[1] for row in self.classifier.predict_proba(X)]

        candidates_by_source: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for (entity1, entity2), score in zip(pairs, scores, strict=True):
            candidates_by_source[entity1.id].append((entity2.id, float(score)))

        return self.matching.match(candidates_by_source)

    @staticmethod
    def _sample_random_negatives(
        candidates: list[tuple[Entity, Entity]],
        ground_truth_set: set[tuple[str, str]],
        num_positives: int,
        negative_ratio: int,
        random_state: int | None,
    ) -> list[tuple[Entity, Entity]]:
        negatives_pool = [
            (e1, e2) for e1, e2 in candidates if (e1.id, e2.id) not in ground_truth_set
        ]
        if not negatives_pool:
            raise LinkingTKError(
                "No negative pairs survived blocking; fit() cannot train a classifier "
                "on a single class. `blocking` is only returning true-positive pairs "
                "here — try a broader blocking strategy (e.g. a larger `top_k`/"
                "`max_matches`, or a looser `threshold`) so it also returns some "
                "incorrect candidates to learn from, or pass `negatives` explicitly."
            )
        sample_size = min(len(negatives_pool), negative_ratio * num_positives)
        return random.Random(random_state).sample(negatives_pool, sample_size)

    def _fit_tfidf(
        self, dataset1: list[Entity], dataset2: list[Entity]
    ) -> tuple[TfidfVectors, TfidfVectors]:
        """Fit ``self.vectorizer`` once on both datasets and return its output, split by dataset.

        Reuses the single ``fit_transform`` matrix instead of a separate
        ``transform`` call per dataset, unlike ``_tfidf_vectors``
        (used post-fit, in ``link``) which has no such matrix to reuse.
        """
        texts1 = [_combined_text(entity) for entity in dataset1]
        texts2 = [_combined_text(entity) for entity in dataset2]
        try:
            matrix = self.vectorizer.fit_transform(texts1 + texts2)
        except ValueError:
            logger.warning(
                "Skipping TF-IDF cosine feature: vectorizer found an empty vocabulary "
                "in dataset1/dataset2's label+description text."
            )
            return {}, {}
        self._tfidf_fitted = True
        tfidf1 = {entity.id: matrix[i] for i, entity in enumerate(dataset1)}
        tfidf2 = {entity.id: matrix[len(dataset1) + i] for i, entity in enumerate(dataset2)}
        return tfidf1, tfidf2

    def _tfidf_vectors(
        self, dataset1: list[Entity], dataset2: list[Entity]
    ) -> tuple[TfidfVectors, TfidfVectors]:
        if not self._tfidf_fitted:
            return {}, {}
        matrix1 = self.vectorizer.transform([_combined_text(entity) for entity in dataset1])
        matrix2 = self.vectorizer.transform([_combined_text(entity) for entity in dataset2])
        tfidf1 = {entity.id: matrix1[i] for i, entity in enumerate(dataset1)}
        tfidf2 = {entity.id: matrix2[i] for i, entity in enumerate(dataset2)}
        return tfidf1, tfidf2

    def _feature_matrix(
        self,
        pairs: list[tuple[Entity, Entity]],
        tfidf1: TfidfVectors,
        tfidf2: TfidfVectors,
    ) -> list[list[float]]:
        rows = []
        for entity1, entity2 in pairs:
            row = [fn(entity1, entity2) for _, fn in self.features]
            if self._tfidf_fitted:
                row.append(float(cosine_similarity(tfidf1[entity1.id], tfidf2[entity2.id])[0, 0]))
            rows.append(row)
        return rows

"""Feature-based classifier linker for Entity Alignment and Word Sense Alignment.

Scores each blocked candidate pair with a classical ML classifier trained
on hand-crafted similarity features, rather than a single fixed metric —
in the style of EntMatcher (https://github.com/DexterZeng/EntMatcher, see
DESIGN.md's Entity Alignment references). EntMatcher itself is a research
repo, not a dependency of this project; the idea reused here is that a
*globally optimal* one-to-one assignment (``matching="optimal"``) can
outperform independent per-source argmax matching, regardless of what
produces the underlying similarity score.
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from collections.abc import Callable
from typing import Any, Literal, Protocol

from scipy.optimize import linear_sum_assignment
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from linkingtk.algorithms.base import (
    DEFAULT_BLOCKING,
    BaseLinker,
    Graph,
    greedy_matches,
    rank_candidates,
)
from linkingtk.algorithms.string_similarity import jaccard, levenshtein_similarity, word_overlap
from linkingtk.blocking.base import BlockingStrategy
from linkingtk.blocking.embedding import Vectorizer
from linkingtk.core.entity import Entity, description_text
from linkingtk.core.result import AlignmentResult
from linkingtk.core.text import Field, resolve_field
from linkingtk.exceptions import LinkingTKError

logger = logging.getLogger("linkingtk")

FeatureFn = Callable[[Entity, Entity], float]
TfidfVectors = dict[str, Any]

# Cosine similarity is always in [0, 1] for predict_proba outputs, so any
# cost (1 - score) above 1.0 is guaranteed worse than every real candidate.
_SENTINEL_COST = 2.0

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

    Unlike :class:`~linkingtk.algorithms.string_similarity.StringSimilarityLinker`
    (a single fixed metric), this combines several similarity signals —
    :data:`DEFAULT_FEATURES` (string-overlap/edit-distance metrics on the
    ``label`` and ``description`` fields, reused from
    :mod:`linkingtk.algorithms.string_similarity`) plus a TF-IDF cosine
    similarity feature over combined label+description text — into a
    feature vector scored by a trainable ``classifier``. Must be
    :meth:`fit` before :meth:`link` can be called.

    Args:
        features: ``(name, fn)`` pairs, each ``fn`` scoring one
            ``(entity1, entity2)`` pair. Defaults to :data:`DEFAULT_FEATURES`.
        vectorizer: Any object exposing scikit-learn's
            ``fit_transform``/``transform`` interface, used for the TF-IDF
            cosine similarity feature. Defaults to
            ``TfidfVectorizer(stop_words="english")``. Fit once in
            :meth:`fit`, over both datasets' combined label+description
            text, and reused via ``transform`` in :meth:`link` — unlike
            :class:`~linkingtk.blocking.embedding.EmbeddingSimilarityBlocker`,
            which has no explicit fit step and refits per call.
        classifier: Any object exposing scikit-learn's ``fit``/
            ``predict_proba`` interface, expected to return probabilities
            in ``[0, 1]`` for its positive class. Defaults to
            ``StandardScaler`` + ``LogisticRegression`` in a
            :func:`sklearn.pipeline.make_pipeline`.
        matching: ``"greedy"`` (default) picks each source entity's
            highest-scoring candidate independently, like
            ``StringSimilarityLinker`` — multiple source entities may map
            to the same target. ``"optimal"`` instead finds a single
            globally optimal one-to-one assignment via
            :func:`scipy.optimize.linear_sum_assignment` (the Hungarian
            algorithm), which can outperform greedy matching when two
            source entities' individually-best candidate is the same
            target. See :class:`~linkingtk.algorithms.ea.entmatcher.EntMatcherLinker`
            for a preconfigured instance using ``"optimal"``.
    """

    def __init__(
        self,
        features: list[tuple[str, FeatureFn]] | None = None,
        vectorizer: Vectorizer | None = None,
        classifier: Classifier | None = None,
        matching: Literal["greedy", "optimal"] = "greedy",
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
        negative_ratio: int = 1,
        random_state: int | None = None,
    ) -> FeatureClassifierLinker:
        """Train the classifier on candidate pairs from ``blocking``.

        Positive examples are the candidate pairs found in
        ``ground_truth``; negative examples are sampled from the
        remaining candidate pairs (up to ``negative_ratio`` times the
        number of positives). This is a simple, self-contained sampling
        strategy — see the ``linkingtk.blocking`` hard-negative-sampling
        utility (once available) for more deliberate negative mining.

        Args:
            dataset1: Source entities.
            dataset2: Target entities.
            ground_truth: List of ``(source_id, target_id)`` true pairs.
            blocking: Strategy used to generate training candidate pairs.
            negative_ratio: Maximum number of negative examples sampled
                per positive example.
            random_state: Seed for negative-example sampling.

        Returns:
            ``self``, for chaining.

        Raises:
            LinkingTKError: If no ``ground_truth`` pair survives
                ``blocking`` — there would be no positive examples to
                learn from.
        """
        candidates = blocking.candidate_pairs(dataset1, dataset2)
        ground_truth_set = set(ground_truth)
        positives = [(e1, e2) for e1, e2 in candidates if (e1.id, e2.id) in ground_truth_set]
        if not positives:
            raise LinkingTKError(
                "No ground-truth pairs survived blocking; fit() has nothing to learn "
                "from. Check that `blocking` actually finds the ground-truth pairs "
                "(see Evaluator.evaluate_blocking) before training a classifier on top "
                "of it."
            )
        negatives_pool = [
            (e1, e2) for e1, e2 in candidates if (e1.id, e2.id) not in ground_truth_set
        ]
        if not negatives_pool:
            raise LinkingTKError(
                "No negative pairs survived blocking; fit() cannot train a classifier "
                "on a single class. `blocking` is only returning true-positive pairs "
                "here — try a broader blocking strategy (e.g. a larger `top_k`/"
                "`max_matches`, or a looser `threshold`) so it also returns some "
                "incorrect candidates to learn from."
            )
        sample_size = min(len(negatives_pool), negative_ratio * len(positives))
        negatives = random.Random(random_state).sample(negatives_pool, sample_size)

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
        dataset2: list[Entity],
        graph: Graph = None,
        blocking: BlockingStrategy = DEFAULT_BLOCKING,
    ) -> list[AlignmentResult]:
        if not self._fitted:
            raise LinkingTKError("FeatureClassifierLinker.link() called before fit().")

        pairs = blocking.candidate_pairs(dataset1, dataset2)
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

        if self.matching == "greedy":
            return greedy_matches(candidates_by_source)
        return self._optimal_matches(candidates_by_source)

    def _fit_tfidf(
        self, dataset1: list[Entity], dataset2: list[Entity]
    ) -> tuple[TfidfVectors, TfidfVectors]:
        """Fit ``self.vectorizer`` once on both datasets and return its output, split by dataset.

        Reuses the single ``fit_transform`` matrix instead of a separate
        ``transform`` call per dataset, unlike :meth:`_tfidf_vectors`
        (used post-fit, in :meth:`link`) which has no such matrix to reuse.
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

    @staticmethod
    def _optimal_matches(
        candidates_by_source: dict[str, list[tuple[str, float]]],
    ) -> list[AlignmentResult]:
        source_ids = list(candidates_by_source.keys())
        if not source_ids:
            return []

        scores_by_pair: dict[tuple[str, str], float] = {
            (source_id, target_id): score
            for source_id, candidates in candidates_by_source.items()
            for target_id, score in candidates
        }
        target_ids = sorted({target_id for _, target_id in scores_by_pair})

        # A dense cost matrix (rather than a sparse min-weight bipartite
        # matching) is the right tradeoff at this class's intended scale —
        # blocking-reduced candidate sets on the toy/small datasets this
        # milestone targets, not the thousands-of-entities KG-scale
        # workloads that graph-embedding-based EA (a later milestone)
        # would need to worry about.
        cost_matrix = [
            [
                1.0 - score
                if (score := scores_by_pair.get((source_id, target_id))) is not None
                else _SENTINEL_COST
                for target_id in target_ids
            ]
            for source_id in source_ids
        ]
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        results = []
        for row, col in zip(row_indices, col_indices, strict=True):
            source_id, target_id = source_ids[row], target_ids[col]
            if (source_id, target_id) not in scores_by_pair:
                continue  # matched to the sentinel: no real candidate available
            ranked = rank_candidates(candidates_by_source[source_id])
            results.append(
                AlignmentResult(
                    source_id=source_id,
                    target_id=target_id,
                    score=scores_by_pair[(source_id, target_id)],
                    alternatives=[t for t, _ in ranked if t != target_id],
                )
            )
        return results

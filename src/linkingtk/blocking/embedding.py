"""Fuzzy blocking strategy based on TF-IDF vector similarity.

A non-neural, classical-ML alternative to :class:`~linkingtk.blocking.label_overlap.LabelOverlap`:
entities are compared by cosine similarity of vectors from a pluggable
scikit-learn-style text vectorizer (TF-IDF by default) rather than surface-
string overlap, so paraphrased or reordered text (e.g. a description and a
differently-worded gloss of the same sense) can still block together.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from linkingtk.blocking.base import BlockingStrategy, rank_top_matches
from linkingtk.core.entity import Entity
from linkingtk.core.text import Field, resolve_field


class Vectorizer(Protocol):
    """Structural type for a fit_transform/transform text vectorizer (e.g. scikit-learn's)."""

    def fit_transform(self, raw_documents: Iterable[str]) -> Any: ...
    def transform(self, raw_documents: Iterable[str]) -> Any: ...


class EmbeddingSimilarityBlocker(BlockingStrategy):
    """Blocks entities by cosine similarity of vectorized text.

    Fits ``vectorizer`` on ``dataset2`` (treating it as the reference
    corpus) and projects ``dataset1`` into that same vector space, then for
    each ``dataset1`` entity retrieves the ``top_k`` closest ``dataset2``
    entities by cosine similarity via
    :class:`sklearn.neighbors.NearestNeighbors`.

    Args:
        field: Which text to compare entities on: ``"label"`` (all
            labels, space-joined), ``"description"``, or ``"context"``.
            A callable taking an ``Entity`` and returning ``str`` may be
            passed instead, for fields not covered above.
        top_k: Maximum number of candidates to keep per source entity.
        threshold: Optional minimum cosine similarity a candidate must
            reach to be kept. If not set, candidates with a similarity of
            exactly zero are still dropped, since ``top_k`` nearest-
            neighbor search otherwise always returns ``top_k`` results
            regardless of relevance. This default assumes a non-negative
            vector representation (true for ``TfidfVectorizer``,
            ``CountVectorizer``, and other bag-of-words-style vectorizers,
            where a zero similarity exactly means "shares nothing"); for a
            dense embedding vectorizer, where small positive or negative
            similarities can be meaningful, pass an explicit ``threshold``
            instead of relying on this default.
        vectorizer: Any object exposing scikit-learn's
            ``fit_transform``/``transform`` interface (e.g.
            ``TfidfVectorizer``, ``CountVectorizer``, or a custom embedding
            wrapper). Defaults to
            ``TfidfVectorizer(stop_words="english")``. Document-frequency
            pruning, n-gram ranges, and other vectorization choices are
            configured on the vectorizer itself rather than on this class.
    """

    def __init__(
        self,
        field: Field = "label",
        top_k: int = 10,
        threshold: float | None = None,
        vectorizer: Vectorizer | None = None,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        self.field = field
        self.top_k = top_k
        self.threshold = threshold
        self.vectorizer: Vectorizer = (
            vectorizer if vectorizer is not None else TfidfVectorizer(stop_words="english")
        )

    def candidate_pairs(
        self, dataset1: list[Entity], dataset2: list[Entity]
    ) -> list[tuple[Entity, Entity]]:
        if not dataset1 or not dataset2:
            return []

        extract = resolve_field(self.field)
        texts2 = [extract(entity) for entity in dataset2]
        texts1 = [extract(entity) for entity in dataset1]

        matrix2 = self.vectorizer.fit_transform(texts2)
        matrix1 = self.vectorizer.transform(texts1)

        n_neighbors = min(self.top_k, matrix2.shape[0])
        model = NearestNeighbors(metric="cosine", n_neighbors=n_neighbors).fit(matrix2)
        distances, indices = model.kneighbors(matrix1)

        entities2_by_id = {entity.id: entity for entity in dataset2}
        threshold = self.threshold

        def keep(score: float) -> bool:
            return score >= threshold if threshold is not None else score > 0.0

        pairs: list[tuple[Entity, Entity]] = []
        for row, entity1 in enumerate(dataset1):
            scores = {
                dataset2[col].id: 1.0 - dist
                for col, dist in zip(indices[row], distances[row], strict=True)
            }
            matches = rank_top_matches(
                scores, entities2_by_id, self.top_k, descending=True, keep=keep
            )
            pairs.extend((entity1, entity2) for entity2 in matches)
        return pairs

"""Fuzzy blocking strategy based on TF-IDF vector similarity.

A non-neural, classical-ML alternative to :class:`~linkingtk.blocking.label_overlap.LabelOverlap`:
entities are compared by cosine similarity of TF-IDF-weighted term
vectors rather than surface-string overlap, so paraphrased or reordered
text (e.g. a description and a differently-worded gloss of the same
sense) can still block together.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable

from linkingtk.blocking.base import BlockingStrategy, rank_top_matches
from linkingtk.core.entity import Entity
from linkingtk.core.text import Field, resolve_field, tokenize

TfIdfVector = dict[str, float]


def _document_frequency(token_lists: Iterable[list[str]]) -> dict[str, int]:
    frequency: dict[str, int] = defaultdict(int)
    for tokens in token_lists:
        for token in set(tokens):
            frequency[token] += 1
    return frequency


def _vectorize(tokens: list[str], idf: dict[str, float]) -> TfIdfVector:
    term_frequency: dict[str, int] = defaultdict(int)
    for token in tokens:
        term_frequency[token] += 1
    return {token: count * idf[token] for token, count in term_frequency.items() if token in idf}


def _norm(vector: TfIdfVector) -> float:
    return math.sqrt(sum(weight * weight for weight in vector.values()))


class EmbeddingSimilarityBlocker(BlockingStrategy):
    """Blocks entities by cosine similarity of TF-IDF term vectors.

    Builds a TF-IDF index over ``dataset2`` (IDF weights come from
    ``dataset2`` alone, treating it as the reference corpus), then for
    each ``dataset1`` entity retrieves the ``top_k`` closest ``dataset2``
    entities by cosine similarity. Only candidates sharing at least one
    term are ever scored, via an inverted index, so this stays well
    below the full O(n*m) comparison space, as long as ``dataset2`` has no
    term shared by an unbounded fraction of its entities (see
    ``max_document_frequency``).

    Args:
        field: Which text to compare entities on: ``"label"`` (all
            labels, space-joined), ``"description"``, or ``"context"``.
            A callable taking an ``Entity`` and returning ``str`` may be
            passed instead, for fields not covered above.
        top_k: Maximum number of candidates to keep per source entity.
        threshold: Optional minimum cosine similarity a candidate must
            reach to be kept.
        max_document_frequency: Terms shared by more than this many
            ``dataset2`` entities are dropped as uninformative before
            matching, bounding the cost of any one term's posting list.
    """

    def __init__(
        self,
        field: Field = "label",
        top_k: int = 10,
        threshold: float | None = None,
        max_document_frequency: int = 1000,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        self.field = field
        self.top_k = top_k
        self.threshold = threshold
        self.max_document_frequency = max_document_frequency

    def candidate_pairs(
        self, dataset1: list[Entity], dataset2: list[Entity]
    ) -> list[tuple[Entity, Entity]]:
        extract = resolve_field(self.field)

        entities2_by_id = {entity.id: entity for entity in dataset2}
        token_lists2 = {entity.id: tokenize(extract(entity)) for entity in dataset2}
        idf = {
            token: math.log((1 + len(dataset2)) / (1 + df)) + 1.0
            for token, df in _document_frequency(token_lists2.values()).items()
            if df <= self.max_document_frequency
        }

        # Vector norms and the inverted index are both derived from the
        # same per-entity TF-IDF vector, so build them in one pass rather
        # than three separate traversals of dataset2.
        norms2: dict[str, float] = {}
        inverted_index: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for entity_id, tokens in token_lists2.items():
            vector = _vectorize(tokens, idf)
            norms2[entity_id] = _norm(vector)
            for token, weight in vector.items():
                inverted_index[token].append((entity_id, weight))

        keep = (lambda score: score >= self.threshold) if self.threshold is not None else None
        pairs: list[tuple[Entity, Entity]] = []
        for entity1 in dataset1:
            vector1 = _vectorize(tokenize(extract(entity1)), idf)
            norm1 = _norm(vector1)
            if norm1 == 0.0:
                continue

            dot_products: dict[str, float] = defaultdict(float)
            for token, weight1 in vector1.items():
                for entity_id, weight2 in inverted_index.get(token, []):
                    dot_products[entity_id] += weight1 * weight2

            scores = {
                entity_id: dot / (norm1 * norms2[entity_id])
                for entity_id, dot in dot_products.items()
            }
            matches = rank_top_matches(
                scores, entities2_by_id, self.top_k, descending=True, keep=keep
            )
            pairs.extend((entity1, entity2) for entity2 in matches)
        return pairs

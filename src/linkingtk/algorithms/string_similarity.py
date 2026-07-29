"""General string-similarity baseline linker.

Scores each blocked candidate pair by the similarity of a configurable
text field from either side (e.g. a mention's context against a
candidate's gloss, or two entities' labels), then ranks candidates per
source entity. This is a general-purpose baseline for Entity Alignment,
Entity Linking and Word Sense Alignment; the classic Lesk algorithm for
WSD (:class:`linkingtk.algorithms.wsd.lesk.LeskLinker`) is implemented as
a preconfigured instance of this class.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Callable
from functools import cache
from typing import Literal

from linkingtk.algorithms.base import DEFAULT_BLOCKING, BaseLinker, Graph
from linkingtk.blocking.base import BlockingStrategy
from linkingtk.blocking.trie import edit_distance
from linkingtk.core.entity import Entity, context_text, description_text, label_texts
from linkingtk.core.result import AlignmentResult

FieldName = Literal["label", "description", "context"]
Field = FieldName | Callable[[Entity], str]
MetricName = Literal["word_overlap", "jaccard", "levenshtein"]
Metric = MetricName | Callable[[str, str], float]

_TOKEN_RE = re.compile(r"[a-zA-Z]+")

_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "of",
        "to",
        "in",
        "on",
        "at",
        "for",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "it",
        "its",
        "this",
        "that",
        "with",
        "as",
        "by",
        "from",
        "has",
        "have",
        "had",
        "not",
        "he",
        "she",
        "they",
        "we",
        "you",
        "i",
    }
)


@cache
def _tokenize(text: str) -> frozenset[str]:
    """Lowercase, alphabetic-token bag-of-words with stopwords removed."""
    return frozenset(
        token.lower() for token in _TOKEN_RE.findall(text) if token.lower() not in _STOPWORDS
    )


def _word_overlap(text1: str, text2: str) -> float:
    """Count of shared tokens, as used by the classic Lesk algorithm."""
    return float(len(_tokenize(text1) & _tokenize(text2)))


def _jaccard(text1: str, text2: str) -> float:
    """Token-set Jaccard similarity, normalized to ``[0, 1]``."""
    tokens1, tokens2 = _tokenize(text1), _tokenize(text2)
    union = tokens1 | tokens2
    return len(tokens1 & tokens2) / len(union) if union else 0.0


def _levenshtein_similarity(text1: str, text2: str) -> float:
    """Character-level similarity derived from normalized edit distance, in ``[0, 1]``."""
    text1, text2 = text1.lower(), text2.lower()
    denom = len(text1) + len(text2)
    if denom == 0:
        return 1.0
    return 1.0 - edit_distance(text1, text2) / denom


_FIELD_EXTRACTORS: dict[FieldName, Callable[[Entity], str]] = {
    "label": lambda entity: " ".join(label_texts(entity)),
    "description": description_text,
    "context": context_text,
}

_METRICS: dict[MetricName, Callable[[str, str], float]] = {
    "word_overlap": _word_overlap,
    "jaccard": _jaccard,
    "levenshtein": _levenshtein_similarity,
}


class StringSimilarityLinker(BaseLinker):
    """Baseline linker that scores candidates by string similarity.

    For each blocked ``(entity1, entity2)`` pair, extracts ``source_field``
    text from ``entity1`` and ``target_field`` text from ``entity2``, and
    scores the pair with ``metric`` (higher is always better, regardless
    of metric). For each source entity, the highest-scoring candidate is
    returned as the predicted link, with the rest kept as ranked
    ``alternatives``.

    Args:
        source_field: Which text to extract from each ``dataset1`` entity:
            ``"label"`` (all labels, space-joined), ``"description"``, or
            ``"context"``. A callable taking an ``Entity`` and returning
            ``str`` may be passed instead, for fields not covered above.
        target_field: Which text to extract from each ``dataset2`` entity;
            same options as ``source_field``.
        metric: ``"word_overlap"`` (raw shared-token count), ``"jaccard"``
            (normalized token overlap), or ``"levenshtein"`` (normalized
            edit-distance similarity). A callable taking two strings and
            returning a similarity score (higher is better) may be passed
            instead, for metrics not covered above.
    """

    def __init__(
        self,
        source_field: Field = "label",
        target_field: Field = "label",
        metric: Metric = "jaccard",
    ) -> None:
        self.source_field = source_field
        self.target_field = target_field
        self.metric = metric

    def link(
        self,
        dataset1: list[Entity],
        dataset2: list[Entity],
        graph: Graph = None,
        blocking: BlockingStrategy = DEFAULT_BLOCKING,
    ) -> list[AlignmentResult]:
        extract_source = (
            _FIELD_EXTRACTORS[self.source_field]
            if isinstance(self.source_field, str)
            else self.source_field
        )
        extract_target = (
            _FIELD_EXTRACTORS[self.target_field]
            if isinstance(self.target_field, str)
            else self.target_field
        )
        score = _METRICS[self.metric] if isinstance(self.metric, str) else self.metric

        pairs = blocking.candidate_pairs(dataset1, dataset2)

        # Extract each entity's comparison text once, rather than
        # re-extracting (and, for token-based metrics, re-tokenizing) it
        # for every candidate pair it appears in.
        source_texts = {entity.id: extract_source(entity) for entity in dataset1}
        target_texts = {entity.id: extract_target(entity) for entity in dataset2}

        candidates_by_source: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for entity1, entity2 in pairs:
            candidates_by_source[entity1.id].append(
                (entity2.id, score(source_texts[entity1.id], target_texts[entity2.id]))
            )

        results = []
        for source_id, candidates in candidates_by_source.items():
            candidates.sort(key=lambda item: (-item[1], item[0]))
            best_id, best_score = candidates[0]
            results.append(
                AlignmentResult(
                    source_id=source_id,
                    target_id=best_id,
                    score=best_score,
                    alternatives=[target_id for target_id, _ in candidates[1:]],
                )
            )
        return results

"""Lesk-style Word Sense Disambiguation linker.

References:
    Lesk, M. (1986). Automatic sense disambiguation using machine readable
    dictionaries: how to tell a pine cone from an ice cream cone. In
    Proceedings of the 5th annual international conference on Systems
    documentation (SIGDOC '86).
"""

from __future__ import annotations

import re
from collections import defaultdict

from linkingtk.algorithms.base import DEFAULT_BLOCKING, BaseLinker, Graph
from linkingtk.blocking.base import BlockingStrategy
from linkingtk.core.entity import Entity
from linkingtk.core.result import AlignmentResult

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


def _tokenize(text: str) -> set[str]:
    """Lowercase, alphabetic-token bag-of-words with stopwords removed."""
    return {token.lower() for token in _TOKEN_RE.findall(text) if token.lower() not in _STOPWORDS}


def _context_text(entity: Entity) -> str:
    if entity.context is None:
        return ""
    return entity.context if isinstance(entity.context, str) else entity.context[0]


def _description_text(entity: Entity) -> str:
    if entity.description is None:
        return ""
    return entity.description if isinstance(entity.description, str) else entity.description[0]


class LeskLinker(BaseLinker):
    """WSD linker that scores candidate senses by context/gloss overlap.

    For each mention in ``dataset1``, candidate senses in ``dataset2`` are
    ranked by the size of the token overlap between the mention's
    ``context`` and the candidate's ``description`` (gloss). The
    highest-overlap sense is returned as the predicted link, with the rest
    kept as ranked ``alternatives`` for evaluation with
    :meth:`linkingtk.eval.Evaluator.evaluate_ranked`.
    """

    def link(
        self,
        dataset1: list[Entity],
        dataset2: list[Entity],
        graph: Graph = None,
        blocking: BlockingStrategy = DEFAULT_BLOCKING,
    ) -> list[AlignmentResult]:
        pairs = blocking.candidate_pairs(dataset1, dataset2)

        candidates_by_source: dict[str, list[tuple[str, int]]] = defaultdict(list)
        for mention, sense in pairs:
            context_tokens = _tokenize(_context_text(mention))
            gloss_tokens = _tokenize(_description_text(sense))
            overlap = len(context_tokens & gloss_tokens)
            candidates_by_source[mention.id].append((sense.id, overlap))

        results = []
        for source_id, candidates in candidates_by_source.items():
            candidates.sort(key=lambda item: item[1], reverse=True)
            best_id, best_score = candidates[0]
            results.append(
                AlignmentResult(
                    source_id=source_id,
                    target_id=best_id,
                    score=float(best_score),
                    alternatives=[target_id for target_id, _ in candidates[1:]],
                )
            )
        return results

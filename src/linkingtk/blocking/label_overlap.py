"""Approximate string matching blocking strategy.

Ported from Naisc's ``ApproximateStringMatching``:
https://github.com/insight-centre/naisc/blob/master/naisc-core/src/main/java/org/insightcentre/uld/naisc/blocking/ApproximateStringMatching.java

Two interchangeable metrics are provided, matching the reference
implementation:

* ``"ngram"``: candidates are ranked by a frequency-weighted count of
  shared character n-grams (fast, index-based, no edit-distance
  computation).
* ``"levenshtein"``: candidates are ranked by normalized Levenshtein
  distance, using the :class:`~linkingtk.blocking.trie.PatriciaTrie` for
  an approximate nearest-neighbor search that stays tractable on larger
  datasets.

Unlike the original, both metrics support an optional ``threshold`` to
drop weak candidates outright.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

from linkingtk.blocking.base import BlockingStrategy, rank_top_matches
from linkingtk.blocking.trie import PatriciaTrie
from linkingtk.core.entity import Entity, label_texts

logger = logging.getLogger("linkingtk")

Metric = Literal["ngram", "levenshtein"]

# Naisc caps n-gram extraction on the query side to the first N characters,
# for speed; the label_length used for scoring is unaffected by this cap.
_NGRAM_QUERY_SCAN_LIMIT = 100

# If pruning common n-grams (see max_ngram_document_frequency) removes more
# than this fraction of the index, matching is likely to be poor and worth
# a warning.
_COMMON_NGRAM_WARN_RATIO = 0.9


@dataclass
class _NgramStat:
    frequency: int
    label_length: int


class LabelOverlap(BlockingStrategy):
    """Blocks entities by approximate string similarity of their labels.

    Args:
        metric: ``"ngram"`` (default) ranks candidates by shared
            character n-grams; ``"levenshtein"`` ranks them by normalized
            edit distance.
        ngram_size: Character n-gram length. Only used by the ``"ngram"``
            metric.
        max_matches: Maximum number of candidates to keep per source
            entity.
        lowercase: Whether to lowercase labels before comparing.
        threshold: Optional cutoff used to drop weak candidates. For
            ``"ngram"`` this is a *minimum* score to keep (higher is
            better, unbounded above). For ``"levenshtein"`` this is a
            *maximum* normalized edit distance to keep (0 = identical, 1
            = completely different).
        queue_max: Maximum search-frontier size for the trie-based
            search. Only used by the ``"levenshtein"`` metric. Defaults
            to ``max_matches * 20``, as in the original implementation.
        max_ngram_document_frequency: N-grams shared by more than this
            many target entities are dropped as uninformative before
            matching. Only used by the ``"ngram"`` metric.
    """

    ranked = True

    def __init__(
        self,
        metric: Metric = "ngram",
        ngram_size: int = 3,
        max_matches: int = 5,
        lowercase: bool = True,
        threshold: float | None = None,
        queue_max: int | None = None,
        max_ngram_document_frequency: int = 1000,
    ) -> None:
        if max_matches < 1:
            raise ValueError("max_matches must be at least 1")
        if ngram_size < 1:
            raise ValueError("ngram_size must be at least 1")
        self.metric = metric
        self.ngram_size = ngram_size
        self.max_matches = max_matches
        self.lowercase = lowercase
        self.threshold = threshold
        self.queue_max = queue_max if queue_max is not None else max_matches * 20
        self.max_ngram_document_frequency = max_ngram_document_frequency

    def candidate_pairs(
        self, dataset1: list[Entity], dataset2: list[Entity]
    ) -> list[tuple[Entity, Entity]]:
        entities2_by_id = {entity.id: entity for entity in dataset2}
        if self.metric == "ngram":
            return self._ngram_candidate_pairs(dataset1, dataset2, entities2_by_id)
        return self._levenshtein_candidate_pairs(dataset1, dataset2, entities2_by_id)

    def _normalize(self, text: str) -> str:
        return text.lower() if self.lowercase else text

    def _extract_ngrams(self, text: str) -> list[str]:
        n = self.ngram_size
        return [text[i : i + n] for i in range(len(text) - n + 1)]

    def _ngram_candidate_pairs(
        self,
        dataset1: list[Entity],
        dataset2: list[Entity],
        entities2_by_id: dict[str, Entity],
    ) -> list[tuple[Entity, Entity]]:
        index: dict[str, dict[str, _NgramStat]] = defaultdict(dict)

        for entity2 in dataset2:
            for label in label_texts(entity2):
                text = self._normalize(label)
                for ngram in self._extract_ngrams(text):
                    stat = index[ngram].get(entity2.id)
                    if stat is None:
                        index[ngram][entity2.id] = _NgramStat(frequency=1, label_length=len(text))
                    else:
                        stat.frequency += 1

        # Heuristically drop n-grams that are too common to be informative.
        total_ngrams = len(index)
        index = {
            ngram: matches
            for ngram, matches in index.items()
            if len(matches) <= self.max_ngram_document_frequency
        }
        if total_ngrams and len(index) / total_ngrams < _COMMON_NGRAM_WARN_RATIO:
            logger.warning(
                "N-gram blocking is pruning many common n-grams (kept %d/%d); "
                "consider a larger ngram_size (currently %d)",
                len(index),
                total_ngrams,
                self.ngram_size,
            )

        keep = (lambda score: score >= self.threshold) if self.threshold is not None else None
        pairs: list[tuple[Entity, Entity]] = []
        for entity1 in dataset1:
            scores: dict[str, float] = defaultdict(float)
            seen: dict[str, int] = defaultdict(int)
            for label in label_texts(entity1):
                text = self._normalize(label)
                for ngram in self._extract_ngrams(text[:_NGRAM_QUERY_SCAN_LIMIT]):
                    matches = index.get(ngram)
                    if not matches:
                        continue
                    for entity2_id, stat in matches.items():
                        if seen[ngram] < stat.frequency:
                            scores[entity2_id] += 1.0 / (stat.label_length + len(text))
                    seen[ngram] += 1

            matches_for_entity1 = rank_top_matches(
                scores, entities2_by_id, self.max_matches, descending=True, keep=keep
            )
            pairs.extend((entity1, entity2) for entity2 in matches_for_entity1)
        return pairs

    def _levenshtein_candidate_pairs(
        self,
        dataset1: list[Entity],
        dataset2: list[Entity],
        entities2_by_id: dict[str, Entity],
    ) -> list[tuple[Entity, Entity]]:
        trie: PatriciaTrie[str] = PatriciaTrie()
        for entity2 in dataset2:
            for label in label_texts(entity2):
                trie.insert(self._normalize(label), entity2.id)

        keep = (lambda score: score <= self.threshold) if self.threshold is not None else None
        pairs: list[tuple[Entity, Entity]] = []
        for entity1 in dataset1:
            best_by_id: dict[str, float] = {}
            for label in label_texts(entity1):
                text = self._normalize(label)
                for entity2_id, score in trie.nearest(text, self.max_matches, self.queue_max):
                    if entity2_id not in best_by_id or score < best_by_id[entity2_id]:
                        best_by_id[entity2_id] = score

            matches_for_entity1 = rank_top_matches(
                best_by_id, entities2_by_id, self.max_matches, descending=False, keep=keep
            )
            pairs.extend((entity1, entity2) for entity2 in matches_for_entity1)
        return pairs

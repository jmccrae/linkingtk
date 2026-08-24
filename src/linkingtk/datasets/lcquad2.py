"""LCQuAD 2.0 Entity Linking dataset loader.

LCQuAD 2.0 (Dubey et al., 2019,
https://figshare.com/projects/LCQuAD_2_0/62270) is a large-scale KGQA
benchmark: ~30,000 crowd-sourced natural-language questions, each paired
with a SPARQL query over Wikidata (and a DBpedia translation, unused
here). Hosted as two public figshare articles (CC BY 4.0, direct HTTPS
download, no auth): "training set for lcquad 2.0" (article 8479055,
24180 rows) and "test set for lcquad 2.0" (article 8479052, 6046 rows).
LCQuAD 2.0 has no separate validation split.

The release publishes no explicit entity-mention span annotations. Many
rows carry an ``NNQT_question`` template with curly-/angle-brace-
delimited slots, but a slot's alignment to a specific KG id is
template-specific and not reliably recoverable by position alone --
verified directly: a slot's left-to-right order in ``NNQT_question``
does not track its corresponding id's order inside ``sparql_wikidata``
(e.g. an answer-type constraint's slot can appear *before* its
subject entity's, following that row's ``template`` pattern rather than
SPARQL triple order).

This loader instead derives mentions directly from the SPARQL query: every
``wd:Qxxx`` reference inside ``sparql_wikidata`` is a gold entity for that
question. Its text span is located by a literal (case-insensitive)
substring search for the entity's Wikidata label, tried first against the
``question`` field then the ``paraphrased_question`` fallback. Measured on
a live 300-question sample of the real test set: ~81% of (question,
entity) references resolve this way (341/421) -- the rest are skipped
(some Wikidata labels have drifted or the item has been merged/deleted
since the 2019 release; others are genuine paraphrases using different
words than the canonical label, e.g. "Ebola virus" vs. the label "Ebola
hemorrhagic fever"). This is the same "skip what can't be resolved,
document plainly" convention as
[AidaConllDataset][linkingtk.datasets.aida_conll.AidaConllDataset]'s
NIL-mention skip, just applied at label-matching time instead of
NIL-annotation time. Property (``wdt:``/``p:``/``ps:``/``pq:`` ``Pxxx``)
references are deliberately out of scope -- this loader links entities,
not relations, matching every other EL loader in this package.

The release carries only bare QIDs, no entity metadata -- KB entity
labels/descriptions are fetched live from Wikidata's ``wbgetentities``
API (batched 50 ids/request, its own per-request limit; disk-cached
through ``fetch_cached``; retried with backoff on 429/503, since a full
run needs on the order of ~450 batches for the real dataset's ~23,000
unique referenced QIDs -- easy to trip Wikidata's anonymous rate limit
without it).

Unlike [AidaConllDataset][linkingtk.datasets.aida_conll.AidaConllDataset]
or [ZeshelDataset][linkingtk.datasets.zeshel.ZeshelDataset],
``load_splits()`` here can't skip the network fetch: resolving a mention's
span itself requires each candidate entity's Wikidata label, not just its
id, so the label lookup is unavoidable even for ground-truth-only access.
It's still provided, for interface parity with those loaders, but only
skips building the full KB `Entity` list -- not the label fetch itself.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlencode

from linkingtk.core.entity import Entity
from linkingtk.datasets._util import fetch_cached
from linkingtk.datasets.base import DatasetLoader

logger = logging.getLogger("linkingtk")

_TRAIN_URL = "https://ndownloader.figshare.com/files/15738824"
_TEST_URL = "https://ndownloader.figshare.com/files/15738818"
_SPLITS = ("train", "test")
_WIKIDATA_API = "https://www.wikidata.org/w/api.php"
_LABEL_BATCH_SIZE = 50
_MAX_RETRIES = 5
# Wikidata's API (like Wikipedia's) 403s on urllib's default User-Agent --
# it requires a descriptive one identifying the client.
_USER_AGENT = "linkingtk/0.1 (https://github.com/jmccrae/linkingtk)"

_QID_RE = re.compile(r"\bwd:(Q\d+)\b")

LabelFetcher = Callable[[list[str]], dict[str, tuple[str, str]]]


def _entity_qids(sparql: str) -> list[str]:
    """The distinct ``Qxxx`` ids referenced by a SPARQL query, in first-occurrence order."""
    return list(dict.fromkeys(_QID_RE.findall(sparql)))


def _find_span(text: str, label: str) -> tuple[int, int] | None:
    if not text or not label:
        return None
    index = text.lower().find(label.lower())
    return (index, index + len(label)) if index != -1 else None


def _question_texts(row: dict[str, Any]) -> list[str]:
    paraphrased = row.get("paraphrased_question")
    texts = [row.get("question") or ""]
    if isinstance(paraphrased, str) and paraphrased:
        texts.append(paraphrased)
    return texts


def fetch_wikidata_labels(
    qids: list[str], cache_dir: Path | None = None
) -> dict[str, tuple[str, str]]:
    """Batch-fetch each QID's English label + description, cached to disk.

    Uses ``wbgetentities`` in batches of ``_LABEL_BATCH_SIZE`` (Wikidata's
    own per-request id limit). Each request is retried with exponential
    backoff on a 429/503 response -- LCQuAD 2.0's real train+test split
    references ~23,000 unique QIDs (~450 batches), enough to reliably hit
    Wikidata's anonymous rate limit without backoff.

    Args:
        qids: Wikidata QIDs (``"Q..."``).
        cache_dir: Forwarded to ``fetch_cached``.

    Returns:
        ``qid -> (label, description)``. A QID with no English label (a
        page merged/deleted since the 2019 release) maps to ``("", "")``.
    """
    unique_qids = list(dict.fromkeys(qids))
    result: dict[str, tuple[str, str]] = {}
    for start in range(0, len(unique_qids), _LABEL_BATCH_SIZE):
        batch = unique_qids[start : start + _LABEL_BATCH_SIZE]
        params = {
            "action": "wbgetentities",
            "format": "json",
            "props": "labels|descriptions",
            "languages": "en",
            "ids": "|".join(batch),
        }
        url = f"{_WIKIDATA_API}?{urlencode(params)}"
        payload = json.loads(_fetch_with_retry(url, cache_dir))
        for qid, entity in payload.get("entities", {}).items():
            label = entity.get("labels", {}).get("en", {}).get("value", "")
            description = entity.get("descriptions", {}).get("en", {}).get("value", "")
            result[qid] = (label, description)
    return {qid: result.get(qid, ("", "")) for qid in qids}


def _fetch_with_retry(url: str, cache_dir: Path | None) -> bytes:
    for attempt in range(_MAX_RETRIES):
        try:
            return fetch_cached(url, cache_dir, headers={"User-Agent": _USER_AGENT})
        except HTTPError as error:
            if error.code not in (429, 503) or attempt == _MAX_RETRIES - 1:
                raise
            delay = 2**attempt
            logger.info("Wikidata API returned %s, retrying in %ss", error.code, delay)
            time.sleep(delay)
    raise AssertionError("unreachable")  # pragma: no cover


def _read_rows(source: str, cache_dir: Path | None) -> list[dict[str, Any]]:
    if source.startswith(("http://", "https://", "file://")):
        data = fetch_cached(source, cache_dir)
    else:
        data = Path(source).read_bytes()
    rows: list[dict[str, Any]] = json.loads(data)
    return rows


class Lcquad2Dataset(DatasetLoader):
    """LCQuAD 2.0: crowd-sourced KGQA questions, entities linked to Wikidata.

    See the module docstring for how mentions are derived (no native span
    annotations exist -- spans are recovered by matching each SPARQL
    entity's Wikidata label against the question text) and why
    ``load_splits()`` can't avoid the label-lookup network cost the way
    sibling loaders' can.

    ``dataset1`` (mentions) carries ``context=(text, start, end)`` --
    matching [algorithms.el][linkingtk.algorithms.el]'s "mentions carry
    context, no description" convention. ``dataset2`` (KB entries) carries
    a label + description fetched live from Wikidata.

    Args:
        train_source: URL or local path to the training-split JSON file.
        test_source: URL or local path to the test-split JSON file.
        cache_dir: Override for the Wikidata-label/JSON-file download
            cache directory. Ignored if ``label_fetcher`` is given.
        label_fetcher: Overrides how KB entity labels/descriptions are
            sourced, given the list of unique QIDs referenced anywhere in
            the loaded rows. Defaults to
            [fetch_wikidata_labels][linkingtk.datasets.lcquad2.fetch_wikidata_labels]
            -- pass a fake here in tests to avoid a real network call.
    """

    def __init__(
        self,
        train_source: str = _TRAIN_URL,
        test_source: str = _TEST_URL,
        cache_dir: Path | None = None,
        label_fetcher: LabelFetcher | None = None,
    ) -> None:
        self.train_source = train_source
        self.test_source = test_source
        self.cache_dir = cache_dir
        self.label_fetcher = (
            label_fetcher
            if label_fetcher is not None
            else lambda qids: fetch_wikidata_labels(qids, self.cache_dir)
        )

    def load(self) -> tuple[list[Entity], list[Entity], list[tuple[str, str]]]:
        mentions, kb, ground_truth, _ = self._build(build_kb=True)
        return mentions, kb, ground_truth

    def load_splits(
        self,
    ) -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[tuple[str, str]]]:
        """Load this dataset's native train/test ground-truth split.

        LCQuAD 2.0 has no native validation split -- the third element is
        always empty, matching the shape of
        [AidaConllDataset.load_splits][linkingtk.datasets.aida_conll.AidaConllDataset.load_splits].
        Unlike that method, this still performs the full label lookup (see
        the module docstring) -- it only skips building `dataset2`'s
        `Entity` objects.

        Returns:
            ``(train_pairs, test_pairs, [])``.
        """
        _mentions, _kb, _ground_truth, ground_truth_by_split = self._build(build_kb=False)
        return ground_truth_by_split["train"], ground_truth_by_split["test"], []

    def _build(
        self, build_kb: bool
    ) -> tuple[list[Entity], list[Entity], list[tuple[str, str]], dict[str, list[tuple[str, str]]]]:
        sources = {"train": self.train_source, "test": self.test_source}
        rows_by_split = {
            split: _read_rows(source, self.cache_dir) for split, source in sources.items()
        }

        all_qids: set[str] = set()
        for rows in rows_by_split.values():
            for row in rows:
                all_qids.update(_entity_qids(row.get("sparql_wikidata") or ""))
        labels = self.label_fetcher(list(all_qids))

        mentions: list[Entity] = []
        ground_truth: list[tuple[str, str]] = []
        ground_truth_by_split: dict[str, list[tuple[str, str]]] = {}
        seen_qids: set[str] = set()

        for split in _SPLITS:
            split_ground_truth: list[tuple[str, str]] = []
            for row in rows_by_split[split]:
                uid = row.get("uid")
                texts = _question_texts(row)
                for qid in _entity_qids(row.get("sparql_wikidata") or ""):
                    label, _description = labels.get(qid, ("", ""))
                    match = next(
                        (
                            (text, span)
                            for text in texts
                            if (span := _find_span(text, label)) is not None
                        ),
                        None,
                    )
                    if match is None:
                        continue
                    text, (start, end) = match
                    mention_id = f"lcquad2:{split}:{uid}:{qid}"
                    mentions.append(
                        Entity(
                            id=mention_id,
                            labels=[text[start:end]],
                            context=(text, start, end),
                        )
                    )
                    split_ground_truth.append((mention_id, qid))
                    seen_qids.add(qid)
            ground_truth_by_split[split] = split_ground_truth
            ground_truth.extend(split_ground_truth)

        kb: list[Entity] = []
        if build_kb:
            for qid in seen_qids:
                label, description = labels.get(qid, ("", ""))
                kb.append(
                    Entity(
                        id=qid,
                        labels=[label] if label else [],
                        description=description or None,
                    )
                )

        return mentions, kb, ground_truth, ground_truth_by_split

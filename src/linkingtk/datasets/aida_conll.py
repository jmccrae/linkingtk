"""AIDA-CoNLL Entity Linking dataset loader.

AIDA-CoNLL (Hoffart et al., 2011) links named-entity mentions in Reuters
newswire text (the CoNLL 2003 NER corpus) to Wikipedia/Wikidata entities.
The original release's licensing is murky (the underlying CoNLL 2003 text
has its own Reuters-derived redistribution history); this loader instead
uses a community republish already redistributed under a
clear license: the Hugging Face Hub dataset
``cyanic-selkie/aida-conll-yago-wikidata`` (cc-by-sa-3.0), which maps the
original AIDA-YAGO2 annotations onto Wikidata QIDs. Loaded via the
``datasets`` library (already a hard dependency of this package -- no new
install).

Each document's gold entities carry a Wikipedia page title but no
description; this loader fetches one via Wikipedia's MediaWiki API
(``action=query&prop=extracts``, batched and disk-cached through
``fetch_cached``) so KB entities have
real label+description text, not just a bare title.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from linkingtk.core.entity import Entity
from linkingtk.datasets._util import fetch_cached
from linkingtk.datasets.base import DatasetLoader

logger = logging.getLogger("linkingtk")

_SOURCE = "cyanic-selkie/aida-conll-yago-wikidata"
_SPLITS = ("train", "validation", "test")
_WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
_EXTRACT_BATCH_SIZE = 50
# Wikipedia's API 403s on urllib's default User-Agent -- it requires a
# descriptive one identifying the client, per its API etiquette policy.
_USER_AGENT = "linkingtk/0.1 (https://github.com/jmccrae/linkingtk)"

DescriptionFetcher = Callable[[list[str]], dict[str, str]]


def _mention_id(document_id: int, start: int, end: int) -> str:
    return f"mention:{document_id}:{start}:{end}"


def _entity_id(qid: int) -> str:
    return f"Q{qid}"


def fetch_wikipedia_extracts(titles: list[str], cache_dir: Path | None = None) -> dict[str, str]:
    """Batch-fetch each title's Wikipedia lead-paragraph extract, cached to disk.

    Uses MediaWiki's ``action=query&prop=extracts`` API in batches of
    ``_EXTRACT_BATCH_SIZE`` titles per request -- not one request per
    entity, since a full AIDA-CoNLL split has thousands of unique gold
    entities. A batch of ``extracts`` this size is too large for MediaWiki
    to answer in one response: it returns as many pages as fit plus a
    ``continue`` token, silently *omitting* the ``extract`` field (not an
    empty string -- the key is simply absent) for whichever pages didn't
    fit. Measured concretely on AIDA-CoNLL's real title list: the batch
    containing "United Kingdom", "Spain", "Jimi Hendrix" and 27 other
    well-known pages returned real extracts for only 20 of 50 -- silently
    leaving the rest empty, not an error, before this method followed
    ``continue`` (see issue #45's benchmark investigation). Each request
    (initial and every continuation) is cached via ``fetch_cached``, keyed
    by its full URL.

    Args:
        titles: Wikipedia page titles (underscored, as stored by the
            source dataset).
        cache_dir: Forwarded to ``fetch_cached``.

    Returns:
        ``title -> extract`` for every input title, in the same order.
        A title with no extract available (e.g. a deleted page) maps to
        ``""``.
    """
    unique_titles = list(dict.fromkeys(titles))
    extracts: dict[str, str] = {}
    for start in range(0, len(unique_titles), _EXTRACT_BATCH_SIZE):
        batch = unique_titles[start : start + _EXTRACT_BATCH_SIZE]
        base_params = {
            "action": "query",
            "format": "json",
            "prop": "extracts",
            "exintro": "1",
            "explaintext": "1",
            "redirects": "1",
            "titles": "|".join(batch),
        }
        continue_params: dict[str, str] = {}
        while True:
            url = f"{_WIKIPEDIA_API}?{urlencode({**base_params, **continue_params})}"
            payload = json.loads(fetch_cached(url, cache_dir, headers={"User-Agent": _USER_AGENT}))
            query = payload.get("query", {})
            pages = query.get("pages", {})
            # MediaWiki resolves input titles through normalization, then
            # redirects, before landing on a final page -- reverse both maps
            # to recover which *input* title each returned page corresponds to.
            source_by_normalized = {
                item["to"]: item["from"] for item in query.get("normalized", [])
            }
            source_by_redirect = {item["to"]: item["from"] for item in query.get("redirects", [])}
            for page in pages.values():
                if "extract" not in page:
                    continue  # this page didn't fit in this response; it'll come in a later one
                resolved_title = page.get("title", "")
                original = source_by_redirect.get(resolved_title, resolved_title)
                original = source_by_normalized.get(original, original)
                extracts[original] = page["extract"]
            if "continue" not in payload:
                break
            continue_params = payload["continue"]
    return {title: extracts.get(title, "") for title in titles}


class AidaConllDataset(DatasetLoader):
    """AIDA-CoNLL: named-entity mentions in Reuters newswire, linked to Wikidata.

    ``dataset1`` (mentions) carries ``context=(text, start, end)`` -- the
    full document text plus the mention's character span, matching
    [algorithms.el][linkingtk.algorithms.el]'s "mentions carry context, no
    description" convention. ``dataset2`` (KB entries) carries a label
    (the Wikipedia page title, humanized) and a description (a fetched
    Wikipedia lead-paragraph extract). Mentions with no valid gold entity
    (NIL) are skipped entirely, matching the AIDA-CoNLL literature's own
    "InKB" evaluation convention.

    Args:
        source: A Hugging Face Hub dataset id, or a local path to a
            directory of ``train``/``validation``/``test`` parquet files
            in the same shape -- used by tests to point at a tiny local
            fixture instead of the real (946+216+231 document) Hub
            dataset.
        cache_dir: Override for the Wikipedia-extract download cache
            directory. Ignored if ``description_fetcher`` is given.
        description_fetcher: Overrides how KB entity descriptions are
            sourced, given the list of unique Wikipedia titles this split
            needs. Defaults to
            [fetch_wikipedia_extracts][linkingtk.datasets.aida_conll.fetch_wikipedia_extracts]
            -- pass a fake here in tests to avoid a real network call.
    """

    def __init__(
        self,
        source: str = _SOURCE,
        cache_dir: Path | None = None,
        description_fetcher: DescriptionFetcher | None = None,
    ) -> None:
        self.source = source
        self.cache_dir = cache_dir
        self.description_fetcher = (
            description_fetcher
            if description_fetcher is not None
            else lambda titles: fetch_wikipedia_extracts(titles, self.cache_dir)
        )

    def load(self) -> tuple[list[Entity], list[Entity], list[tuple[str, str]]]:
        mentions, kb, ground_truth, _ = self._build(_SPLITS, fetch_descriptions=True)
        return mentions, kb, ground_truth

    def load_splits(
        self,
    ) -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[tuple[str, str]]]:
        """Load this dataset's native train/test/validation ground-truth split.

        Unlike ``load()``, this skips description fetching entirely (only
        mention/entity ids are needed) -- same shape and argument order as
        ``_KGZipDataset.load_splits`` (``kg_zip.py``), the established
        precedent for "native supervised split" loaders.

        Returns:
            ``(train_pairs, test_pairs, val_pairs)``.
        """
        _, _, _, ground_truth_by_split = self._build(_SPLITS, fetch_descriptions=False)
        return (
            ground_truth_by_split["train"],
            ground_truth_by_split["test"],
            ground_truth_by_split["validation"],
        )

    def _build(
        self, splits: tuple[str, ...], fetch_descriptions: bool
    ) -> tuple[list[Entity], list[Entity], list[tuple[str, str]], dict[str, list[tuple[str, str]]]]:
        import datasets as hf_datasets

        raw: Any = hf_datasets.load_dataset(self.source)

        mentions: list[Entity] = []
        titles_by_id: dict[str, str] = {}
        ground_truth: list[tuple[str, str]] = []
        ground_truth_by_split: dict[str, list[tuple[str, str]]] = {}

        for split in splits:
            split_ground_truth: list[tuple[str, str]] = []
            for row in raw[split]:
                document_id = row["document_id"]
                text = row["text"]
                for entity in row["entities"]:
                    if entity["qid"] is None:
                        continue
                    mention_id = _mention_id(document_id, entity["start"], entity["end"])
                    entity_id = _entity_id(entity["qid"])
                    mentions.append(
                        Entity(
                            id=mention_id,
                            labels=[text[entity["start"] : entity["end"]]],
                            context=(text, entity["start"], entity["end"]),
                        )
                    )
                    titles_by_id.setdefault(entity_id, entity["title"])
                    split_ground_truth.append((mention_id, entity_id))
            ground_truth_by_split[split] = split_ground_truth
            ground_truth.extend(split_ground_truth)

        descriptions = (
            self.description_fetcher(list(titles_by_id.values())) if fetch_descriptions else {}
        )
        kb = [
            Entity(
                id=entity_id,
                labels=[title.replace("_", " ")],
                description=descriptions.get(title, ""),
            )
            for entity_id, title in titles_by_id.items()
        ]

        return mentions, kb, ground_truth, ground_truth_by_split

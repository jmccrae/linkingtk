"""SemCor 2026 Word Sense Disambiguation dataset loader.

[SemCor 2026](https://github.com/globalwordnet/semcor) is a fork and
reannotation of the classic Brown-Corpus-based SemCor corpus, kept aligned
with the current [Open English WordNet](https://en-word.net/) release.
Unlike the original SemCor release (a Princeton-hosted tarball, WordNet
1.6 sense keys), this fork is distributed directly as per-document files
in its GitHub repository -- one
[Teanga YAML](https://teanga.io/) file per Brown Corpus document, storing
raw text, token character spans, lemmas, POS tags, and three parallel
sense-key layers (``wn16_key``, ``wn30_key``, ``oewn_key``). This loader
uses ``oewn_key``: it's already in `wn`'s own synset-id format (e.g.
``"oewn-08437235-n"``), so no sense-key translation is needed -- unlike
[UfsacDataset][linkingtk.datasets.ufsac.UfsacDataset]'s WordNet 3.0 sense
keys, see [sensekey_to_synset_id][linkingtk.sources.wn.sensekey_to_synset_id].

Since a sense-tagged corpus has no fixed candidate-sense list to
enumerate (every open-class word can in principle be any synset in the
dictionary), ``dataset2`` is a
[WnEntitySource][linkingtk.sources.wn.WnEntitySource] rather than a
materialized ``list[Entity]`` -- this is the loader-side half of that
design; see [EntitySource][linkingtk.core.source.EntitySource].
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from linkingtk.core.entity import Entity
from linkingtk.datasets._util import fetch_cached
from linkingtk.datasets.base import DatasetLoader
from linkingtk.sources.wn import WnEntitySource

_SOURCE = "globalwordnet/semcor"
_REF = "main"
_API_BASE = "https://api.github.com"
_RAW_BASE = "https://raw.githubusercontent.com"
_USER_AGENT = "linkingtk/0.1 (https://github.com/jmccrae/linkingtk)"


def _list_yaml_files(source: str, ref: str, cache_dir: Path | None) -> list[str]:
    """List every ``data/**/*.yaml`` path in `source`, relative to its root."""
    local_root = Path(source)
    if local_root.is_dir():
        return sorted(
            str(path.relative_to(local_root)) for path in (local_root / "data").rglob("*.yaml")
        )

    url = f"{_API_BASE}/repos/{source}/git/trees/{ref}?recursive=1"
    payload = json.loads(fetch_cached(url, cache_dir, headers={"User-Agent": _USER_AGENT}))
    return sorted(
        entry["path"]
        for entry in payload["tree"]
        if entry["path"].startswith("data/") and entry["path"].endswith(".yaml")
    )


def _fetch_yaml_file(source: str, ref: str, relpath: str, cache_dir: Path | None) -> bytes:
    local_root = Path(source)
    if local_root.is_dir():
        return (local_root / relpath).read_bytes()
    return fetch_cached(f"{_RAW_BASE}/{source}/{ref}/{relpath}", cache_dir)


def _document_id(relpath: str) -> str:
    return str(Path(relpath).relative_to("data").with_suffix(""))


def _parse_document(doc_id: str, raw: dict[str, Any]) -> tuple[list[Entity], list[tuple[str, str]]]:
    mentions: list[Entity] = []
    ground_truth: list[tuple[str, str]] = []
    for sent_id, sentence in raw.items():
        if sent_id == "_meta":
            continue
        text = sentence["text"]
        tokens = sentence["tokens"]
        for index, synset_id in sentence.get("oewn_key", []):
            start, end = tokens[index]
            mention_id = f"semcor:{doc_id}:{sent_id}:{index}"
            mentions.append(
                Entity(id=mention_id, labels=[text[start:end]], context=(text, start, end))
            )
            ground_truth.append((mention_id, synset_id))
    return mentions, ground_truth


class SemCorDataset(DatasetLoader):
    """SemCor 2026: Brown Corpus text sense-tagged against Open English WordNet.

    ``dataset1`` (mentions) carries ``context=(sentence_text, start, end)``
    for every sense-tagged content word across every document in the
    corpus (352 documents, ~190K sense-tagged tokens as of this corpus's
    2026 state) -- not a toy subset. ``dataset2`` is a
    [WnEntitySource][linkingtk.sources.wn.WnEntitySource] querying the
    `lexicon` this corpus's ``oewn_key`` layer is aligned to.

    Args:
        source: A ``"owner/repo"`` GitHub id, or a local directory
            containing the same ``data/<category>/<file>.yaml`` layout --
            used by tests to point at a tiny local fixture instead of the
            real (46MB, 352-file) GitHub repository.
        ref: The git ref to fetch from, when `source` names a GitHub repo.
        categories: Restrict to these ``data/`` subdirectory names (e.g.
            ``["press_reportage"]``) instead of the full corpus.
        cache_dir: Override for the download cache directory.
        lexicon: The `wn` lexicon `dataset2` queries -- must match the
            WordNet release this corpus's ``oewn_key`` layer is aligned
            to (the corpus tracks the *current* Open English WordNet
            release, so this should track it too).
    """

    def __init__(
        self,
        source: str = _SOURCE,
        ref: str = _REF,
        categories: list[str] | None = None,
        cache_dir: Path | None = None,
        lexicon: str = "oewn:2021",
    ) -> None:
        self.source = source
        self.ref = ref
        self.categories = categories
        self.cache_dir = cache_dir
        self.lexicon = lexicon

    def load(self) -> tuple[list[Entity], WnEntitySource, list[tuple[str, str]]]:
        mentions: list[Entity] = []
        ground_truth: list[tuple[str, str]] = []
        for relpath in self._relpaths():
            doc_id = _document_id(relpath)
            raw = yaml.safe_load(_fetch_yaml_file(self.source, self.ref, relpath, self.cache_dir))
            doc_mentions, doc_ground_truth = _parse_document(doc_id, raw)
            mentions.extend(doc_mentions)
            ground_truth.extend(doc_ground_truth)
        return mentions, WnEntitySource(lexicon=self.lexicon, lang="en"), ground_truth

    def _relpaths(self) -> list[str]:
        relpaths = _list_yaml_files(self.source, self.ref, self.cache_dir)
        if self.categories is None:
            return relpaths
        categories = set(self.categories)
        return [path for path in relpaths if Path(path).parts[1] in categories]

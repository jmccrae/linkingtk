"""DaMuEL multilingual dataset loader.

DaMuEL (Kubeša & Straka, "DaMuEL: A Large Multilingual Dataset for Entity
Linking", 2023, https://arxiv.org/abs/2306.09288) links named-entity
mentions inside Wikipedia articles (53 languages) to Wikidata QIDs.
Published on LINDAT/CLARIAH-CZ (item uuid
``10e1cc03-b24b-4e41-9df9-b0fe4324ccbe``, handle ``11234/1-5047``), a
stable, freely downloadable host -- unlike TAC KBP (LDC-licensed, no free
redistribution at all -- see ``zeshel.py``'s docstring), this needed no
substitute host.

Each of DaMuEL's 53 languages is published as one bitstream,
``damuel_1.0_<lang>.tar`` (278MB-26.3GB -- English alone is 26.3GB), an
*uncompressed* outer tar containing exactly 500 ``part-NNNNN.xz`` members
in shuffled order (confirmed by inspection: streaming the first bytes of
a real language's tar yields part numbers like 247, then 325, not 0 then
1). Each part decompresses (xz) to JSON Lines, one Wikidata entity per
line, e.g.::

    {"qid": "Q504261", "lang": "da", "label": "Bräcke Kommune",
     "aliases": ["Bræcke kommun"], "description": "kommune i ...",
     "wiki": {"title": "...", "text": "...",
              "tokens": [{"start": 1, "end": 7, ...}, ...],
              "links": [{"start": 16, "end": 17, "origin": "wiki",
                         "title": "Bräcke", "qid": "Q995545"}, ...]}}

Most lines have no ``"wiki"`` key (no Wikipedia article in that
language) -- they're still valid KB entries (real negatives/candidates
for linking), just never a source of mentions. ``wiki.tokens[i]``'s
``start``/``end`` are *character* offsets into ``wiki.text``;
``wiki.links[i]``'s ``start``/``end`` are *token-index* offsets (small
integers, e.g. 16/17) that must be translated through ``tokens`` to get
a character span for ``Entity.context``. Each link's ``origin``
distinguishes the one real, hand-authored Wikipedia hyperlink per entity
mention (``"wiki"``) from several *automatically detected* extra
mentions of the same entities elsewhere in the article (``"title"``,
``"label"``, ``"alias"``, ``"redirects"``, ``"anchors"``, and lemmatized
variants -- the paper: "Wikipedia documents deliberately annotate only a
single mention for every entity present; we further automatically
detect all mentions"). This loader defaults to ``origin == "wiki"``
only -- gold, not silver, mentions -- via the ``mention_origins``
constructor argument.

Since even one language's full tar can be tens of GB, this loader
*streams* it (``tarfile.open(fileobj=..., mode="r|")``, no seek) and
stops after ``max_parts`` members -- since part order is already
shuffled, this is a random sample, not a biased prefix. This is a
partial read by design, so unlike every other loader in this package it
is **not** routed through ``fetch_cached`` (which reads a whole response
into memory before caching it -- exactly the multi-GB download this
loader exists to avoid); re-running re-streams the same small prefix,
cheap by design. There is no native train/dev/test split published for
DaMuEL (unlike AIDA-CoNLL/Zeshel) -- only the per-language and a
language-agnostic-KB (``damuel_1.0_wikidata.tar``, Wikidata
claims/named-entity-type per QID, out of scope here -- label+description
already fits every EL algorithm's ``Entity`` shape) bitstreams, so this
loader has no ``load_splits()``.
"""

from __future__ import annotations

import json
import lzma
import tarfile
from pathlib import Path
from typing import IO, Any
from urllib.request import urlopen

from linkingtk.core.entity import Entity
from linkingtk.datasets._util import fetch_cached
from linkingtk.datasets.base import DatasetLoader

_ITEM_UUID = "10e1cc03-b24b-4e41-9df9-b0fe4324ccbe"
_REST_BASE = "https://lindat.mff.cuni.cz/repository/server/api/core"
_BUNDLE_NAME = "ORIGINAL"


def _resolve_language_tar_url(language: str, cache_dir: Path | None = None) -> str:
    """The direct download URL for one language's ``damuel_1.0_<language>.tar`` bitstream.

    Two small JSON REST calls (list the item's bundles, then the
    ORIGINAL bundle's bitstreams) -- these responses are tiny (a
    listing, not tar content), so caching the whole response via
    `fetch_cached` is fine, unlike the tar itself (see module
    docstring).
    """
    bundles = json.loads(fetch_cached(f"{_REST_BASE}/items/{_ITEM_UUID}/bundles", cache_dir))
    bundle = next(b for b in bundles["_embedded"]["bundles"] if b["name"] == _BUNDLE_NAME)
    bitstreams_url = bundle["_links"]["bitstreams"]["href"]
    bitstreams = json.loads(fetch_cached(bitstreams_url, cache_dir))
    filename = f"damuel_1.0_{language}.tar"
    bitstream = next(b for b in bitstreams["_embedded"]["bitstreams"] if b["name"] == filename)
    url: str = bitstream["_links"]["content"]["href"]
    return url


def _open_source(source: str) -> IO[bytes]:
    if source.startswith(("http://", "https://")):
        response: IO[bytes] = urlopen(source)
        return response
    return Path(source).open("rb")


def _iter_entities(source: str, max_parts: int) -> list[dict[str, Any]]:
    """The parsed JSON entity lines from the first `max_parts` `.xz` members of `source`'s tar.

    Streams (``mode="r|"``, no seek) and stops as soon as `max_parts`
    parts have been read, closing the underlying connection early --
    the whole reason this works for multi-GB language tars. Part order
    inside the tar is already shuffled (see module docstring), so this
    is a random sample, not a biased prefix.
    """
    entities: list[dict[str, Any]] = []
    stream = _open_source(source)
    try:
        with tarfile.open(fileobj=stream, mode="r|") as tar:
            parts_read = 0
            for member in tar:
                if not member.isfile() or not member.name.endswith(".xz"):
                    continue
                extracted = tar.extractfile(member)
                if extracted is None:
                    continue
                data = lzma.decompress(extracted.read())
                for line in data.splitlines():
                    if line:
                        entities.append(json.loads(line))
                parts_read += 1
                if parts_read >= max_parts:
                    break
    finally:
        stream.close()
    return entities


def _mention_spans(
    entity: dict[str, Any], mention_origins: tuple[str, ...]
) -> list[tuple[int, int, str]]:
    """`(char_start, char_end, target_qid)` for each qualifying link in a `"wiki"`-bearing entity.

    Translates each link's token-index span (`wiki.links[i].start`/
    `.end`) through `wiki.tokens` to a character span, since `Entity`'s
    `context` convention is character offsets, not token indices.
    """
    wiki = entity.get("wiki")
    if wiki is None:
        return []
    tokens = wiki["tokens"]
    spans: list[tuple[int, int, str]] = []
    for link in wiki["links"]:
        if link["origin"] not in mention_origins:
            continue
        # A real hyperlink can point at a page with no resolved Wikidata QID
        # at all (confirmed on real data, e.g. Danish shards) -- the
        # equivalent of AIDA-CoNLL's NIL mentions; skip rather than error.
        target_qid = link.get("qid")
        if target_qid is None:
            continue
        start_token, end_token = link["start"], link["end"]
        if end_token <= start_token or end_token > len(tokens):
            continue
        char_start = tokens[start_token]["start"]
        char_end = tokens[end_token - 1]["end"]
        spans.append((char_start, char_end, target_qid))
    return spans


class DamuelDataset(DatasetLoader):
    """DaMuEL: Wikipedia mentions (53 languages) linked to Wikidata QIDs.

    See the module docstring for the real archive format and why this
    loader streams and samples rather than materializing a full
    language.

    Args:
        language: A DaMuEL language code (e.g. ``"en"``, ``"da"`` -- the
            two-letter codes the source itself uses).
        max_parts: How many of the language's 500 shuffled ``.xz``
            shards to stream and parse. A mention's target entity can
            land in *any* of the 500 parts (independent of which part
            its source article is in), so `ground_truth` density scales
            roughly with `max_parts / 500`, not with the mention count
            alone -- confirmed on real data: 2 English parts (~29s)
            resolved 2571 real mention/target pairs out of a much larger
            KB (114669 entities), while 2 Danish parts (a far smaller
            language) resolved only 41. Low-resource languages need a
            larger `max_parts` for a non-empty `ground_truth` (e.g. `wo`,
            the smallest language, resolved 0 pairs at `max_parts=3`).
            Defaults to ``2`` -- fast and already useful for a
            higher-resource language; raise it for denser ground truth
            or a lower-resource one. The full corpus is impractically
            large to materialize in memory in one `load()` call.
        mention_origins: Which `wiki.links[].origin` values count as
            real mentions. Defaults to `("wiki",)` -- DaMuEL's one real,
            hand-authored hyperlink per entity per article, not its
            additional automatically-detected mentions (see module
            docstring). Widen this (e.g. add `"anchors"`) for more
            mentions at the cost of silver- rather than gold-standard
            precision.
        source: Override for the language tar's location -- a local
            path or `http(s)://` URL, used by tests to point at a tiny
            local fixture instead of resolving and streaming the real
            (278MB-26GB) release.
        cache_dir: Override for the small bitstream-listing-lookup cache
            directory. Ignored if `source` is given directly (no lookup
            needed), and never covers the tar stream itself, which is
            never cached (see module docstring).

    Unlike [AidaConllDataset][linkingtk.datasets.aida_conll.AidaConllDataset]
    and [ZeshelDataset][linkingtk.datasets.zeshel.ZeshelDataset], DaMuEL
    publishes no native train/dev/test split, so there's no
    `load_splits()` here.
    """

    def __init__(
        self,
        language: str = "en",
        max_parts: int = 2,
        mention_origins: tuple[str, ...] = ("wiki",),
        source: str | None = None,
        cache_dir: Path | None = None,
    ) -> None:
        self.language = language
        self.max_parts = max_parts
        self.mention_origins = mention_origins
        self.source = source
        self.cache_dir = cache_dir

    def load(self) -> tuple[list[Entity], list[Entity], list[tuple[str, str]]]:
        source = self.source
        if source is None:
            source = _resolve_language_tar_url(self.language, self.cache_dir)
        entities = _iter_entities(source, self.max_parts)

        kb: list[Entity] = []
        kb_ids: set[str] = set()
        mentions: list[Entity] = []
        raw_ground_truth: list[tuple[str, str]] = []

        for entity in entities:
            qid = entity["qid"]
            if qid not in kb_ids:
                kb_ids.add(qid)
                # Some entities have no Wikidata label in this language at all --
                # only aliases, or only a description (confirmed on real data:
                # e.g. Danish shards carry entities with just qid/lang/description).
                label = [entity["label"]] if "label" in entity else []
                labels: list[str | tuple[str, str]] = [*label, *entity.get("aliases", [])]
                kb.append(Entity(id=qid, labels=labels, description=entity.get("description")))

            wiki = entity.get("wiki")
            if wiki is None:
                continue
            text = wiki["text"]
            for char_start, char_end, target_qid in _mention_spans(entity, self.mention_origins):
                mention_id = f"damuel:{qid}:{char_start}:{char_end}"
                mentions.append(
                    Entity(
                        id=mention_id,
                        labels=[text[char_start:char_end]],
                        context=(text, char_start, char_end),
                    )
                )
                raw_ground_truth.append((mention_id, target_qid))

        ground_truth = [(mid, tid) for mid, tid in raw_ground_truth if tid in kb_ids]
        resolved_mention_ids = {mid for mid, _ in ground_truth}
        mentions = [m for m in mentions if m.id in resolved_mention_ids]

        return mentions, kb, ground_truth

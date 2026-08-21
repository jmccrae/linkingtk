"""UFSAC sense-annotated corpus loader.

[UFSAC](https://github.com/getalp/UFSAC) (Vial et al., 2018) unifies 16+
Word Sense Disambiguation corpora (SemCor, WordNet Gloss Tagged, MASC,
OMSTI, the SensEval/SemEval all-words and lexical-sample tasks, ...) into
one shared XML schema, all tagged with WordNet 3.0 sense keys.

Unlike this package's other real-world dataset loaders, UFSAC has no
stable, programmatically fetchable per-corpus URL: the whole collection is
distributed as a single Google Drive archive bundling every corpus's XML
together (see the repository's README), not the kind of per-file HTTPS
link `fetch_cached` (or any other loader in this package) can point at.
So unlike e.g. [AidaConllDataset][linkingtk.datasets.aida_conll.AidaConllDataset],
this loader doesn't fetch UFSAC's data itself -- `source` must be a path
or URL to a single corpus's XML (or ``.xml.xz``, UFSAC's own distributed
compression) file that you've already extracted from that archive, e.g.
``UfsacDataset("~/Downloads/ufsac-public-2.1/semcor.xml")``. The parser
itself handles UFSAC's one shared schema, so this loader works for any of
the 16+ corpora, not just SemCor.

UFSAC's sense keys are WordNet 3.0 sense keys (e.g.
``"group%1:03:00::"``), not `wn` synset ids, so ``dataset2`` (a
[WnEntitySource][linkingtk.sources.wn.WnEntitySource]) needs each one
translated via [sensekey_to_synset_id][linkingtk.sources.wn.sensekey_to_synset_id]
before it can serve as a ground-truth id `dataset2.get()`/`.search()` can
resolve -- unlike [SemCorDataset][linkingtk.datasets.semcor.SemCorDataset],
whose ``oewn_key`` layer is already in `wn`'s own id format.
"""

from __future__ import annotations

import lzma
from io import BytesIO
from pathlib import Path
from xml.etree import ElementTree

from linkingtk.core.entity import Entity
from linkingtk.datasets._util import fetch_cached
from linkingtk.datasets.base import DatasetLoader
from linkingtk.sources.wn import WnEntitySource, sensekey_to_synset_id


def _read(source: str, cache_dir: Path | None) -> bytes:
    if source.startswith(("http://", "https://", "file://")):
        data = fetch_cached(source, cache_dir)
    else:
        data = Path(source).read_bytes()
    return lzma.decompress(data) if source.endswith(".xz") else data


def _parse(data: bytes) -> tuple[list[Entity], list[tuple[str, str]]]:
    """Parse UFSAC's ``<corpus><document><paragraph><sentence><word/></sentence>...`` schema.

    Streams via ``iterparse`` (some corpora, e.g. OMSTI's ~1M sense-tagged
    instances, are too large to comfortably hold as one DOM) and buffers
    only one sentence's words at a time -- UFSAC has no raw untokenized
    sentence text, so each mention's ``context`` is reconstructed by
    joining that sentence's ``surface_form``s with spaces.

    Each mention's ``labels`` is its WordNet ``lemma`` attribute, not its
    ``surface_form`` -- candidate generation (``ExactMatch`` querying an
    ``EntitySource`` by label) needs the dictionary form (e.g. "catch"),
    since an inflected surface form (e.g. "caught") won't exact-match
    WordNet's own lemma index. The surface form is still recoverable from
    ``context``'s span for anything that needs the literal text (e.g.
    [GlossBertEncoder][linkingtk.algorithms.wsd.glossbert.GlossBertEncoder]'s
    gloss-text formatting).

    A ``wn30_key`` can list more than one sense key, ``;``-separated --
    some gold-standard WSD instances genuinely accept several correct
    answers (not a compound-word artifact). Every key becomes its own
    ``ground_truth`` row sharing that mention's id, so
    [Evaluator.evaluate][linkingtk.eval.evaluator.Evaluator.evaluate]'s
    multi-answer support (a prediction matching *any* of them counts as
    correct) applies -- rare overall (0.9% of SemEval-2007's instances)
    but not always (18% of SemEval-2015's).

    Each mention's id is the word's own ``id`` attribute when present
    (e.g. ``"d000.s000.t000"``, the original Raganato et al. instance id
    -- present on the ``raganato_*.xml`` corpora, so a system's output can
    be converted straight back to their submission format for
    cross-checking against their own ``Scorer.java``), falling back to a
    positional ``ufsac:{document}:{sentence}:{index}`` id for corpora that
    don't carry one (e.g. UFSAC's own non-Raganato ``semcor.xml``).

    Each mention also carries its word's own Penn Treebank ``pos`` tag
    (e.g. ``"NN"``, ``"VBD"``) as ``properties["pos"]`` when present --
    unused by [GlossBertLinker][linkingtk.algorithms.wsd.glossbert.GlossBertLinker]
    (whose own paper deliberately doesn't restrict candidates by POS), but
    needed to replicate
    [EwiserLinker][linkingtk.algorithms.wsd.ewiser.EwiserLinker]'s
    reference's own candidate generation, which restricts each mention to
    its tagged part of speech's senses (confirmed directly against
    ``ewiser/fairseq_ext/data/wsd_dataset.py``'s ``lemma_pos_to_possible_senses``).
    """
    mentions: list[Entity] = []
    ground_truth: list[tuple[str, str]] = []
    doc_id = ""
    sent_id = ""
    sent_index = 0
    words: list[tuple[str, str, list[str], str | None, str | None]] = []

    def flush_sentence() -> None:
        text = " ".join(surface for surface, _lemma, _keys, _id, _pos in words)
        offset = 0
        for index, (surface, lemma, sense_keys, instance_id, pos) in enumerate(words):
            start = offset
            end = offset + len(surface)
            offset = end + 1
            if sense_keys:
                mention_id = instance_id or f"ufsac:{doc_id}:{sent_id}:{index}"
                properties = {"pos": pos} if pos else {}
                mentions.append(
                    Entity(
                        id=mention_id,
                        labels=[lemma],
                        context=(text, start, end),
                        properties=properties,
                    )
                )
                ground_truth.extend((mention_id, sense_key) for sense_key in sense_keys)
        words.clear()

    for event, elem in ElementTree.iterparse(BytesIO(data), events=("start", "end")):
        if event == "start" and elem.tag == "document":
            doc_id = elem.get("id", "")
        elif event == "start" and elem.tag == "sentence":
            sent_id = elem.get("id") or str(sent_index)
            sent_index += 1
        elif event == "end" and elem.tag == "word":
            surface = elem.get("surface_form", "")
            lemma = elem.get("lemma", surface)
            key_attr = elem.get("wn30_key")
            words.append(
                (
                    surface,
                    lemma,
                    key_attr.split(";") if key_attr else [],
                    elem.get("id"),
                    elem.get("pos"),
                )
            )
            elem.clear()
        elif event == "end" and elem.tag == "sentence":
            flush_sentence()
            elem.clear()
        elif event == "end" and elem.tag in ("paragraph", "document"):
            elem.clear()

    return mentions, ground_truth


class UfsacDataset(DatasetLoader):
    """A UFSAC-format sense-annotated corpus (SemCor, OMSTI, SensEval/SemEval, ...).

    See the module docstring for why `source` must be a corpus file
    you've already downloaded, and why ``dataset2``'s ids need sense-key
    translation unlike [SemCorDataset][linkingtk.datasets.semcor.SemCorDataset].

    Args:
        source: A local path or ``http(s)://``/``file://`` URL to a single
            UFSAC corpus's ``.xml`` or ``.xml.xz`` file (e.g.
            ``semcor.xml``, ``omsti.xml`` -- see UFSAC's README for the
            full list of per-corpus filenames).
        cache_dir: Override for the download cache directory. Ignored for
            a local `source`.
        lexicon: The `wn` lexicon `dataset2` queries and sense keys are
            resolved against. Defaults to ``"omw-en:1.4"`` (OMW's English
            WordNet, built directly from WordNet 3.0 -- the release
            UFSAC's sense keys are aligned to).

    Mentions whose sense key doesn't resolve to a synset in `lexicon` are
    dropped entirely (from both `dataset1` and `ground_truth`), same as
    AIDA-CoNLL's NIL-mention convention -- rare in practice, but WordNet
    3.0 sense keys and `lexicon`'s own sense inventory can disagree at the
    margins.
    """

    def __init__(
        self,
        source: str,
        cache_dir: Path | None = None,
        lexicon: str = "omw-en:1.4",
    ) -> None:
        self.source = source
        self.cache_dir = cache_dir
        self.lexicon = lexicon

    def load(self) -> tuple[list[Entity], WnEntitySource, list[tuple[str, str]]]:
        mentions, raw_ground_truth = _parse(_read(self.source, self.cache_dir))
        resolved: dict[str, str | None] = {}
        ground_truth: list[tuple[str, str]] = []
        for mention_id, sense_key in raw_ground_truth:
            if sense_key not in resolved:
                resolved[sense_key] = sensekey_to_synset_id(sense_key, lexicon=self.lexicon)
            synset_id = resolved[sense_key]
            if synset_id is not None:
                ground_truth.append((mention_id, synset_id))
        resolved_mention_ids = {mention_id for mention_id, _ in ground_truth}
        mentions = [entity for entity in mentions if entity.id in resolved_mention_ids]
        return mentions, WnEntitySource(lexicon=self.lexicon, lang="en"), ground_truth

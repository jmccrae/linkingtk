"""Private, plain-numpy (torch-free) helpers for
[KDCoELinker][linkingtk.algorithms.ea.kdcoe.KDCoELinker].

Ported from OpenEA's reference implementation
(https://github.com/nju-websoft/OpenEA -- ``approaches/kdcoe.py``'s
``_get_desc_input``/``_get_local_name_by_name_triple``). Independently
testable without ``torch`` installed -- see ``_kdcoe_torch.py`` for the
description-encoder network and training-step functions.
"""

from __future__ import annotations

import re
import zipfile
from io import BytesIO
from typing import TYPE_CHECKING, Any

import numpy as np

from linkingtk.core.entity import Entity, label_texts
from linkingtk.datasets._util import fetch_cached
from linkingtk.utils.graph import Triple

if TYPE_CHECKING:
    from pathlib import Path

    import numpy.typing as npt

_PUNCTUATION_RE = re.compile(r"""[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~]+""")
_DESCRIPTION_PREDICATE_RE = re.compile("escription")


def extract_description_text(
    entities: list[Entity],
    attribute_triples: list[Triple],
) -> dict[str, str]:
    """Each entity's description text, falling back to its own label.

    Ports ``_get_desc_input``'s entity-description selection: an entity's
    text is the first attribute-triple value whose predicate URI contains
    ``"escription"`` (catching ``description``/``Description``, e.g.
    ``dc:description``, ``dbo:description``) -- the same substring check
    OpenEA's own code uses. Entities with no such triple fall back to
    their own local name (``_get_local_name_by_name_triple``'s fallback --
    this repo's dataset loaders already populate ``Entity.labels`` with
    exactly that text, see ``linkingtk.datasets.openea_native``, so no
    separate name-triple lookup is needed here).

    This is why the description pathway never strictly needs
    ``attribute_triples`` to produce *some* text for every entity -- see
    [KDCoELinker][linkingtk.algorithms.ea.kdcoe.KDCoELinker]'s docstring.

    Args:
        entities: Every entity needing description text (both KGs,
            concatenated by the caller).
        attribute_triples: ``(entity_id, predicate, value)`` triples (both
            KGs, concatenated by the caller) -- values are plain text
            (already RDF-literal-stripped by the dataset loader).

    Returns:
        ``entity_id -> description text`` (untokenized -- see
        [tokenize_description][linkingtk.algorithms.ea._kdcoe_text.tokenize_description]).
    """
    descriptions: dict[str, str] = {}
    for entity_id, predicate, value in attribute_triples:
        if entity_id not in descriptions and _DESCRIPTION_PREDICATE_RE.search(predicate):
            descriptions[entity_id] = value

    result: dict[str, str] = {}
    for entity in entities:
        if entity.id in descriptions:
            result[entity.id] = descriptions[entity.id]
        else:
            texts = label_texts(entity)
            result[entity.id] = texts[0] if texts else ""
    return result


def tokenize_description(text: str) -> list[str]:
    """Lowercased, punctuation-stripped whitespace tokens.

    Ports the ``string.punctuation`` regex replace + whitespace split
    OpenEA applies to both description and local-name text before word-id
    lookup.
    """
    stripped = _PUNCTUATION_RE.sub("", text).lower()
    return [token for token in stripped.split() if token]


def build_word_ids(
    tokens_by_entity: dict[str, list[str]],
    word_to_index: dict[str, int],
    default_desc_length: int,
    oov_id: int,
) -> dict[str, npt.NDArray[np.int64]]:
    """Each entity's tokens -> a fixed-length array of word ids.

    Ports ``lookup_and_padding``: truncates to ``default_desc_length``
    words, padding with ``oov_id`` if shorter. A token absent from
    ``word_to_index`` also maps to ``oov_id`` -- the same handling OpenEA
    gives words unseen in the pretrained vectors (see
    [load_fasttext_vectors][linkingtk.algorithms.ea._kdcoe_text.load_fasttext_vectors]).

    Args:
        tokens_by_entity: Per-entity token lists, e.g. from tokenizing
            [extract_description_text][linkingtk.algorithms.ea._kdcoe_text.extract_description_text]'s
            output.
        word_to_index: Vocabulary word -> row index (e.g. into a
            pretrained embedding table).
        default_desc_length: Fixed output length. OpenEA's published
            value is ``4``.
        oov_id: Id used for padding and out-of-vocabulary words --
            conventionally one past the real vocabulary's last index.

    Returns:
        ``entity_id -> (default_desc_length,) int64 array``.
    """
    result: dict[str, npt.NDArray[np.int64]] = {}
    for entity_id, tokens in tokens_by_entity.items():
        ids = [word_to_index.get(token, oov_id) for token in tokens[:default_desc_length]]
        ids += [oov_id] * (default_desc_length - len(ids))
        result[entity_id] = np.array(ids, dtype=np.int64)
    return result


def load_fasttext_vectors(
    zip_url: str,
    vocabulary: set[str],
    cache_dir: Path | None = None,
) -> dict[str, npt.NDArray[np.floating[Any]]]:
    """Pretrained word vectors for ``vocabulary``'s words only, from a fastText ``.vec.zip``.

    Fetches ``zip_url`` via ``fetch_cached`` (``linkingtk.datasets._util``
    -- cached on disk after the first call; ``file://`` urls, used by
    tests to point at a tiny local fixture, are never cached, matching
    ``fetch_cached``'s existing behavior) and streams the archived
    ``.vec`` member line-by-line (``zipfile.ZipFile(...).open(member)``)
    rather than decoding the whole decompressed text at once -- for the
    real fastText file (``wiki-news-300d-1M.vec``, ~1M words, ~2.3GB
    decompressed) this keeps peak new memory bounded by ``len(vocabulary)``
    rather than the file's full vocabulary.

    Args:
        zip_url: URL of a zip containing exactly one ``.vec`` member in
            fastText's plain-text format (first line ``"<count> <dim>"``,
            then one ``"word f1 f2 ... fN"`` line per word).
        vocabulary: Words to keep; every other row is skipped without
            being parsed into an array.
        cache_dir: Passed through to ``fetch_cached``.

    Returns:
        ``word -> (dim,) float32 array``, restricted to words that are
        both in ``vocabulary`` and the file. Words in ``vocabulary`` but
        absent from the file are simply omitted -- callers treat a
        missing word as out-of-vocabulary, same as
        [build_word_ids][linkingtk.algorithms.ea._kdcoe_text.build_word_ids]'s
        ``oov_id`` fallback.
    """
    archive_bytes = fetch_cached(zip_url, cache_dir)
    vectors: dict[str, npt.NDArray[np.floating[Any]]] = {}
    with zipfile.ZipFile(BytesIO(archive_bytes)) as archive:
        member_name = next(name for name in archive.namelist() if name.endswith(".vec"))
        with archive.open(member_name) as member:
            for line_number, raw_line in enumerate(member):
                if line_number == 0:
                    continue  # header: "<count> <dim>"
                parts = raw_line.decode("utf-8").split()
                if not parts:
                    continue
                word = parts[0]
                if word in vocabulary:
                    vectors[word] = np.array(parts[1:], dtype=np.float32)
    return vectors

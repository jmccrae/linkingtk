"""Private, plain-numpy (torch-free) helpers for
[AttrELinker][linkingtk.algorithms.ea.attre.AttrELinker].

Ported from OpenEA's reference implementation
(https://github.com/nju-websoft/OpenEA -- ``approaches/attre.py``'s
``formatting_attr_triples``/``clean_attribute_triples``,
``modules/train/batch.py``'s ``generate_neg_attribute_triples``).
Independently testable without ``torch`` installed -- see
``_attre_torch.py`` for the character-composition network and
training-step functions.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt


def clean_attribute_value(value: str) -> str:
    """Strip parenthetical content and light punctuation from an attribute value.

    Ports ``clean_attribute_triples``'s per-value cleaning exactly, except
    its final ``v.split('"')[0]`` step -- this repo's dataset loaders
    already strip RDF literal quoting upstream (see
    ``linkingtk.datasets._util.parse_literal_value``), so by the time a
    value reaches here it's already unquoted plain text.
    """
    value = value.split("(")[0].rstrip(" ")
    for ch in ".(),":
        value = value.replace(ch, "")
    return value.replace("_", " ").replace("-", " ")


def select_char_vocabulary(values: list[str], frequency_threshold: float) -> dict[str, int]:
    """Characters appearing in at least ``frequency_threshold`` of all (distinct-value) occurrences.

    Ports ``formatting_attr_triples``'s character selection: counts each
    character's occurrences across the *distinct* values only (a value
    repeated across many triples counts once), keeps characters whose
    share of the total count clears ``frequency_threshold`` (OpenEA's
    published value is ``0.0001``). Ids start at ``1`` -- ``0`` is
    reserved for padding and for characters this filters out, matching
    ``formatting_attr_triples``'s ``char_id_dict`` numbering.

    **Deviation**: OpenEA builds its character list via a plain
    ``list(set(...))``, whose order (and therefore each kept character's
    exact id) isn't reproducible across runs/Python versions. This sorts
    the kept characters first for determinism -- same precedent as
    ``_jape_training.py``'s ``generate_training_pairs``. This doesn't
    change *which* characters are kept, only their arbitrary id numbering.

    Args:
        values: Attribute values, e.g. already cleaned via
            [clean_attribute_value][linkingtk.algorithms.ea._attre_text.clean_attribute_value].
            Duplicates are collapsed before counting.
        frequency_threshold: Minimum share of total character occurrences
            (across distinct values) required to keep a character.

    Returns:
        ``character -> id`` (``1``-based).
    """
    counts: dict[str, int] = defaultdict(int)
    for value in set(values):
        for ch in value:
            counts[ch] += 1
    total = sum(counts.values())
    if total == 0:
        return {}
    kept = sorted(ch for ch, n in counts.items() if n / total >= frequency_threshold)
    return {ch: index + 1 for index, ch in enumerate(kept)}


def build_value_char_ids(
    values: list[str],
    char_to_id: dict[str, int],
    literal_len: int,
) -> dict[str, npt.NDArray[np.int64]]:
    """Each distinct value's characters -> a fixed-length array of character ids.

    Ports ``formatting_attr_triples``'s ``value_char_ids_dict`` building:
    truncates to ``literal_len`` characters, pads (and maps
    out-of-vocabulary characters) to id ``0``.

    Args:
        values: Attribute values (e.g. already cleaned).
        char_to_id: From
            [select_char_vocabulary][linkingtk.algorithms.ea._attre_text.select_char_vocabulary].
        literal_len: Fixed output length. OpenEA's published value is
            ``5``.

    Returns:
        ``value -> (literal_len,) int64 array``, one entry per distinct
        value in ``values``.
    """
    result: dict[str, npt.NDArray[np.int64]] = {}
    for value in set(values):
        ids = [0] * literal_len
        for i in range(min(len(value), literal_len)):
            ids[i] = char_to_id.get(value[i], 0)
        result[value] = np.array(ids, dtype=np.int64)
    return result


def sample_negative_attribute_triples(
    positive_triples: npt.NDArray[np.int64],
    entity_pool: npt.NDArray[np.int64],
    real_triples: set[tuple[int, int, int]],
    rng: np.random.Generator,
    max_tries: int = 10,
) -> npt.NDArray[np.int64]:
    """Corrupt the entity endpoint of each positive ``(entity, attribute, value)`` triple.

    Ports OpenEA's ``generate_neg_attribute_triples``. Unlike
    relation-triple negative sampling
    (``_iptranse_training.py``'s ``sample_negative_triples``, a
    head-or-tail coin flip), attribute-triple negatives **always** corrupt
    the entity -- the attribute predicate and value are never corrupted,
    matching the reference exactly (there's no pool of "other valid
    attributes/values" to draw a corruption from the way there's a pool of
    "other valid entities").

    Args:
        positive_triples: ``(n, 3)`` int64 array of ``(entity, attribute,
            value)`` ids.
        entity_pool: That triple's own KG's entity ids to sample
            replacements from.
        real_triples: Set of ``(entity, attribute, value)`` id tuples to
            avoid reproducing. Retries up to ``max_tries``, then accepts
            the last draw regardless.
        rng: Random generator, for reproducibility.
        max_tries: Retries per row before giving up.

    Returns:
        ``(n, 3)`` int64 array of corrupted triples, same shape as
        ``positive_triples``.
    """
    negatives = positive_triples.copy()
    for i in range(len(positive_triples)):
        e, a, v = (int(x) for x in positive_triples[i])
        for _ in range(max_tries):
            e = int(rng.choice(entity_pool))
            if (e, a, v) not in real_triples:
                break
        negatives[i] = (e, a, v)
    return negatives

"""Torch-free bootstrap-alignment helpers for
[IMUSELinker][linkingtk.algorithms.ea.imuse.IMUSELinker].

Ports OpenEA's ``interactive_model`` and its supporting functions
(``src/openea/approaches/imuse.py``) -- string-similarity-driven discovery
of an initial entity alignment from attribute triples alone, with no
seed pairs. Kept separate from ``_imuse_torch.py`` (which builds/consumes
PyTorch tensors) so this stays independently testable without ``torch``
installed, matching this repo's established ``_*_text.py`` convention.

``rapidfuzz`` is imported lazily, inside ``levenshtein_ratio`` itself, not
at module top -- matching the convention every ``_*_torch.py`` helper in
this package already uses for ``torch``: nowhere is an optional
third-party dependency imported at module scope, since
``linkingtk.algorithms.ea``'s ``__init__.py`` eagerly imports every linker
module (including this one's caller, ``imuse.py``), and importing an
optional dependency there would break ``from linkingtk.algorithms.ea
import <anything>`` for users who haven't installed the ``kge`` extra.
"""

from __future__ import annotations

from collections import defaultdict

from linkingtk.utils.graph import Triple


def levenshtein_ratio(a: str, b: str) -> float:
    """Normalized indel-based string similarity in ``[0, 1]``.

    Thin wrapper over ``rapidfuzz.distance.Indel.normalized_similarity``
    -- **not** ``rapidfuzz.distance.Levenshtein`` (that one uses
    substitution cost ``1`` and normalizes by ``max(len_a, len_b)``, a
    different formula). ``Indel`` disallows substitutions (a substitution
    costs a delete + an insert, i.e. `2`) and normalizes by
    ``len_a + len_b`` -- confirmed by hand-computation to be exactly the
    formula ``python-Levenshtein``'s ``.ratio()`` uses, which is what
    OpenEA's own ``compute_two_values_similarity``/
    ``get_aligned_attr_pair_by_name_similarity`` call, so this reproduces
    the reference source's actual similarity values.
    """
    from rapidfuzz.distance import Indel

    return float(Indel.normalized_similarity(a, b))


def local_name(uri: str) -> str:
    """The last ``/``-separated segment of a predicate URI.

    Ports ``id_attr_dict[a].split('/')[-1]`` (``imuse.py``'s
    ``get_aligned_attr_pair_by_name_similarity``).
    """
    return uri.split("/")[-1]


def align_attributes_by_name(
    attribute_triples1: list[Triple],
    attribute_triples2: list[Triple],
    threshold: float,
    top_k: int,
) -> set[tuple[str, str]]:
    """Greedily pair KG1/KG2 attribute predicates by local-name similarity.

    Ports ``get_aligned_attr_pair_by_name_similarity``: for every KG1
    predicate, keep its single best-matching KG2 predicate if the match
    clears ``threshold`` and that KG2 predicate hasn't already been
    claimed by an earlier KG1 predicate (dedup happens *after* scanning
    every KG2 candidate, not on every improving candidate -- this
    function's own OpenEA implementation already gets this right, unlike
    ``align_entity_by_attributes``/``align_attribute_by_entities`` below,
    see their docstrings). Finally keeps only the ``top_k`` resulting
    pairs by combined (both-KG) triple count.
    """
    predicates1 = sorted({predicate for _, predicate, _ in attribute_triples1})
    predicates2 = sorted({predicate for _, predicate, _ in attribute_triples2})

    aligned: set[tuple[str, str]] = set()
    claimed2: set[str] = set()
    for predicate1 in predicates1:
        name1 = local_name(predicate1)
        best_predicate2: str | None = None
        best_sim = threshold
        for predicate2 in predicates2:
            sim = levenshtein_ratio(name1, local_name(predicate2))
            if sim > best_sim:
                best_predicate2 = predicate2
                best_sim = sim
        if best_predicate2 is not None and best_predicate2 not in claimed2:
            aligned.add((predicate1, best_predicate2))
            claimed2.add(best_predicate2)

    if not aligned:
        return aligned

    counts1: dict[str, int] = defaultdict(int)
    for _, predicate, _ in attribute_triples1:
        counts1[predicate] += 1
    counts2: dict[str, int] = defaultdict(int)
    for _, predicate, _ in attribute_triples2:
        counts2[predicate] += 1

    ranked = sorted(aligned, key=lambda pair: counts1[pair[0]] + counts2[pair[1]], reverse=True)
    return set(ranked[:top_k])


def _grouped_attribute_values(
    attribute_triples: list[Triple],
    *,
    group_by_entity: bool,
    restrict_to: set[str],
) -> tuple[dict[str, set[str]], dict[tuple[str, str], str]]:
    """First value per ``(key, other)``, restricted to ``restrict_to`` on the other axis.

    ``group_by_entity=True`` groups by entity, restricted to a set of
    predicates -- ports ``filter_by_aligned_attributes``.
    ``group_by_entity=False`` groups by predicate, restricted to a set of
    entities -- ports ``filter_by_aligned_entities``, OpenEA's own
    near-duplicate of ``filter_by_aligned_attributes`` with the roles
    swapped; one generic helper covers both call sites instead of
    duplicating the loop. Either way, only the first triple seen for a
    given ``(key, other)`` is kept (matches OpenEA's own
    ``if (e, a) not in ent_attr_value_dict``).
    """
    grouped: dict[str, set[str]] = defaultdict(set)
    value_by_pair: dict[tuple[str, str], str] = {}
    for entity, predicate, value in attribute_triples:
        key, other = (entity, predicate) if group_by_entity else (predicate, entity)
        if other in restrict_to and (key, other) not in value_by_pair:
            value_by_pair[(key, other)] = value
            grouped[key].add(other)
    return grouped, value_by_pair


def _value_index(value_by_pair: dict[tuple[str, str], str]) -> dict[str, set[str]]:
    """``normalized value -> entities`` inverted index, for candidate pruning.

    OpenEA brute-forces every ``(e1, e2)`` pair (up to ~15K x 15K at this
    dataset's scale), parallelized across 8 processes -- not tractable in
    a single Python process. Since two entities can only score above `0`
    similarity if they share a value on some aligned predicate (an empty
    intersection leaves ``sim_cnt == 0``, so the score never clears any
    positive threshold), only entities sharing at least one attribute
    value (after casefolding) can ever match -- this index restricts
    scoring to exactly those candidates. Trade-off: misses pairs whose
    *only* signal is a near-but-not-exact value match (e.g. minor spelling
    variation); if a live benchmark underperforms, widening this (e.g. a
    token/n-gram fallback) is the first thing worth trying.
    """
    index: dict[str, set[str]] = defaultdict(set)
    for (entity, _predicate), value in value_by_pair.items():
        index[value.casefold()].add(entity)
    return index


def _best_match(
    entity1: str,
    candidates2: set[str],
    aligned_pairs: set[tuple[str, str]],
    entity_predicates1: dict[str, set[str]],
    entity_predicates2: dict[str, set[str]],
    value_by_pair1: dict[tuple[str, str], str],
    value_by_pair2: dict[tuple[str, str], str],
    threshold: float,
) -> str | None:
    predicates1 = entity_predicates1[entity1]
    best_entity2: str | None = None
    best_sim = threshold
    for entity2 in candidates2:
        predicates2 = entity_predicates2[entity2]
        total = 0.0
        count = 0
        for predicate1, predicate2 in aligned_pairs:
            if predicate1 in predicates1 and predicate2 in predicates2:
                total += levenshtein_ratio(
                    value_by_pair1[(entity1, predicate1)], value_by_pair2[(entity2, predicate2)]
                )
                count += 1
        sim = total / count if count else 0.0
        if sim > best_sim:
            best_entity2 = entity2
            best_sim = sim
    return best_entity2


def align_entities_by_attribute_values(
    attribute_triples1: list[Triple],
    attribute_triples2: list[Triple],
    aligned_attr_pairs: set[tuple[str, str]],
    threshold: float,
) -> set[tuple[str, str]]:
    """Greedily pair entities by average similarity across aligned-predicate values.

    Ports ``align_entity_by_attributes``/``run_one_ea``, single-process
    (OpenEA parallelizes this across 8 processes; not needed once the
    O(n1*n2) brute force is replaced by the indexed candidate pruning
    described in ``_value_index``'s docstring), with one deliberate
    correctness fix: OpenEA's own ``run_one_ea`` accepts a candidate
    *inside* the inner "scan every e2" loop, on every improving match --
    so a KG1 entity's earlier, since-superseded best match stays in the
    result set and keeps its claimed KG2 entity permanently unavailable,
    even after a better match for the *same* KG1 entity is found later in
    the same scan. This is inconsistent with the textbook-correct version
    of the identical greedy-accept pattern just a few dozen lines away in
    ``get_aligned_attr_pair_by_name_similarity`` (``align_attributes_by_name``
    above) -- treated as an unintentional bug rather than a deliberate
    design choice, so this port accepts only the single best match found
    after scanning all of a KG1 entity's candidates, not every improving
    one along the way.
    """
    if not aligned_attr_pairs:
        return set()

    predicates1 = {p1 for p1, _ in aligned_attr_pairs}
    predicates2 = {p2 for _, p2 in aligned_attr_pairs}
    entity_predicates1, value_by_pair1 = _grouped_attribute_values(
        attribute_triples1, group_by_entity=True, restrict_to=predicates1
    )
    entity_predicates2, value_by_pair2 = _grouped_attribute_values(
        attribute_triples2, group_by_entity=True, restrict_to=predicates2
    )
    index2 = _value_index(value_by_pair2)

    aligned: set[tuple[str, str]] = set()
    claimed2: set[str] = set()
    for entity1 in sorted(entity_predicates1):
        own_values = {
            value_by_pair1[(entity1, predicate1)] for predicate1 in entity_predicates1[entity1]
        }
        candidates2 = set()
        for value in own_values:
            candidates2 |= index2.get(value.casefold(), set())
        candidates2 -= claimed2
        if not candidates2:
            continue
        match = _best_match(
            entity1,
            candidates2,
            aligned_attr_pairs,
            entity_predicates1,
            entity_predicates2,
            value_by_pair1,
            value_by_pair2,
            threshold,
        )
        if match is not None:
            aligned.add((entity1, match))
            claimed2.add(match)
    return aligned


def align_attributes_by_entities(
    attribute_triples1: list[Triple],
    attribute_triples2: list[Triple],
    aligned_ent_pairs: set[tuple[str, str]],
    threshold: float,
) -> set[tuple[str, str]]:
    """Greedily pair attribute predicates by average similarity across aligned-entity values.

    Ports ``align_attribute_by_entities``/``run_one_ae`` -- the mirror
    image of ``align_entities_by_attribute_values``, restricted to values
    of already-aligned entities. Same accept-after-full-scan fix as that
    function (see its docstring). Only exercised when
    ``IMUSELinker(bootstrap_iterations=...)`` is set above the published
    default of ``1`` -- OpenEA's own default (``interactive_model_iter_num:
    1``) never calls this, since its outer loop breaks after one entity-
    alignment pass.
    """
    if not aligned_ent_pairs:
        return set()

    entities1 = {e1 for e1, _ in aligned_ent_pairs}
    entities2 = {e2 for _, e2 in aligned_ent_pairs}
    predicate_entities1, value_by_pair1 = _grouped_attribute_values(
        attribute_triples1, group_by_entity=False, restrict_to=entities1
    )
    predicate_entities2, value_by_pair2 = _grouped_attribute_values(
        attribute_triples2, group_by_entity=False, restrict_to=entities2
    )
    index2 = _value_index(value_by_pair2)

    aligned: set[tuple[str, str]] = set()
    claimed2: set[str] = set()
    for predicate1 in sorted(predicate_entities1):
        own_values = {
            value_by_pair1[(predicate1, entity1)] for entity1 in predicate_entities1[predicate1]
        }
        candidates2 = set()
        for value in own_values:
            candidates2 |= index2.get(value.casefold(), set())
        candidates2 -= claimed2
        if not candidates2:
            continue
        match = _best_match(
            predicate1,
            candidates2,
            aligned_ent_pairs,
            predicate_entities1,
            predicate_entities2,
            value_by_pair1,
            value_by_pair2,
            threshold,
        )
        if match is not None:
            aligned.add((predicate1, match))
            claimed2.add(match)
    return aligned


def bootstrap_alignment(
    attribute_triples1: list[Triple],
    attribute_triples2: list[Triple],
    name_sim_threshold: float,
    entity_sim_threshold: float,
    attr_sim_threshold: float,
    top_k_attribute_pairs: int,
    iterations: int,
) -> set[tuple[str, str]]:
    """Discover an initial entity alignment from attribute triples alone.

    Ports ``interactive_model``: seed an aligned-attribute-predicate set
    by name similarity, then alternate entity alignment (from attribute
    values) and attribute-predicate re-alignment (from the entities just
    aligned) for up to ``iterations`` rounds, stopping early if a round
    finds no new attribute pairs. At the published default
    (``iterations=1``), this runs exactly one entity-alignment pass and
    returns -- ``align_attributes_by_entities`` is never called.
    """
    aligned_entities: set[tuple[str, str]] = set()
    aligned_attrs = align_attributes_by_name(
        attribute_triples1, attribute_triples2, name_sim_threshold, top_k_attribute_pairs
    )

    for iteration in range(1, iterations + 1):
        aligned_entities |= align_entities_by_attribute_values(
            attribute_triples1, attribute_triples2, aligned_attrs, entity_sim_threshold
        )
        if iteration >= iterations:
            break
        new_attrs = align_attributes_by_entities(
            attribute_triples1, attribute_triples2, aligned_entities, attr_sim_threshold
        )
        if new_attrs <= aligned_attrs:
            break
        aligned_attrs |= new_attrs

    return aligned_entities

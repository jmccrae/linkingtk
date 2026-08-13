"""Torch-free supervision/predicate-alignment helpers for
[MultiKELinker][linkingtk.algorithms.ea.multike.MultiKELinker].

Ports OpenEA's ``src/openea/approaches/predicate_alignmnet.py`` (mutual-
best-match Levenshtein predicate alignment), the attribute-triple
cleaning/weighting logic embedded in
``src/openea/approaches/multi_ke.py``'s ``clear_attribute_triples``/
``add_weights``, and ``src/openea/modules/load/read.py``'s
``generate_sup_relation_triples``/``generate_sup_attribute_triples`` --
the seed-pair id-substitution mechanism that is MultiKE's only real
supervision channel (everything else is unsupervised view-training).
Kept separate from ``_multike_torch.py`` (which builds/consumes PyTorch
tensors) so this stays independently testable without ``torch``
installed, matching this repo's established ``_*_text.py`` convention.
"""

from __future__ import annotations

from collections import defaultdict

from linkingtk.algorithms.ea._imuse_text import levenshtein_ratio, local_name
from linkingtk.utils.graph import Triple


def clean_attribute_value(value: str) -> str:
    """Normalize an already-literal-unwrapped attribute value's punctuation.

    ``EnFr15KAttrDataset.load_attribute_triples()`` already strips RDF
    quoting/``^^<datatype>``/``@lang`` suffixes (via
    ``linkingtk.datasets._util.parse_literal_value``), so unlike OpenEA's
    own ``clear_attribute_triples`` step 2, no re-stripping is needed here
    -- only its punctuation normalization: ``.``, ``(``, ``)``, ``,``,
    ``"`` are dropped; ``_``, ``-``, ``/`` become spaces. Returns ``""``
    if the cleaned value still contains ``"http"`` (an unresolved/leftover
    URI, not real literal text, same as OpenEA dropping such values
    entirely); callers should skip triples whose cleaned value is empty.
    """
    cleaned = value
    for char in '.(),"':
        cleaned = cleaned.replace(char, "")
    for char in "_-/":
        cleaned = cleaned.replace(char, " ")
    if "http" in cleaned:
        return ""
    return cleaned


def filter_frequent_predicates(
    triples1: list[Triple], triples2: list[Triple], min_count: int = 10
) -> tuple[list[Triple], list[Triple]]:
    """Keep only triples whose predicate has ``>= min_count`` triples across both KGs.

    Ports ``clear_attribute_triples`` step 1: attribute predicates used
    fewer than ``min_count`` times (combined across both KGs) are treated
    as noise, not real literal-encodable signal.
    """
    counts: dict[str, int] = defaultdict(int)
    for _, predicate, _ in triples1:
        counts[predicate] += 1
    for _, predicate, _ in triples2:
        counts[predicate] += 1
    frequent = {predicate for predicate, count in counts.items() if count >= min_count}
    return (
        [t for t in triples1 if t[1] in frequent],
        [t for t in triples2 if t[1] in frequent],
    )


def align_predicates_by_name(
    predicates1: set[str], predicates2: set[str], threshold: float
) -> dict[tuple[str, str], float]:
    """Mutual-best-match predicate pairs by Levenshtein ratio of humanized local names.

    Ports ``init_predicate_alignment``: for every predicate on each side,
    find its single best match on the other side by ``levenshtein_ratio``
    of ``local_name(...).replace('_', ' ')`` (matches
    ``predicate_alignmnet.py``'s own ``get_local_name``, which humanizes
    underscores -- unlike ``_imuse_text.local_name`` alone, which doesn't).
    A pair is kept only if each is the other's best match *and* that
    similarity clears ``threshold``.
    """

    def humanized_names(predicates: set[str]) -> dict[str, str]:
        return {predicate: local_name(predicate).replace("_", " ") for predicate in predicates}

    names1 = humanized_names(predicates1)
    names2 = humanized_names(predicates2)

    def best_matches(
        source: dict[str, str], target: dict[str, str]
    ) -> dict[str, tuple[str, float]]:
        result: dict[str, tuple[str, float]] = {}
        for predicate, name in source.items():
            best_predicate = ""
            best_sim = 0.0
            for other_predicate, other_name in target.items():
                sim = levenshtein_ratio(name, other_name)
                if sim > best_sim:
                    best_predicate = other_predicate
                    best_sim = sim
            result[predicate] = (best_predicate, best_sim)
        return result

    best1 = best_matches(names1, names2)
    best2 = best_matches(names2, names1)

    aligned: dict[tuple[str, str], float] = {}
    for predicate1, (predicate2, sim) in best1.items():
        if predicate2 and best2.get(predicate2, ("", 0.0))[0] == predicate1 and sim > threshold:
            aligned[(predicate1, predicate2)] = sim
    return aligned


def zoom_weight(weight: float, min_w_before: float, min_w_after: float = 0.5) -> float:
    """Linearly rescale a raw similarity into a training weight. Ports ``zoom_weight`` verbatim."""
    return 1.0 - (1.0 - weight) * (1.0 - min_w_after) / (1.0 - min_w_before)


def weight_attribute_triples(
    triples: list[Triple],
    aligned_predicates: dict[tuple[str, str], float],
    predicate_soft_sim: float,
    *,
    is_kg1: bool,
) -> list[tuple[str, str, str, float]]:
    """Per-triple training weight from predicate-alignment status.

    Ports ``add_weights``: a triple whose predicate is part of an aligned
    pair gets ``zoom_weight(similarity, predicate_soft_sim)``; every other
    triple gets a flat ``0.2``. ``is_kg1`` selects which side of
    ``aligned_predicates``'s keys to match this batch of triples against.
    Only consumed by the attribute view's main loss -- the reference's own
    ``relation_triples_w_weights`` is computed by ``PredicateAlignModel``
    but never actually used anywhere in ``multi_ke.py``'s training methods,
    so it isn't ported here either.
    """
    weight_by_predicate = {
        (p1 if is_kg1 else p2): sim for (p1, p2), sim in aligned_predicates.items()
    }
    weighted: list[tuple[str, str, str, float]] = []
    for entity, predicate, value in triples:
        sim = weight_by_predicate.get(predicate)
        weight = zoom_weight(sim, predicate_soft_sim) if sim is not None else 0.2
        weighted.append((entity, predicate, value, weight))
    return weighted


def _index_by_head_and_tail(
    triples: list[Triple],
) -> tuple[dict[str, set[tuple[str, str]]], dict[str, set[tuple[str, str]]]]:
    by_head: dict[str, set[tuple[str, str]]] = defaultdict(set)
    by_tail: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for head, relation, tail in triples:
        by_head[head].add((relation, tail))
        by_tail[tail].add((head, relation))
    return by_head, by_tail


def substitute_relation_triples_one_link(
    e1: str,
    e2: str,
    by_head: dict[str, set[tuple[str, str]]],
    by_tail: dict[str, set[tuple[str, str]]],
) -> set[Triple]:
    """Relabel every relation triple touching ``e1`` with ``e2``'s id.

    Ports ``generate_sup_relation_triples_one_link`` -- half of MultiKE's
    only real supervision signal: for a seed pair, every relation triple
    KG1 has about ``e1`` becomes a "supervised" triple about ``e2``,
    trained directly into the shared entity table's ``e2`` row.
    """
    new_triples: set[Triple] = set()
    for relation, tail in by_head.get(e1, ()):
        new_triples.add((e2, relation, tail))
    for head, relation in by_tail.get(e1, ()):
        new_triples.add((head, relation, e2))
    return new_triples


def generate_cross_kg_relation_triples(
    seed_pairs: list[tuple[str, str]], triples1: list[Triple], triples2: list[Triple]
) -> tuple[list[Triple], list[Triple]]:
    """Ports ``generate_sup_relation_triples`` over every seed pair, both directions."""
    by_head1, by_tail1 = _index_by_head_and_tail(triples1)
    by_head2, by_tail2 = _index_by_head_and_tail(triples2)
    new1: set[Triple] = set()
    new2: set[Triple] = set()
    for e1, e2 in seed_pairs:
        new1 |= substitute_relation_triples_one_link(e1, e2, by_head1, by_tail1)
        new2 |= substitute_relation_triples_one_link(e2, e1, by_head2, by_tail2)
    return sorted(new1), sorted(new2)


def _index_by_entity(triples: list[Triple]) -> dict[str, set[tuple[str, str]]]:
    av_dict: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for entity, attribute, value in triples:
        av_dict[entity].add((attribute, value))
    return av_dict


def substitute_attribute_triples_one_link(
    e1: str, e2: str, av_dict: dict[str, set[tuple[str, str]]]
) -> set[Triple]:
    """Relabel every attribute triple touching ``e1`` with ``e2``'s id.

    Ports ``generate_sup_attribute_triples_one_link`` -- the attribute-view
    half of MultiKE's supervision signal, symmetric to
    ``substitute_relation_triples_one_link``.
    """
    return {(e2, attribute, value) for attribute, value in av_dict.get(e1, ())}


def generate_cross_kg_attribute_triples(
    seed_pairs: list[tuple[str, str]], triples1: list[Triple], triples2: list[Triple]
) -> tuple[list[Triple], list[Triple]]:
    """Ports ``generate_sup_attribute_triples`` over every seed pair, both directions."""
    av_dict1 = _index_by_entity(triples1)
    av_dict2 = _index_by_entity(triples2)
    new1: set[Triple] = set()
    new2: set[Triple] = set()
    for e1, e2 in seed_pairs:
        new1 |= substitute_attribute_triples_one_link(e1, e2, av_dict1)
        new2 |= substitute_attribute_triples_one_link(e2, e1, av_dict2)
    return sorted(new1), sorted(new2)


def substitute_predicate_triples(
    triples: list[Triple],
    aligned_predicates: dict[tuple[str, str], float],
    *,
    is_kg1: bool,
) -> list[tuple[str, str, str, float]]:
    """Relabel a triple's predicate to its aligned cross-KG partner, weighted by similarity.

    Ports ``generate_sup_predicate_triples`` for one side: the
    predicate-level analogue of ``generate_cross_kg_*_triples``'s entity
    substitution -- trains the shared predicate embedding table by having
    KG1's own triples "use" KG2's aligned predicate id (and vice versa).
    """
    partner_and_weight = {
        (p1 if is_kg1 else p2): (p2 if is_kg1 else p1, sim)
        for (p1, p2), sim in aligned_predicates.items()
    }
    substituted: list[tuple[str, str, str, float]] = []
    for entity, predicate, value in triples:
        match = partner_and_weight.get(predicate)
        if match is not None:
            partner, weight = match
            substituted.append((entity, partner, value, weight))
    return substituted

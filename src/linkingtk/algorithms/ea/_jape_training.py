"""Private, plain-numpy (torch-free) helpers for
[JAPELinker][linkingtk.algorithms.ea.jape.JAPELinker].

Ported from OpenEA's reference implementation
(https://github.com/nju-websoft/OpenEA -- ``approaches/attr2vec.py``'s
non-TensorFlow helper functions). Independently testable without ``torch``
installed -- see ``_jape_torch.py`` for the training-step functions that
build/consume PyTorch tensors (attribute-embedding NCE training, the
structural triple loss, and the attribute-similarity regularizer).
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from typing import TYPE_CHECKING, Any

import numpy as np

from linkingtk.utils.graph import Triple

if TYPE_CHECKING:
    import numpy.typing as npt


def select_popular_attributes(
    attribute_triples1: list[Triple],
    attribute_triples2: list[Triple],
    threshold: float,
) -> set[str]:
    """The most frequent attribute predicates, per KG, unioned across both.

    Mirrors OpenEA's ``get_kgs_popular_attributes``: for each KG
    independently, ranks attribute predicates by how many triples use
    them and keeps the top ``threshold`` fraction; the two KGs' selected
    sets are then unioned (a predicate popular in either KG is kept).

    Args:
        attribute_triples1: KG1's ``(entity_id, predicate, value)`` triples.
        attribute_triples2: KG2's attribute triples.
        threshold: Fraction of each KG's distinct attribute predicates to
            keep, by descending frequency. OpenEA's published value is
            ``0.9``.

    Returns:
        The union of both KGs' selected predicate sets.
    """

    def popular(triples: list[Triple]) -> set[str]:
        counts: dict[str, int] = defaultdict(int)
        for _, predicate, _ in triples:
            counts[predicate] += 1
        selected_count = int(len(counts) * threshold)
        ranked = sorted(counts, key=lambda predicate: counts[predicate], reverse=True)
        return set(ranked[:selected_count])

    return popular(attribute_triples1) | popular(attribute_triples2)


def build_entity_attributes(
    attribute_triples1: list[Triple],
    attribute_triples2: list[Triple],
    selected_attributes: set[str],
    seed_pairs: list[tuple[str, str]],
) -> dict[str, set[str]]:
    """Each entity's selected attribute predicates, with seed pairs merged.

    For an entity that's a seed pair's source or target, its attribute
    set is unioned with its counterpart's attribute set *before*
    intersecting with ``selected_attributes`` -- this is what makes the
    resulting training data cross-lingual-aware (the same real-world
    entity is often described with different attribute predicates in
    each KG). Ported from ``generate_training_data``.

    Args:
        attribute_triples1: KG1's attribute triples.
        attribute_triples2: KG2's attribute triples.
        selected_attributes: Predicates to keep, e.g. from
            [select_popular_attributes][linkingtk.algorithms.ea._jape_training.select_popular_attributes].
        seed_pairs: Known-correct ``(source_id, target_id)`` pairs.

    Returns:
        ``entity_id -> selected attribute predicates``. Entities left
        with an empty set after selection are omitted.
    """
    entity_attrs: dict[str, set[str]] = defaultdict(set)
    for entity_id, predicate, _ in itertools.chain(attribute_triples1, attribute_triples2):
        entity_attrs[entity_id].add(predicate)

    counterpart = {source: target for source, target in seed_pairs}
    counterpart.update({target: source for source, target in seed_pairs})

    merged: dict[str, set[str]] = {}
    for entity_id, attrs in entity_attrs.items():
        combined = attrs | entity_attrs.get(counterpart.get(entity_id, ""), set())
        selected = combined & selected_attributes
        if selected:
            merged[entity_id] = selected
    return merged


def generate_training_pairs(entity_attributes: dict[str, set[str]]) -> list[tuple[str, str]]:
    """Skip-gram (input, context) training pairs from each entity's attribute set.

    Every entity's attribute set contributes all pairwise combinations of
    its (selected) attribute predicates -- ported from
    ``generate_training_data``'s ``itertools.combinations(attributes, 2)``
    loop, including its asymmetry: only one direction per combination is
    emitted, not both. OpenEA iterates a Python ``set`` directly, whose
    iteration order (and therefore which direction of each pair gets
    emitted) isn't reproducible across runs; this sorts each entity's
    attributes first so the direction is deterministic here.

    Args:
        entity_attributes: From
            [build_entity_attributes][linkingtk.algorithms.ea._jape_training.build_entity_attributes].

    Returns:
        ``(attribute, context_attribute)`` pairs.
    """
    pairs: list[tuple[str, str]] = []
    for attrs in entity_attributes.values():
        pairs.extend(itertools.combinations(sorted(attrs), 2))
    return pairs


def pool_entity_attribute_vectors(
    entity_ids: list[str],
    entity_attributes: dict[str, set[str]],
    attr_to_id: dict[str, int],
    attr_embeds: npt.NDArray[np.floating[Any]],
) -> npt.NDArray[np.floating[Any]]:
    """Mean-pool, then L2-normalize, each entity's selected-attribute embeddings.

    Ports ``get_ent_embeds_from_attributes``. Entities with no selected
    attributes get a zero vector (before normalization -- normalization
    leaves an all-zero row as all-zero, not NaN, since the zero norm is
    guarded).

    Args:
        entity_ids: Entity ids, in the order rows should be returned.
        entity_attributes: From
            [build_entity_attributes][linkingtk.algorithms.ea._jape_training.build_entity_attributes].
        attr_to_id: Attribute predicate -> row index into ``attr_embeds``.
        attr_embeds: Trained attribute-predicate embeddings, ``(num_attributes, dim)``.

    Returns:
        ``(len(entity_ids), dim)`` array of entity attribute vectors.
    """
    dim = attr_embeds.shape[1]
    vectors = np.zeros((len(entity_ids), dim), dtype=np.float32)
    for row, entity_id in enumerate(entity_ids):
        attrs = entity_attributes.get(entity_id)
        if not attrs:
            continue
        indices = [attr_to_id[attr] for attr in attrs if attr in attr_to_id]
        if indices:
            vectors[row] = attr_embeds[indices].mean(axis=0)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def reference_pools(
    entity_ids1: list[str],
    entity_ids2: list[str],
    seed_pairs: list[tuple[str, str]],
) -> tuple[list[str], list[str]]:
    """Every entity id on each side that isn't already a seed pair's source/target.

    Same "reference pool" concept already established for
    [IPTransELinker][linkingtk.algorithms.ea.iptranse.IPTransELinker]'s
    bootstrap pool (see its module docstring's deviation note) -- reused
    here for the attribute-similarity matrix's reference entities.

    Args:
        entity_ids1: Candidate KG1 entity ids (already filtered to those
            with a trained structural embedding).
        entity_ids2: Candidate KG2 entity ids.
        seed_pairs: Known-correct ``(source_id, target_id)`` pairs.

    Returns:
        ``(pool1, pool2)``.
    """
    seed_sources = {source for source, _ in seed_pairs}
    seed_targets = {target for _, target in seed_pairs}
    pool1 = [entity_id for entity_id in entity_ids1 if entity_id not in seed_sources]
    pool2 = [entity_id for entity_id in entity_ids2 if entity_id not in seed_targets]
    return pool1, pool2

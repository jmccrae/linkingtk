"""Private, plain-numpy (torch-free) helpers for
[GCNAlignLinker][linkingtk.algorithms.ea.gcn_align.GCNAlignLinker].

Ported from OpenEA's reference implementation
(https://github.com/nju-websoft/OpenEA -- ``approaches/gcn_align.py``'s
``GCN_Utils.func``/``ifunc``/``get_weighted_adj``,
``GCN_Align.train_embeddings``'s uniform negative-sampling arrays, and
``load_attr`` for the attribute-presence feature matrix). Independently
testable without ``torch`` installed -- see ``_gcn_align_torch.py`` for
the model/training-step functions.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

    from linkingtk.utils.graph import Triple

_DEFAULT_MIN_WEIGHT = 0.3
_DEFAULT_TOP_FRACTION = 0.7


def compute_relation_functionality(
    triples: npt.NDArray[np.int64],
) -> tuple[dict[int, float], dict[int, float]]:
    """Per-relation functionality/inverse-functionality, per OpenEA's ``func``/``ifunc``.

    ``r2f[r]`` = (# distinct head entities of ``r``) / (# triples with
    ``r``) -- how "selective" ``r`` is as a function of its head.
    ``r2if[r]`` is the same but over tail entities. Both are used by
    [build_weighted_adjacency][linkingtk.algorithms.ea._gcn_align_training.build_weighted_adjacency]
    to weight each triple's structural-influence edges.

    Args:
        triples: ``(n, 3)`` int64 array of ``(head, relation, tail)`` ids.

    Returns:
        ``(r2f, r2if)``, both ``{relation_id: float}``. Empty dicts if
        ``triples`` is empty.
    """
    heads_by_relation: dict[int, set[int]] = defaultdict(set)
    tails_by_relation: dict[int, set[int]] = defaultdict(set)
    counts: dict[int, int] = defaultdict(int)
    for head, relation, tail in triples:
        head, relation, tail = int(head), int(relation), int(tail)
        heads_by_relation[relation].add(head)
        tails_by_relation[relation].add(tail)
        counts[relation] += 1
    r2f = {relation: len(heads_by_relation[relation]) / count for relation, count in counts.items()}
    r2if = {
        relation: len(tails_by_relation[relation]) / count for relation, count in counts.items()
    }
    return r2f, r2if


def build_weighted_adjacency(
    triples: npt.NDArray[np.int64],
    r2f: dict[int, float],
    r2if: dict[int, float],
    min_weight: float = _DEFAULT_MIN_WEIGHT,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """Directed, relation-functionality-weighted adjacency, per OpenEA's ``get_weighted_adj``.

    For every triple ``(h, r, t)`` with ``h != t``: edge ``h -> t`` gets
    weight ``max(r2if[r], min_weight)`` and edge ``t -> h`` gets weight
    ``max(r2f[r], min_weight)`` -- edges through more "selective"/functional
    relations carry more weight. Weights from multiple triples sharing the
    same ``(h, t)`` pair are summed. Self-loop triples (``h == t``) are
    skipped entirely (no edge contributed either direction), matching
    OpenEA's own ``if tri[0] == tri[2]: continue``.

    Args:
        triples: ``(n, 3)`` int64 array of ``(head, relation, tail)`` ids.
        r2f: Per-relation functionality, from
            [compute_relation_functionality][linkingtk.algorithms.ea._gcn_align_training.compute_relation_functionality].
        r2if: Per-relation inverse functionality, same source.
        min_weight: Floor applied to every edge weight.

    Returns:
        ``(indices, values)``: ``indices`` is ``(2, nnz)`` int64 (row 0 =
        source/row id, row 1 = target/column id -- ready for
        [normalize_adjacency_coo][linkingtk.utils.sparse_gcn.normalize_adjacency_coo]),
        ``values`` is ``(nnz,)`` float64. Empty (shape ``(2, 0)``/``(0,)``)
        if every triple is a self-loop or ``triples`` is empty.
    """
    weights: dict[tuple[int, int], float] = defaultdict(float)
    for head, relation, tail in triples:
        head, relation, tail = int(head), int(relation), int(tail)
        if head == tail:
            continue
        weights[(head, tail)] += max(r2if[relation], min_weight)
        weights[(tail, head)] += max(r2f[relation], min_weight)

    if not weights:
        return np.empty((2, 0), dtype=np.int64), np.empty(0, dtype=np.float64)

    pairs = list(weights.keys())
    rows = np.array([pair[0] for pair in pairs], dtype=np.int64)
    cols = np.array([pair[1] for pair in pairs], dtype=np.int64)
    values = np.array([weights[pair] for pair in pairs], dtype=np.float64)
    return np.stack([rows, cols]), values


def sample_negatives(
    seed_pairs: list[tuple[int, int]],
    num_entities: int,
    neg_triple_num: int,
    rng: np.random.Generator,
) -> tuple[
    npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64]
]:
    """Two-sided uniform negative sampling, per OpenEA's ``train_embeddings`` setup.

    For ``t`` seed pairs and ``k = neg_triple_num`` negatives per pair:
    right-corruption keeps each pair's own left id fixed (repeated ``k``
    times) and draws ``k`` random right ids; left-corruption is the mirror
    image. Both id sides are drawn from the same combined ``[0,
    num_entities)`` id space -- GCN-Align trains one shared embedding table
    over both KGs' entities, not two independently-indexed spaces.

    Args:
        seed_pairs: ``(source_id, target_id)`` id pairs (already mapped via
            `entity_to_id`, not raw entity labels).
        num_entities: Total entity count (both KGs combined) to sample
            negatives from.
        neg_triple_num: Negatives per pair per side (``k``).
        rng: Source of randomness.

    Returns:
        ``(neg_left, neg_right, neg2_left, neg2_right)``, each ``(t * k,)``
        int64 -- pass directly to
        [margin_ranking_loss_l1][linkingtk.algorithms.ea._ea_losses.margin_ranking_loss_l1].
    """
    left = np.array([source for source, _ in seed_pairs], dtype=np.int64)
    right = np.array([target for _, target in seed_pairs], dtype=np.int64)
    k = neg_triple_num
    neg_left = np.repeat(left, k)
    neg_right = rng.integers(0, num_entities, size=len(left) * k, dtype=np.int64)
    neg2_left = rng.integers(0, num_entities, size=len(left) * k, dtype=np.int64)
    neg2_right = np.repeat(right, k)
    return neg_left, neg_right, neg2_left, neg2_right


def build_attribute_features(
    attribute_triples1: list[Triple],
    attribute_triples2: list[Triple],
    entity_to_id: dict[str, int],
    top_fraction: float = _DEFAULT_TOP_FRACTION,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64], int]:
    """One-hot attribute-*predicate*-presence features, per OpenEA's ``load_attr``.

    Ports ``load_attr`` exactly: the feature vocabulary is the top
    ``top_fraction`` (by how many distinct entities carry each predicate,
    combined across both KGs) attribute *predicates* -- not values, since
    ``load_attr`` reads its input from ``entity_attributes_dict`` (entity
    -> set of predicate URIs that entity has an attribute triple for),
    discarding literal values entirely. An entity gets feature ``1.0`` at
    a predicate's column iff it has at least one attribute triple with
    that predicate; the specific value(s) never matter. Predicates outside
    the top ``top_fraction`` are dropped (contribute no feature column at
    all, not folded into an "other" bucket), matching OpenEA's own
    ``if v in attr2id`` guard.

    Args:
        attribute_triples1: KG1's ``(entity_id, predicate, value)``
            triples, e.g. from
            [_OpenEANativeDataset.load_attribute_triples][linkingtk.datasets.openea_native._OpenEANativeDataset.load_attribute_triples].
        attribute_triples2: KG2's own attribute triples.
        entity_to_id: Combined entity label -> id mapping, e.g. from
            [build_id_mappings][linkingtk.utils.graph.build_id_mappings].
            Triples whose entity id isn't a key here are dropped.
        top_fraction: Fraction of the combined, frequency-ranked predicate
            vocabulary to keep as feature columns. OpenEA's own value
            (hardcoded, not configurable in the reference) is ``0.7``.

    Returns:
        ``(indices, values, num_attributes)``: ``indices`` is ``(2, nnz)``
        int64 (row 0 = entity id, row 1 = attribute-column id -- ready for
        [normalize_adjacency_coo][linkingtk.utils.sparse_gcn.normalize_adjacency_coo]-free
        use directly via
        [coo_to_torch_sparse][linkingtk.utils.sparse_gcn.coo_to_torch_sparse]
        with ``size=(num_entities, num_attributes)``), ``values`` is
        ``(nnz,)`` float64 (always ``1.0``), ``num_attributes`` is the
        feature vocabulary size (``0`` if there are no attribute triples
        at all).
    """
    predicates_by_entity: dict[str, set[str]] = defaultdict(set)
    for entity_id, predicate, _value in attribute_triples1 + attribute_triples2:
        predicates_by_entity[entity_id].add(predicate)

    counts: Counter[str] = Counter()
    for predicates in predicates_by_entity.values():
        counts.update(predicates)

    ranked_predicates = [predicate for predicate, _count in counts.most_common()]
    num_attributes = int(top_fraction * len(ranked_predicates))
    predicate_to_id = {
        predicate: i for i, predicate in enumerate(ranked_predicates[:num_attributes])
    }

    rows: list[int] = []
    cols: list[int] = []
    for entity_id, predicates in predicates_by_entity.items():
        row = entity_to_id.get(entity_id)
        if row is None:
            continue
        for predicate in predicates:
            col = predicate_to_id.get(predicate)
            if col is not None:
                rows.append(row)
                cols.append(col)

    if not rows:
        return np.empty((2, 0), dtype=np.int64), np.empty(0, dtype=np.float64), num_attributes

    indices = np.array([rows, cols], dtype=np.int64)
    values = np.ones(len(rows), dtype=np.float64)
    return indices, values, num_attributes

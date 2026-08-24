"""Private, plain-numpy (torch-free) helpers for
[GCNAlignLinker][linkingtk.algorithms.ea.gcn_align.GCNAlignLinker].

Ported from OpenEA's reference implementation
(https://github.com/nju-websoft/OpenEA -- ``approaches/gcn_align.py``'s
``GCN_Utils.func``/``ifunc``/``get_weighted_adj`` and
``GCN_Align.train_embeddings``'s uniform negative-sampling arrays).
Independently testable without ``torch`` installed -- see
``_gcn_align_torch.py`` for the model/training-step functions.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

_DEFAULT_MIN_WEIGHT = 0.3


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

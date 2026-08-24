"""Private, plain-numpy (torch-free) helpers for
[RDGCNLinker][linkingtk.algorithms.ea.rdgcn.RDGCNLinker].

Ported from OpenEA's reference implementation
(https://github.com/nju-websoft/OpenEA -- ``approaches/rdgcn.py``'s
``get_mat``/``get_sparse_tensor`` (primal adjacency), ``rfunc``/``compute_r``/
``get_dual_input`` (relation "dual graph" and per-relation features), and
``_get_desc_input``/``_get_local_name_by_name_triple`` (name-embedding
init)). Independently testable without ``torch`` installed -- see
``_rdgcn_torch.py`` for the attention/GCN model and training-step functions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from linkingtk.algorithms.ea._kdcoe_text import tokenize_description
from linkingtk.core.entity import label_texts

if TYPE_CHECKING:
    import numpy.typing as npt

    from linkingtk.core.entity import Entity


def build_primal_adjacency(
    triples: npt.NDArray[np.int64], num_entities: int
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """Unweighted, symmetric primal (entity) adjacency, per OpenEA's ``get_mat``.

    For every triple ``(h, r, t)`` with ``h != t``, both ``h -> t`` and
    ``t -> h`` get weight ``1.0`` -- a plain undirected adjacency,
    deduplicated (unlike
    [build_weighted_adjacency][linkingtk.algorithms.ea._gcn_align_training.build_weighted_adjacency],
    repeated triples between the same entity pair do **not** accumulate
    weight, matching OpenEA's own ``if (h, t) not in pos: pos[(h, t)] = 1``
    dedup-to-1 behavior). Self-loop triples (``h == t``) are skipped, same
    as
    [build_weighted_adjacency][linkingtk.algorithms.ea._gcn_align_training.build_weighted_adjacency].
    Pass the result to
    [normalize_adjacency_coo][linkingtk.utils.sparse_gcn.normalize_adjacency_coo]
    (``add_self_loops=True``) -- symmetric here, so its row-sum-vs-column-sum
    distinction is a no-op.

    Args:
        triples: ``(n, 3)`` int64 array of ``(head, relation, tail)`` ids.
        num_entities: Total entity count (both KGs combined).

    Returns:
        ``(indices, values)`` -- ``(2, nnz)`` int64 / ``(nnz,)`` float64.
    """
    pairs: set[tuple[int, int]] = set()
    for head, _relation, tail in triples:
        head, tail = int(head), int(tail)
        if head == tail:
            continue
        pairs.add((head, tail))
        pairs.add((tail, head))

    if not pairs:
        return np.empty((2, 0), dtype=np.int64), np.empty(0, dtype=np.float64)

    rows = np.array([pair[0] for pair in pairs], dtype=np.int64)
    cols = np.array([pair[1] for pair in pairs], dtype=np.int64)
    values = np.ones(len(pairs), dtype=np.float64)
    return np.stack([rows, cols]), values


def build_dual_graph(
    triples: npt.NDArray[np.int64], num_relations: int, num_entities: int
) -> npt.NDArray[np.float64]:
    """Dense relation-to-relation "dual graph", per OpenEA's ``get_dual_input``.

    Two relations are dual-graph-adjacent with weight ``jaccard(head-entity
    sets) + jaccard(tail-entity sets)`` -- including the diagonal (``i ==
    j``, weight ``2.0``, both Jaccard similarities of a set with itself).
    Kept dense (not sparse COO) since the relation count is small (low
    hundreds for OpenEA-scale datasets) and every pair genuinely
    participates in the downstream dense-attention softmax's masking, unlike
    the primal/entity-scale adjacencies elsewhere in this package.
    Vectorized via head/tail membership matrices rather than OpenEA's own
    ``O(R^2)`` nested-loop-over-Python-sets -- same result, just computed as
    one matrix multiply per side.

    Args:
        triples: ``(n, 3)`` int64 array of ``(head, relation, tail)`` ids.
        num_relations: Total relation count.
        num_entities: Total entity count (both KGs combined).

    Returns:
        ``(num_relations, num_relations)`` float64 dense array.
    """
    head_matrix = np.zeros((num_relations, num_entities), dtype=np.float64)
    tail_matrix = np.zeros((num_relations, num_entities), dtype=np.float64)
    for head, relation, tail in triples:
        head_matrix[int(relation), int(head)] = 1.0
        tail_matrix[int(relation), int(tail)] = 1.0

    def _jaccard(membership: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        counts = membership.sum(axis=1)
        intersection = membership @ membership.T
        union = counts[:, None] + counts[None, :] - intersection
        with np.errstate(divide="ignore", invalid="ignore"):
            result = np.where(union > 0, intersection / union, 0.0)
        return result

    return _jaccard(head_matrix) + _jaccard(tail_matrix)


def build_relation_masks(
    triples: npt.NDArray[np.int64], num_relations: int
) -> tuple[
    tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]],
    tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]],
]:
    """Row-normalized head/tail entity-membership matrices, per OpenEA's ``rfunc``/``compute_r``.

    ``head_mask[r, e] = 1/|heads(r)|`` if ``e`` is ever a head of relation
    ``r`` (``0`` otherwise); ``tail_mask`` is the mirror image over tails.
    Membership is binary per (relation, entity) pair before normalizing --
    an entity appearing as relation ``r``'s head in multiple triples still
    only contributes once, matching OpenEA's own ``head_r[h][r] = 1``
    (assignment, not accumulation). Pre-normalizing (dividing by each
    relation's head/tail count) here means the torch side's per-relation
    feature computation is a single sparse matmul (``head_mask @
    primal_embeddings``) rather than a matmul-then-divide.

    Args:
        triples: ``(n, 3)`` int64 array of ``(head, relation, tail)`` ids.
        num_relations: Total relation count.

    Returns:
        ``((head_indices, head_values), (tail_indices, tail_values))``,
        each ``indices`` shaped ``(2, nnz)`` (row 0 = relation id, row 1 =
        entity id) and ``values`` shaped ``(nnz,)`` float64 -- ready for
        [coo_to_torch_sparse][linkingtk.utils.sparse_gcn.coo_to_torch_sparse]
        with ``size=(num_relations, num_entities)``.
    """
    heads_by_relation: dict[int, set[int]] = {}
    tails_by_relation: dict[int, set[int]] = {}
    for head, relation, tail in triples:
        relation = int(relation)
        heads_by_relation.setdefault(relation, set()).add(int(head))
        tails_by_relation.setdefault(relation, set()).add(int(tail))

    def _mask(by_relation: dict[int, set[int]]) -> tuple[np.ndarray, np.ndarray]:
        rows, cols, values = [], [], []
        for relation, entities in by_relation.items():
            weight = 1.0 / len(entities)
            for entity in entities:
                rows.append(relation)
                cols.append(entity)
                values.append(weight)
        if not rows:
            return np.empty((2, 0), dtype=np.int64), np.empty(0, dtype=np.float64)
        indices = np.array([rows, cols], dtype=np.int64)
        return indices, np.array(values, dtype=np.float64)

    return _mask(heads_by_relation), _mask(tails_by_relation)


def build_edge_relations(
    triples: npt.NDArray[np.int64],
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Parallel per-triple ``(head, tail, relation)`` id arrays, per OpenEA's ``rfunc``'s ``r_mat``.

    Unlike [build_primal_adjacency][linkingtk.algorithms.ea._rdgcn_training.build_primal_adjacency],
    this is **not** deduplicated or symmetrized -- one row per triple,
    directed (head -> tail only), exactly as given. Used by the sparse
    attention layer to look up, for every primal edge, which relation
    connects it.

    Args:
        triples: ``(n, 3)`` int64 array of ``(head, relation, tail)`` ids.

    Returns:
        ``(edge_heads, edge_tails, edge_relations)``, each ``(n,)`` int64.
    """
    if len(triples) == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, empty
    return (
        triples[:, 0].astype(np.int64),
        triples[:, 2].astype(np.int64),
        triples[:, 1].astype(np.int64),
    )


def init_name_embeddings(
    entities: list[Entity],
    word_vectors: dict[str, npt.NDArray[np.floating[Any]]],
    dim: int,
    default_length: int = 4,
) -> dict[str, npt.NDArray[np.float32]]:
    """Each entity's initial embedding: summed word vectors of its first label's tokens.

    Ports OpenEA's ``_get_desc_input``'s name-embedding init
    (``word_em[e_desc_input].sum(axis=1)``): tokenize the entity's local
    name, look up up to ``default_length``
    tokens' pretrained word vectors, sum them. A token with no vector (out
    of ``word_vectors``, or padding past the name's real token count)
    contributes a zero vector -- **not** a random placeholder, matching
    OpenEA's own ``np.zeros`` row for out-of-vocabulary words, so a
    name entirely absent from ``word_vectors`` just gets a zero-vector
    init rather than a fabricated one.

    Uses `Entity.labels` (already populated by this repo's dataset
    loaders, see
    [EnFr15KAttrDataset][linkingtk.datasets.EnFr15KAttrDataset]) as the
    "local name" source rather than OpenEA's own
    ``_get_local_name_by_name_triple`` URI-parsing -- confirmed a no-op
    deviation for EN-FR-15K specifically: that function's dataset-specific
    name-attribute matching (``D_Y``/``D_W`` special-cased predicate
    lists) never triggers for this dataset (falls to its
    ``else: name_attribute_list = {}`` branch), so OpenEA's own code
    already falls back to the entity URI's tail segment here -- the same
    text this repo's loader already populates as the label.

    Args:
        entities: Entities needing an initial embedding (both KGs,
            concatenated by the caller).
        word_vectors: Pretrained word vectors, e.g. from
            [load_fasttext_vectors][linkingtk.algorithms.ea._kdcoe_text.load_fasttext_vectors].
        dim: Embedding dimensionality -- must match ``word_vectors``' own
            vector dimensionality (no projection is applied).
        default_length: Max tokens summed per entity. OpenEA's published
            value is ``4``.

    Returns:
        ``entity_id -> (dim,) float32 array``.
    """
    result: dict[str, npt.NDArray[np.float32]] = {}
    for entity in entities:
        texts = label_texts(entity)
        text = texts[0] if texts else ""
        tokens = tokenize_description(text)[:default_length]
        vectors = [word_vectors[token] for token in tokens if token in word_vectors]
        if vectors:
            result[entity.id] = np.sum(vectors, axis=0).astype(np.float32)
        else:
            result[entity.id] = np.zeros(dim, dtype=np.float32)
    return result

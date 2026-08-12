"""Private training helpers for [IPTransELinker][linkingtk.algorithms.ea.iptranse.IPTransELinker].

Pure, independently-testable functions ported from OpenEA's reference
implementation (https://github.com/nju-websoft/OpenEA --
``approaches/iptranse.py``, ``modules/bootstrapping/alignment_finder.py``,
``modules/train/batch.py``, ``modules/load/read.py``). Not part of the
public API -- see ``iptranse.py`` for the orchestrating class and the
deviations from OpenEA's own code documented in its docstring.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any

import numpy as np

from linkingtk.utils.graph import Triple

if TYPE_CHECKING:
    import numpy.typing as npt


def build_shared_id_mappings(
    triples: list[Triple],
    seed_pairs: list[tuple[str, str]],
) -> tuple[dict[str, int], dict[str, int]]:
    """Entity/relation id mappings with IPTransE's "sharing" alignment semantics.

    Every seed pair's target entity is aliased to its source entity's id --
    both get exactly one embedding row, not two independently-trained rows
    linked by a learned mapping (contrast
    [MTransELinker][linkingtk.algorithms.ea.mtranse.MTransELinker]).
    Relations are *not* merged across KGs (mirrors OpenEA's
    ``generate_sharing_id([], ...)`` fallthrough for relations -- the same
    plain sorted-union behavior
    [build_id_mappings][linkingtk.utils.graph.build_id_mappings] already
    implements, since real relation label strings never coincide across
    the two KGs in these datasets).

    Args:
        triples: Combined triples from both KGs.
        seed_pairs: ``(source_id, target_id)`` pairs to alias together.
            Pairs whose ids aren't present in ``triples`` are ignored.

    Returns:
        ``(entity_to_id, relation_to_id)``. The entity id range has
        exactly ``len(entities) - len(usable seed pairs)`` distinct
        values, matching OpenEA's ``kgs.entities_num`` after merging.
    """
    entities = sorted({s for s, _, _ in triples} | {o for _, _, o in triples})
    relations = sorted({p for _, p, _ in triples})
    relation_to_id = {label: index for index, label in enumerate(relations)}

    entity_set = set(entities)
    alias = {
        target: source
        for source, target in seed_pairs
        if source in entity_set and target in entity_set
    }

    def canonical(entity: str) -> str:
        return alias.get(entity, entity)

    representatives = sorted({canonical(e) for e in entities})
    representative_to_id = {label: index for index, label in enumerate(representatives)}
    entity_to_id = {e: representative_to_id[canonical(e)] for e in entities}
    return entity_to_id, relation_to_id


def generate_two_step_paths(triples: list[Triple]) -> list[tuple[str, str, str, int]]:
    """2-hop relation-composition paths within one KG's own triples.

    For every chain ``h_x --r_x--> m --r_y--> t_y`` where a *direct* edge
    ``h_x --r--> t_y`` also exists, emits ``(r_x, r_y, r, path_weight)``
    where ``path_weight = fanout(h_x, r_x) * fanout(m, r_y)`` -- a proxy
    for how generic/common this 2-hop composition is. Chains with
    ``path_weight >= 101`` are dropped (OpenEA's own hub-node cutoff).

    Mirrors OpenEA's ``generate_2steps_path``
    (``approaches/iptranse.py``), reimplemented with dict grouping instead
    of a pandas groupby/self-join/inner-join pipeline -- this repo has no
    pandas dependency and none should be added for this.

    Args:
        triples: One KG's own triples (not combined across KGs -- paths
            are meaningless across disjoint entity spaces).

    Returns:
        ``(r_x, r_y, r, path_weight)`` tuples, relation labels still
        unmapped to ids.
    """
    fanout: dict[tuple[str, str], int] = defaultdict(int)
    by_head_rel: dict[tuple[str, str], list[str]] = defaultdict(list)
    by_head: dict[str, list[tuple[str, str]]] = defaultdict(list)
    direct_edge: dict[tuple[str, str], list[str]] = defaultdict(list)

    for h, r, t in triples:
        fanout[(h, r)] += 1
        by_head_rel[(h, r)].append(t)
        by_head[h].append((r, t))
        direct_edge[(h, t)].append(r)

    paths: list[tuple[str, str, str, int]] = []
    for (h_x, r_x), mids in by_head_rel.items():
        size_x = fanout[(h_x, r_x)]
        for m in mids:
            for r_y, t_y in by_head.get(m, ()):
                weight = size_x * fanout[(m, r_y)]
                if weight >= 101:
                    continue
                paths.extend((r_x, r_y, r, weight) for r in direct_edge.get((h_x, t_y), ()))
    return paths


def sample_negative_triples(
    positive_triples: npt.NDArray[np.int64],
    entity_pool: npt.NDArray[np.int64],
    real_triples: set[tuple[int, int, int]],
    rng: np.random.Generator,
    max_tries: int = 10,
) -> npt.NDArray[np.int64]:
    """Corrupt one endpoint of each positive triple to build a negative.

    For each row, a fair coin decides whether the head or tail is
    corrupted; the replacement is drawn uniformly from ``entity_pool``
    (that triple's own KG's entities for the main triple loss, or the
    combined KG1+KG2 pool for the bootstrap/alignment loss -- see
    callers). Best-effort avoids reproducing a real triple: retries up to
    ``max_tries``, then accepts the last draw regardless.

    Deliberately simplified from OpenEA's vectorized
    ``generate_neg_triples_fast`` to a plain per-row retry loop, for
    readability -- same statistical behavior, same precedent as
    ``mtranse.py``'s ``_validation_hits1`` simplification.

    Args:
        positive_triples: ``(n, 3)`` int64 array of ``(head, rel, tail)``
            ids.
        entity_pool: Candidate replacement entity ids.
        real_triples: Set of ``(head, rel, tail)`` id tuples to avoid
            reproducing. Pass an empty set to skip avoidance entirely
            (matches OpenEA's alignment-loss negative sampling, which has
            no such check).
        rng: Random generator, for reproducibility.
        max_tries: Retries per row before giving up and accepting the
            last (possibly still-real) draw.

    Returns:
        ``(n, 3)`` int64 array of corrupted triples, same shape as
        ``positive_triples``.
    """
    negatives = positive_triples.copy()
    corrupt_head = rng.random(len(positive_triples)) < 0.5
    for i in range(len(positive_triples)):
        h, r, t = (int(x) for x in positive_triples[i])
        for _ in range(max_tries):
            if corrupt_head[i]:
                h = int(rng.choice(entity_pool))
            else:
                t = int(rng.choice(entity_pool))
            if (h, r, t) not in real_triples:
                break
        negatives[i] = (h, r, t)
    return negatives


def sample_negative_path_relations(
    positive_paths: npt.NDArray[np.int64],
    relation_pool: npt.NDArray[np.int64],
    rng: np.random.Generator,
) -> npt.NDArray[np.int64]:
    """Corrupt a path batch's composed relation for the path loss's negatives.

    Replaces column 2 (``r``, the composed/direct relation) with a
    uniformly random relation from ``relation_pool`` -- no real-triple
    exclusion check, mirroring OpenEA's ``generate_neg_paths``, which has
    none either.

    Args:
        positive_paths: ``(n, 4)`` int64 array of ``(r_x, r_y, r, weight)``
            rows.
        relation_pool: That KG's own relation ids to sample replacements
            from.
        rng: Random generator, for reproducibility.

    Returns:
        ``(n, 4)`` int64 array, same shape as ``positive_paths``, with
        column 2 replaced.
    """
    negatives = positive_paths.copy()
    negatives[:, 2] = rng.choice(relation_pool, size=len(positive_paths))
    return negatives


def find_new_pairs(
    sim_mat: npt.NDArray[np.floating[Any]],
    sim_th: float,
) -> list[tuple[int, int, float]]:
    """Row-wise top-1 + threshold match finder for bootstrapping.

    For each row ``i``, keeps ``(i, argmax_j sim[i, j])`` iff
    ``sim[i, argmax_j] > sim_th``. Reimplements OpenEA's
    ``find_potential_alignment_greedily`` (=
    ``find_alignment(sim_mat, sim_th, k=1)``) directly with numpy -- with
    ``k=1``, intersecting "threshold-filtered pairs" with "each row's
    top-1 neighbor" reduces exactly to this row-wise argmax+threshold
    check, so there's no need to port ``filter_sim_mat``/
    ``search_nearest_k``'s two-pass set-intersection, or the unused (by
    IPTransE) igraph/graph-tool-based ``find_potential_alignment_mwgm``
    variant.

    This is **not** mutual/bipartite matching -- two different rows can
    map to the same column.

    Args:
        sim_mat: ``(n1, n2)`` similarity matrix (e.g. dot products of
            unit-normalized embeddings).
        sim_th: Minimum similarity for a row's best match to count.

    Returns:
        ``(row_index, col_index, similarity)`` tuples for every row whose
        best match clears ``sim_th``. Empty if ``sim_mat`` is empty.
    """
    if sim_mat.size == 0:
        return []
    best_col = np.argmax(sim_mat, axis=1)
    best_val = sim_mat[np.arange(sim_mat.shape[0]), best_col]
    rows = np.nonzero(best_val > sim_th)[0]
    return [(int(i), int(best_col[i]), float(best_val[i])) for i in rows]


def pseudo_triples_for_pair(
    entity_a: int,
    entity_b: int,
    weight: float,
    by_head: dict[int, list[tuple[int, int]]],
    by_tail: dict[int, list[tuple[int, int]]],
) -> set[tuple[int, int, int, float]]:
    """Weighted pseudo-triples generated from a newly bootstrapped pair.

    If ``entity_a`` and ``entity_b`` are plausibly the same real-world
    entity, each of ``entity_a``'s edges plausibly also holds with
    ``entity_b`` substituted in. ``by_head``/``by_tail`` must be
    ``entity_a``'s *own* KG's edge dicts; call this twice per pair (once
    per direction, with ``by_head``/``by_tail`` swapped for the other
    KG's dicts and the arguments swapped) to cover both entities' edges,
    per OpenEA's ``generate_triples_of_latent_ents``.

    Args:
        entity_a: The entity whose real edges are being copied over.
        entity_b: The (bootstrapped) counterpart substituted into those
            edges.
        weight: The pair's similarity score, carried through as the
            pseudo-triple's training weight.
        by_head: ``entity_a``'s own KG's ``head -> [(relation, tail)]``
            index.
        by_tail: ``entity_a``'s own KG's ``tail -> [(head, relation)]``
            index.

    Returns:
        ``(head, relation, tail, weight)`` pseudo-triples with
        ``entity_a`` replaced by ``entity_b``.
    """
    newly: set[tuple[int, int, int, float]] = set()
    for r, t in by_head.get(entity_a, ()):
        newly.add((entity_b, r, t, weight))
    for h, r in by_tail.get(entity_a, ()):
        newly.add((h, r, entity_b, weight))
    return newly

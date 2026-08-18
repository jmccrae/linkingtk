"""Private, plain-numpy (torch-free) helpers for
[BootEALinker][linkingtk.algorithms.ea.bootea.BootEALinker].

Ported from OpenEA's reference implementation
(https://github.com/nju-websoft/OpenEA -- ``approaches/bootea.py``,
``modules/bootstrapping/alignment_finder.py``). Independently testable
without ``torch`` installed -- see ``_bootea_torch.py`` for the
loss/training-step functions.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from linkingtk.algorithms.ea._iptranse_training import pseudo_triples_for_pair

if TYPE_CHECKING:
    import numpy.typing as npt

_ID_ENCODE_SHIFT = 1_000_000


def encode_triples(triples: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    """Pack ``(head, relation, tail)`` id rows into single ``int64`` keys for fast set membership.

    ``(h * _ID_ENCODE_SHIFT + r) * _ID_ENCODE_SHIFT + t`` -- safe for id
    spaces up to a million entities/relations each (``int64`` easily
    covers the ``~10^18`` result), and lets triple-membership checks use
    ``np.isin`` (vectorized, sort-based) instead of a Python ``set`` of
    tuples, which the real-triple-avoidance check in
    [sample_truncated_negative_triples][linkingtk.algorithms.ea._bootea_training.sample_truncated_negative_triples]
    calls often enough (BootEA's ``neg_triple_num=10``, an order of
    magnitude more than every other linker) for the pure-Python
    tuple-construction-plus-set-lookup cost to dominate real-dataset-scale
    training wall-clock time (confirmed via profiling).

    Args:
        triples: ``(n, 3)`` int64 array of ``(head, relation, tail)`` ids.

    Returns:
        ``(n,)`` int64 array of encoded keys, same row order.
    """
    if len(triples) == 0:
        return np.empty(0, dtype=np.int64)
    h, r, t = triples[:, 0], triples[:, 1], triples[:, 2]
    return (h * _ID_ENCODE_SHIFT + r) * _ID_ENCODE_SHIFT + t


def pseudo_triples_for_pairs(
    pairs: list[tuple[int, int]],
    by_head1: dict[int, list[tuple[int, int]]],
    by_tail1: dict[int, list[tuple[int, int]]],
    by_head2: dict[int, list[tuple[int, int]]],
    by_tail2: dict[int, list[tuple[int, int]]],
) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]]]:
    """Unweighted pseudo-triples for a batch of aligned pairs, split by originating KG.

    Ports ``generate_supervised_triples``/``generate_newly_triples``: for
    each ``(source, target)`` pair, every real KG1 edge touching ``source``
    becomes an extra KG1-side triple with ``source`` replaced by
    ``target`` (and symmetrically, every real KG2 edge touching ``target``
    becomes an extra KG2-side triple with ``target`` replaced by
    ``source``). Reuses
    [pseudo_triples_for_pair][linkingtk.algorithms.ea._iptranse_training.pseudo_triples_for_pair]
    for the actual substitution (called with ``weight=1.0`` and the weight
    discarded) -- unlike IPTransE's own bootstrap pseudo-triples, BootEA's
    reference treats every pseudo-triple equally, with no
    similarity-derived weight.

    This is used both for seed pairs (baked into the main structural
    triple set once, at ``fit()`` start) and for the bootstrap-found
    labeled-alignment set (retrained fresh via the separate alignment loss
    every outer iteration) -- see
    [BootEALinker][linkingtk.algorithms.ea.bootea.BootEALinker]'s module
    docstring.

    Args:
        pairs: ``(source_id, target_id)`` id pairs, already mapped to this
            run's combined entity id space.
        by_head1: KG1's own ``head -> [(relation, tail)]`` index (from real
            triples only).
        by_tail1: KG1's own ``tail -> [(head, relation)]`` index.
        by_head2: KG2's own ``head -> [(relation, tail)]`` index.
        by_tail2: KG2's own ``tail -> [(head, relation)]`` index.

    Returns:
        ``(kg1_side_triples, kg2_side_triples)`` -- deduplicated, sorted
        ``(head, relation, tail)`` id tuples. Note these can mix KG1/KG2
        ids in one triple (the substituted endpoint comes from the other
        side) -- this is deliberate, matching OpenEA's own behavior of
        folding them into each side's own triple set regardless.
    """
    triples1: set[tuple[int, int, int]] = set()
    triples2: set[tuple[int, int, int]] = set()
    for source, target in pairs:
        for h, r, t, _ in pseudo_triples_for_pair(source, target, 1.0, by_head1, by_tail1):
            triples1.add((h, r, t))
        for h, r, t, _ in pseudo_triples_for_pair(target, source, 1.0, by_head2, by_tail2):
            triples2.add((h, r, t))
    return sorted(triples1), sorted(triples2)


def find_mwgm_pairs(
    sim_mat: npt.NDArray[np.floating[Any]],
    sim_th: float,
    k: int,
) -> list[tuple[int, int, float]]:
    """Candidate filtering (threshold ∩ row-wise top-k) then maximum-weight bipartite matching.

    Ports ``find_potential_alignment_mwgm``/``find_alignment``: a pair
    ``(i, j)`` is a *candidate* iff ``sim[i, j] > sim_th`` and ``j`` is
    among row ``i``'s ``k`` largest entries (unlike
    [find_new_pairs][linkingtk.algorithms.ea._iptranse_training.find_new_pairs]'s
    plain row-argmax, this keeps up to ``k`` candidates per row, not just
    the single best). Candidates then go through **maximum-weight
    bipartite matching**, via ``scipy.optimize.linear_sum_assignment``
    (already a dependency) rather than OpenEA's ``graph_tool``/``igraph``
    -- a deliberate, documented dependency substitution (same rationale as
    IMUSE #32's ``rapidfuzz``): build a dense cost submatrix over just the
    distinct row/col indices appearing in the candidate set (non-candidate
    cells get a large sentinel cost so they're only chosen if the
    assignment is forced to), solve it, then drop any returned pair that
    wasn't an actual candidate -- this gives the same "true 1-to-1
    assignment, greedy pairs discarded if a better global assignment
    exists" property without the extra dependency.

    Uses a plain per-row ``argsort`` for the top-k step rather than
    OpenEA's ``argpartition``, for readability and to sidestep
    ``argpartition``'s "kth must be less than array size" edge case on
    small matrices -- same simplification precedent as
    ``sample_negative_triples``.

    Args:
        sim_mat: ``(n1, n2)`` similarity matrix.
        sim_th: Minimum similarity for a pair to be a candidate at all.
        k: Max candidates kept per row (OpenEA's published value is
            ``10``).

    Returns:
        ``(row_index, col_index, similarity)`` tuples for the matched
        pairs. Empty if ``sim_mat`` is empty or no candidates clear
        ``sim_th``.
    """
    if sim_mat.size == 0:
        return []
    candidates = _threshold_and_topk_candidates(sim_mat, sim_th, k)
    if not candidates:
        return []

    rows = sorted({i for i, _ in candidates})
    cols = sorted({j for _, j in candidates})
    row_index = {i: idx for idx, i in enumerate(rows)}
    col_index = {j: idx for idx, j in enumerate(cols)}

    sentinel = 1e6
    cost = np.full((len(rows), len(cols)), sentinel, dtype=np.float64)
    for i, j in candidates:
        cost[row_index[i], col_index[j]] = -float(sim_mat[i, j])

    row_assign, col_assign = linear_sum_assignment(cost)
    matched: list[tuple[int, int, float]] = []
    for r, c in zip(row_assign, col_assign, strict=True):
        i, j = rows[r], cols[c]
        if (i, j) in candidates:
            matched.append((i, j, float(sim_mat[i, j])))
    return matched


def _threshold_and_topk_candidates(
    sim_mat: npt.NDArray[np.floating[Any]],
    sim_th: float,
    k: int,
) -> set[tuple[int, int]]:
    """Row-wise top-``k`` columns, threshold-checked only within that (small) working set.

    Equivalent to intersecting "above ``sim_th``" with "row's top-``k``"
    (set intersection doesn't care which filter is applied first), but
    avoids ever materializing the full above-threshold set over the whole
    matrix or looping per row in Python -- one vectorized
    ``argpartition`` call gets every row's top-``k`` column indices at
    once (same fix as
    [compute_truncated_neighbors][linkingtk.algorithms.ea._bootea_training.compute_truncated_neighbors]'s;
    a per-row ``argsort`` loop here measurably bottlenecked real-dataset-
    scale bootstrapping rounds -- 0% GPU utilization while this ran).
    """
    if k <= 0:
        return {(int(i), int(j)) for i, j in zip(*np.where(sim_mat > sim_th), strict=True)}
    n_rows, n_cols = sim_mat.shape
    k = min(k, n_cols)
    top_cols = np.argpartition(-sim_mat, k - 1, axis=1)[:, :k]
    rows = np.repeat(np.arange(n_rows), k)
    cols = top_cols.reshape(-1)
    above_threshold = sim_mat[rows, cols] > sim_th
    kept_rows, kept_cols = rows[above_threshold], cols[above_threshold]
    return {(int(i), int(j)) for i, j in zip(kept_rows, kept_cols, strict=True)}


def edit_labeled_alignment(
    pre_alignment: dict[int, int],
    curr_alignment: set[tuple[int, int]],
    sim_mat: npt.NDArray[np.floating[Any]],
) -> dict[int, int]:
    """Merge this round's matched pairs into the accumulated labeled alignment, with editing.

    Ports ``update_labeled_alignment_x`` (composed with)
    ``update_labeled_alignment_y``:

    1. For each ``(i, j)`` freshly matched this round: if ``i`` already has
       an earlier label, replace it only if the new pair's similarity is
       ``>=`` the old label's similarity (walks back an earlier mistaken
       label when a later round finds better evidence); otherwise it's a
       brand-new label.
    2. Resolve multiple sources now claiming the same target: keep only
       the claimant with the highest similarity to that target.

    Args:
        pre_alignment: The accumulated ``source_index -> target_index``
            labeled alignment from previous rounds (``{}`` on the first
            round).
        curr_alignment: This round's freshly matched ``(source_index,
            target_index)`` pairs, e.g. from
            [find_mwgm_pairs][linkingtk.algorithms.ea._bootea_training.find_mwgm_pairs].
        sim_mat: The same similarity matrix ``curr_alignment`` was matched
            against -- also used to compare candidate labels' confidence
            during editing. Indices in both ``pre_alignment`` and
            ``curr_alignment`` must index into this matrix.

    Returns:
        The updated, edited ``source_index -> target_index`` alignment.
    """
    updated = dict(pre_alignment)
    for i, j in curr_alignment:
        if i in updated:
            if sim_mat[i, j] >= sim_mat[i, updated[i]]:
                updated[i] = j
        else:
            updated[i] = j

    by_target: dict[int, list[int]] = defaultdict(list)
    for i, j in updated.items():
        by_target[j].append(i)

    result: dict[int, int] = {}
    for j, sources in by_target.items():
        best_i = max(sources, key=lambda i: sim_mat[i, j])
        result[best_i] = j
    return result


def sample_truncated_negative_triples(
    positive_triples: npt.NDArray[np.int64],
    neighbor_candidates: dict[int, npt.NDArray[np.int64]],
    fallback_pool: npt.NDArray[np.int64],
    real_triples: npt.NDArray[np.int64],
    rng: np.random.Generator,
    max_tries: int = 10,
) -> npt.NDArray[np.int64]:
    """Like ``sample_negative_triples`` but corrupted from each entity's own truncated neighbor set.

    See
    [sample_negative_triples][linkingtk.algorithms.ea._iptranse_training.sample_negative_triples].

    Ports ``generate_neg_triples_fast``'s ``neighbor``-restricted
    candidate behavior: the replacement head/tail is drawn from
    ``neighbor_candidates.get(entity, fallback_pool)`` -- typically each
    entity's current K-nearest-neighbor set by embedding similarity (see
    [compute_truncated_neighbors][linkingtk.algorithms.ea._bootea_training.compute_truncated_neighbors]),
    which produces harder negatives than uniform sampling. Falls back to
    ``fallback_pool`` for an entity with no computed neighbor set yet
    (e.g. before the first neighbor refresh).

    Vectorized rather than a plain per-row retry loop (unlike
    [sample_negative_triples][linkingtk.algorithms.ea._iptranse_training.sample_negative_triples]'s
    established simplification): BootEA's own published ``neg_triple_num``
    is ``10`` (vs. every other linker's ``1``), so this gets called an
    order of magnitude more per training step -- a scalar-Python retry
    loop measurably bottlenecks real-dataset-scale training (confirmed via
    profiling on EN-FR-15K-V1: 0% GPU utilization, all wall-clock time
    spent in this function). Draws are batched per unique corrupted
    entity via numpy fancy-indexing; the real-triple-avoidance check uses
    [encode_triples][linkingtk.algorithms.ea._bootea_training.encode_triples]
    + ``np.isin`` (vectorized) rather than a Python ``set`` of tuples, and
    each retry pass only re-draws rows that actually collided (typically
    few, and shrinking), not the whole array every pass.

    Args:
        positive_triples: ``(n, 3)`` int64 array of ``(head, rel, tail)``
            ids.
        neighbor_candidates: ``entity_id -> candidate replacement ids``
            (all arrays the same length, as
            [compute_truncated_neighbors][linkingtk.algorithms.ea._bootea_training.compute_truncated_neighbors]
            produces).
        fallback_pool: Candidate pool used when an entity has no entry in
            ``neighbor_candidates``.
        real_triples: Sorted
            [encode_triples][linkingtk.algorithms.ea._bootea_training.encode_triples]-encoded
            keys to avoid reproducing (e.g. ``KGContext.real_triples``).
        rng: Random generator, for reproducibility.
        max_tries: Retry passes over still-colliding rows before accepting
            the last draw regardless.

    Returns:
        ``(n, 3)`` int64 array of corrupted triples, same shape as
        ``positive_triples``.
    """
    n = len(positive_triples)
    negatives = positive_triples.copy()
    if n == 0:
        return negatives

    corrupt_head = rng.random(n) < 0.5
    corrupt_col = np.where(corrupt_head, 0, 2)
    corrupt_entities = positive_triples[np.arange(n), corrupt_col]

    if neighbor_candidates:
        candidate_width = next(iter(neighbor_candidates.values())).shape[0]
        default = (
            fallback_pool
            if len(fallback_pool) == candidate_width
            else np.resize(fallback_pool, candidate_width)
        )
        unique_entities, inverse = np.unique(corrupt_entities, return_inverse=True)
        candidate_rows = np.stack(
            [neighbor_candidates.get(int(e), default) for e in unique_entities]
        )

        def draw(rows: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
            cols = rng.integers(0, candidate_width, size=len(rows))
            return candidate_rows[inverse[rows], cols]
    else:

        def draw(rows: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
            return rng.choice(fallback_pool, size=len(rows))

    all_rows = np.arange(n)
    negatives[all_rows, corrupt_col] = draw(all_rows)
    if len(real_triples) == 0:
        return negatives

    # Only re-check rows still colliding after the previous pass, not the
    # whole array every time -- collisions are rare, so this set of rows
    # shrinks fast. Rescanning everything each pass (an earlier version of
    # this function did) turns `max_tries` full-array checks into the
    # actual bottleneck at real-dataset scale.
    check_rows = all_rows
    for _ in range(max_tries):
        if len(check_rows) == 0:
            break
        # `assume_unique=True` is safe here (query-side duplicates don't
        # affect per-element correctness -- verified) and matters a lot:
        # without it, `np.isin` re-runs `np.unique` over the *whole*
        # `real_triples` array (tens of thousands of rows) on every call,
        # which was the actual remaining bottleneck after switching from a
        # Python set.
        colliding = np.isin(
            encode_triples(negatives[check_rows]), real_triples, assume_unique=True
        )
        check_rows = check_rows[colliding]
        if len(check_rows) == 0:
            break
        negatives[check_rows, corrupt_col[check_rows]] = draw(check_rows)
    return negatives


def compute_truncated_neighbors(
    entity_embeds: npt.NDArray[np.floating[Any]],
    entity_ids: npt.NDArray[np.int64],
    k: int,
) -> dict[int, npt.NDArray[np.int64]]:
    """Each of ``entity_ids``' own top-``k`` nearest neighbors, restricted to ``entity_ids`` itself.

    Ports ``find_neighbours``: similarity is a plain dot product (callers
    pass already L2-normalized embeddings), restricted to one KG's own
    entities (``entity_ids``) so negative-sampling replacements never
    cross the KG boundary. Matches the reference in not excluding an
    entity from its own neighbor set (self-similarity is always highest,
    so an entity typically appears as its own top neighbor) --
    harmless downstream, since
    [sample_truncated_negative_triples][linkingtk.algorithms.ea._bootea_training.sample_truncated_negative_triples]'s
    real-triple check just retries if a self-corruption reproduces the
    positive triple.

    Args:
        entity_embeds: Full ``(num_entities, dim)`` embedding table.
        entity_ids: This KG's own entity ids (a subset of
            ``entity_embeds``'s row indices).
        k: Number of neighbors to keep per entity (OpenEA's
            ``truncated_epsilon``-derived count).

    Returns:
        ``entity_id -> (k,) int64 array`` of neighbor candidate ids, one
        entry per id in ``entity_ids``. Empty dict if ``entity_ids`` is
        empty.
    """
    if len(entity_ids) == 0:
        return {}
    k = min(k, len(entity_ids))
    sub = entity_embeds[entity_ids]
    sim = sub @ sub.T
    # Vectorized top-k per row (unordered within the top-k, which is fine
    # -- callers only need a candidate set) via one `argpartition` call
    # over the whole matrix, rather than a per-entity `argsort` loop --
    # the loop measurably bottlenecked real-dataset-scale training
    # (~15K entities/side), called every outer bootstrapping iteration.
    top = np.argpartition(-sim, k - 1, axis=1)[:, :k]
    neighbor_ids = entity_ids[top]
    return {int(entity_id): neighbor_ids[row] for row, entity_id in enumerate(entity_ids)}

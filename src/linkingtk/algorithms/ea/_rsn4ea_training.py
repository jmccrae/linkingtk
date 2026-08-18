"""Private training helpers for
[RSN4EALinker][linkingtk.algorithms.ea.rsn4ea.RSN4EALinker].

Pure, independently-testable functions ported from OpenEA's reference
implementation (https://github.com/nju-websoft/OpenEA --
``approaches/rsn4ea.py``'s ``BasicReader``/``BasicSampler``). Not part of
the public API -- see ``rsn4ea.py`` for the orchestrating class and the
deviations from OpenEA's own code documented in its docstring.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt


def build_augmented_kb(
    triples: npt.NDArray[np.int64],
    num_entities: int,
    num_relations: int,
    seed_pairs: list[tuple[int, int]],
) -> tuple[npt.NDArray[np.int64], int]:
    """Reverse-edge doubling + cross-KG alias expansion.

    Mirrors OpenEA's ``BasicReader.read``/``add_align_infor``/``add_weight``:
    every triple gets a reverse edge ``(t, r + num_relations, h)`` (so walks
    can traverse edges in either direction, doubling the relation
    vocabulary); then, over that doubled set, each triple's head/tail is
    additionally substituted with its seed-pair alias wherever one exists
    (bidirectional -- a seed pair ``(a, b)`` aliases ``a`` to ``b`` *and*
    ``b`` to ``a``). Every triple therefore contributes up to 4 variants:
    ``(h, r, t)``, ``(alias(h), r, t)``, ``(h, r, alias(t))``,
    ``(alias(h), r, alias(t))`` -- deduplicated at the end (a triple with
    neither endpoint aliased contributes the same variant 4 times, which
    collapses to 1). A relation is never aliased -- RSN4EA carries no
    relation-alignment signal (OpenEA's own ``add_weight`` also builds
    relation-alias variants, but they're always empty in practice since its
    ``rel_mapping`` is always an empty table for this task -- not ported,
    see ``rsn4ea.py``'s module docstring).

    This alias substitution is the *entire* cross-KG alignment mechanism:
    a walk that reaches a seed-aligned entity id can continue through the
    other KG's own edges from that point, since the two ids are now
    mutually substitutable throughout the walkable graph.

    Args:
        triples: ``(n, 3)`` int64 array of ``(head, relation, tail)`` ids,
            over the combined (disjoint-per-KG) id space.
        num_entities: Total number of distinct entity ids (sizes the alias
            lookup array).
        num_relations: Number of *original* (pre-doubling) relation ids.
        seed_pairs: ``(source_id, target_id)`` id pairs.

    Returns:
        ``(augmented_kb, doubled_num_relations)`` -- a deduplicated
        ``(m, 3)`` int64 array of every real, reverse, and alias-substituted
        triple, and the post-doubling relation count.
    """
    doubled_num_relations = num_relations * 2
    if len(triples) == 0:
        return np.empty((0, 3), dtype=np.int64), doubled_num_relations

    reverse = np.stack([triples[:, 2], triples[:, 1] + num_relations, triples[:, 0]], axis=1)
    doubled = np.concatenate([triples, reverse], axis=0)

    alias = np.arange(num_entities, dtype=np.int64)
    for source, target in seed_pairs:
        alias[source] = target
        alias[target] = source

    h, r, t = doubled[:, 0], doubled[:, 1], doubled[:, 2]
    ah, at = alias[h], alias[t]
    variants = np.concatenate(
        [
            doubled,
            np.stack([ah, r, t], axis=1),
            np.stack([h, r, at], axis=1),
            np.stack([ah, r, at], axis=1),
        ],
        axis=0,
    )
    augmented: npt.NDArray[np.int64] = np.unique(variants, axis=0)
    return augmented, doubled_num_relations


def sample_paths(
    kb: npt.NDArray[np.int64],
    seed_pairs: list[tuple[int, int]],
    rng: np.random.Generator,
    max_length: int,
    alpha: float,
    beta: float,
    repeat_times: int,
    max_paths: int | None = None,
) -> npt.NDArray[np.int64]:
    """Biased random walks over ``kb``, matching OpenEA's ``BasicSampler.sample_paths``.

    ``repeat_times`` walks are seeded per row of ``kb`` -- that row's own
    ``(h, r, t)`` is the walk's un-sampled first hop. Every subsequent hop
    picks a ``(relation, tail)`` continuation from the current tail's own
    outgoing edges in ``kb``, weighted by:

    - ``beta``: continuations whose tail is a "cross-KG" entity (appears in
      ``seed_pairs`` at all, either side) get weight ``beta``; others get
      ``1 - beta``.
    - ``alpha`` (every hop *after* the first): continuations whose exact
      ``(relation, tail)`` was *also* directly reachable from the walk's
      own starting entity get weight ``1 - alpha``; others get ``alpha`` --
      biasing exploration away from simply retracing the start node's own
      immediate neighborhood.

    The two weights multiply for hops after the first. A walk that hits a
    dead end (its current tail has no outgoing edges in ``kb``) before
    reaching ``max_length`` columns is dropped entirely rather than padded
    -- OpenEA's own reference has no dead-end handling either (it would
    simply crash); this only matters for entities with zero out-degree even
    after ``build_augmented_kb``'s reverse-doubling, which real datasets
    essentially never have.

    Args:
        kb: ``(m, 3)`` int64 augmented triples, from
            [build_augmented_kb][linkingtk.algorithms.ea._rsn4ea_training.build_augmented_kb].
        seed_pairs: Same pairs used to build ``kb``'s aliasing -- reused
            here for the ``beta`` cross-KG bias.
        rng: Random generator, for reproducibility.
        max_length: Sequence length (must be odd and >= 3): entities at
            even positions, relations at odd positions.
        alpha: Depth-bias weight, see above.
        beta: Cross-KG bias weight, see above.
        repeat_times: Walks sampled per row of ``kb``.
        max_paths: Optional cap on the number of returned walks (random
            subsample, fixed for the call) -- a pragmatic safety valve at
            real-dataset scale, the same precedent as
            [IPTransELinker][linkingtk.algorithms.ea.iptranse.IPTransELinker]'s
            ``bootstrap_pool_size``. ``None`` (default) returns every
            successfully-sampled walk.

    Returns:
        ``(num_paths, max_length)`` int64 array, alternating entity/relation
        ids, starting and ending on an entity.
    """
    by_head: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for h, r, t in kb:
        by_head[int(h)].append((int(r), int(t)))

    cross_kg = {entity for pair in seed_pairs for entity in pair}
    direct_edge = {(int(h), int(r), int(t)) for h, r, t in kb}

    def choose(head: int, start: int | None) -> tuple[int, int] | None:
        candidates = by_head.get(head)
        if not candidates:
            return None
        weights = np.array(
            [beta if t in cross_kg else (1.0 - beta) for _, t in candidates], dtype=np.float64
        )
        if start is not None:
            depth = np.array(
                [(1.0 - alpha) if (start, r, t) in direct_edge else alpha for r, t in candidates],
                dtype=np.float64,
            )
            weights = weights * depth
        total = weights.sum()
        if total <= 0:
            weights = np.ones(len(candidates))
            total = weights.sum()
        weights = weights / total
        index = rng.choice(len(candidates), p=weights)
        return candidates[index]

    num_hops = (max_length - 3) // 2
    walks: list[list[int]] = []
    for h, r, t in kb:
        h, r, t = int(h), int(r), int(t)
        for _ in range(repeat_times):
            seq = [h, r, t]
            start = h
            current_tail = t
            ok = True
            for hop in range(num_hops):
                step = choose(current_tail, None if hop == 0 else start)
                if step is None:
                    ok = False
                    break
                next_r, next_t = step
                seq.extend([next_r, next_t])
                current_tail = next_t
            if ok:
                walks.append(seq)

    if not walks:
        return np.empty((0, max_length), dtype=np.int64)

    paths = np.array(walks, dtype=np.int64)
    if max_paths is not None and len(paths) > max_paths:
        chosen = rng.choice(len(paths), size=max_paths, replace=False)
        paths = paths[chosen]
    return paths

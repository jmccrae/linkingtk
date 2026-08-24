"""Private, plain-numpy (torch-free) helpers for
[AliNetLinker][linkingtk.algorithms.ea.alinet.AliNetLinker].

Ported from OpenEA's reference implementation
(https://github.com/nju-websoft/OpenEA -- ``approaches/alinet.py``'s
``generate_2hop_triples``/``enhance_triples``/``no_weighted_adj``, and
``AliNet.augment``/``augment_neighborhood``'s bootstrapping loop).
Independently testable without ``torch`` installed -- see
``_alinet_torch.py`` for the GCN/attention/highway model and training-step
functions.

Reuses two functions already ported for sibling linkers rather than
reimplementing them:
[build_primal_adjacency][linkingtk.algorithms.ea._rdgcn_training.build_primal_adjacency]
(RDGCN's unweighted-symmetric-adjacency builder is exactly OpenEA's own
``no_weighted_adj``'s edge-set construction -- used here for the 1-hop
adjacency, given a triple list; the 2-hop adjacency's own
[pairs_to_symmetric_adjacency][linkingtk.algorithms.ea._alinet_training.pairs_to_symmetric_adjacency]
below is the same construction over a raw pair set instead) and
[find_new_pairs][linkingtk.algorithms.ea._iptranse_training.find_new_pairs]
(IPTransE's row-argmax-plus-threshold match finder is exactly OpenEA's own
``find_alignment(sim_mat, sim_th, k=1)``, which is what AliNet's own
``augment`` calls too).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import TYPE_CHECKING, Any

import numpy as np

from linkingtk.algorithms.ea._iptranse_training import find_new_pairs

if TYPE_CHECKING:
    import numpy.typing as npt

_SKIP_TOP_K_PATTERNS = 5


def enhance_triples(
    triples1: npt.NDArray[np.int64],
    triples2: npt.NDArray[np.int64],
    seed_pairs: list[tuple[int, int]],
) -> tuple[set[tuple[int, int, int]], set[tuple[int, int, int]]]:
    """Cross-KG-inferred triples, per OpenEA's ``enhance_triples``.

    For each KG1 triple ``(h1, r, t1)`` whose *both* entities have a seed
    counterpart (``h2 = links[h1]``, ``t2 = links[t1]``), infers a matching
    KG2 triple ``(h2, r, t2)`` -- unless KG2 already has a direct ``h2 ->
    t2`` edge (avoiding redundant edges). Mirrored for KG2 -> KG1. This is
    what lets a single seed pair "bridge" one KG's own relational
    structure into the other's, making the eventual 1-hop adjacency
    cross-KG-connected rather than two disjoint components.

    Args:
        triples1: KG1's own ``(head, relation, tail)`` id triples (already
            mapped into the combined id space).
        triples2: KG2's own triples, same id space.
        seed_pairs: ``(kg1_id, kg2_id)`` known-correct pairs.

    Returns:
        ``(enhanced_triples1, enhanced_triples2)`` -- inferred triples to
        add to KG1's/KG2's own triple sets before adjacency construction.
    """
    links1 = dict(seed_pairs)
    links2 = {target: source for source, target in seed_pairs}

    out_edges1: dict[int, set[int]] = defaultdict(set)
    for head, _relation, tail in triples1:
        out_edges1[int(head)].add(int(tail))
    out_edges2: dict[int, set[int]] = defaultdict(set)
    for head, _relation, tail in triples2:
        out_edges2[int(head)].add(int(tail))

    enhanced2: set[tuple[int, int, int]] = set()
    for head, relation, tail in triples1:
        head2 = links1.get(int(head))
        tail2 = links1.get(int(tail))
        if head2 is not None and tail2 is not None and tail2 not in out_edges2.get(head2, set()):
            enhanced2.add((head2, int(relation), tail2))

    enhanced1: set[tuple[int, int, int]] = set()
    for head, relation, tail in triples2:
        head1 = links2.get(int(head))
        tail1 = links2.get(int(tail))
        if head1 is not None and tail1 is not None and tail1 not in out_edges1.get(head1, set()):
            enhanced1.add((head1, int(relation), tail1))

    return enhanced1, enhanced2


def generate_2hop_pairs(
    triples: npt.NDArray[np.int64], skip_top_k: int = _SKIP_TOP_K_PATTERNS
) -> set[tuple[int, int]]:
    """Entity pairs reachable by a 2-hop path, filtered by relation-pattern frequency.

    Ports OpenEA's ``generate_2hop_triples``: finds every ``(h, m, t)``
    path (``h -r1-> m -r2-> t``) whose ``(h, t)`` pair is **not** already a
    direct 1-hop edge (either direction), groups them by ``(r1, r2)``
    relation-composition pattern, and keeps only paths whose pattern is
    **not** among the ``skip_top_k`` most frequent -- confirmed from
    reading ``generate_2hop_triples`` directly: it computes (but never
    uses) a ``p=0.05``-proportional cutoff (``num = int(p *
    len(relation_patterns))``) and its commented-out original loop
    (``# for i in range(20, num)``), but the *actual* active loop is
    ``for i in range(5, len(relation_patterns))`` -- i.e. only the top-5
    most frequent patterns are ever excluded, `num`/`p` are dead code.
    Ported as what's actually executed, not the apparent original intent.

    The relation ids and the synthetic self-loop/composed-relation triples
    OpenEA's own version builds around this selection are dropped here --
    downstream
    ([pairs_to_symmetric_adjacency][linkingtk.algorithms.ea._alinet_training.pairs_to_symmetric_adjacency])
    only needs ``(h, t)`` connectivity, not relation ids, and the
    synthetic self-loop triple `(head, 0, head)` OpenEA's own code adds is
    already redundant with
    [normalize_adjacency_coo][linkingtk.utils.sparse_gcn.normalize_adjacency_coo]'s
    own ``add_self_loops``.

    Args:
        triples: One KG's own ``(head, relation, tail)`` id triples
            (called separately per KG, matching OpenEA's own per-``kg``
            usage -- 2-hop paths are never combined across KGs here).
        skip_top_k: Number of most-frequent ``(r1, r2)`` patterns to
            exclude.

    Returns:
        ``{(head, tail)}`` pairs reachable by a selected 2-hop path.
    """
    by_head: dict[int, list[tuple[int, int]]] = defaultdict(list)
    out_neighbors: dict[int, set[int]] = defaultdict(set)
    in_neighbors: dict[int, set[int]] = defaultdict(set)
    for head, relation, tail in triples:
        head, relation, tail = int(head), int(relation), int(tail)
        by_head[head].append((relation, tail))
        out_neighbors[head].add(tail)
        in_neighbors[tail].add(head)

    pattern_counts: Counter[tuple[int, int]] = Counter()
    quadruples: list[tuple[int, int, int, int]] = []
    for head, relation1, mid in triples:
        head, relation1, mid = int(head), int(relation1), int(mid)
        for relation2, tail in by_head.get(mid, []):
            if tail in out_neighbors.get(head, set()) or head in in_neighbors.get(tail, set()):
                continue
            pattern_counts[(relation1, relation2)] += 1
            quadruples.append((head, relation1, relation2, tail))

    ranked_patterns = [pattern for pattern, _count in pattern_counts.most_common()]
    selected_patterns = set(ranked_patterns[skip_top_k:])

    return {
        (head, tail)
        for head, relation1, relation2, tail in quadruples
        if (relation1, relation2) in selected_patterns
    }


def pairs_to_symmetric_adjacency(
    pairs: set[tuple[int, int]], num_entities: int
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """Unweighted symmetric adjacency from a raw ``{(head, tail)}`` pair set.

    Args:
        pairs: Entity-id pairs, e.g. from
            [generate_2hop_pairs][linkingtk.algorithms.ea._alinet_training.generate_2hop_pairs].
        num_entities: Total entity count (both KGs combined).

    Returns:
        ``(indices, values)`` -- ``(2, nnz)`` int64 / ``(nnz,)`` float64,
        both directions present, ready for
        [normalize_adjacency_coo][linkingtk.utils.sparse_gcn.normalize_adjacency_coo].
    """
    symmetric = {(head, tail) for head, tail in pairs} | {(tail, head) for head, tail in pairs}
    if not symmetric:
        return np.empty((2, 0), dtype=np.int64), np.empty(0, dtype=np.float64)
    rows = np.array([pair[0] for pair in symmetric], dtype=np.int64)
    cols = np.array([pair[1] for pair in symmetric], dtype=np.int64)
    values = np.ones(len(symmetric), dtype=np.float64)
    return np.stack([rows, cols]), values


def sample_uniform_cross_kg_negatives(
    entities1: list[int],
    entities2: list[int],
    batch_size: int,
    neg_triple_num: int,
    rng: np.random.Generator,
    exclude_pairs: set[tuple[int, int]],
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Fully-random cross-KG negative pairs, per OpenEA's ``generate_input_batch``'s uniform branch.

    Unlike GCN-Align's/RDGCN's negatives (each anchored to a real positive
    pair's own entity), AliNet's uniform negatives are **independently**
    drawn from each KG's own entity pool -- ``neg_ent1[i]``/``neg_ent2[i]``
    are unrelated random draws, not a corruption of a specific positive
    pair. Pairs that happen to collide with a known positive
    (``exclude_pairs`` -- the current seed set, so a "negative" is never
    accidentally a real alignment) are dropped.

    **Deliberate scope reduction**: OpenEA's own published config uses
    ``neg_sampling: "truncated"``, switching to nearest-neighbor-restricted
    negatives (via ``find_neighbors``) after the first validation
    checkpoint -- not implemented here; every epoch uses this uniform
    scheme. See the module docstring on
    [alinet][linkingtk.algorithms.ea.alinet] for why.

    Args:
        entities1: Candidate KG1-side entity ids to draw from (OpenEA's
            own pool is ``sup_ent1 + ref_ent1`` -- every entity, not just
            currently-seeded ones).
        entities2: Candidate KG2-side entity ids.
        batch_size: Positive pairs in this batch.
        neg_triple_num: Negatives drawn per positive pair (``k``).
        rng: Source of randomness.
        exclude_pairs: Pairs to drop if drawn (the current seed set).

    Returns:
        ``(neg_left, neg_right)``, each up to ``(batch_size * k,)`` int64
        (shorter if some draws were excluded).
    """
    size = batch_size * neg_triple_num
    left = rng.choice(entities1, size=size)
    right = rng.choice(entities2, size=size)
    keep = [i for i in range(size) if (int(left[i]), int(right[i])) not in exclude_pairs]
    return left[keep].astype(np.int64), right[keep].astype(np.int64)


def run_bootstrapping_round(
    embeddings: npt.NDArray[np.floating[Any]],
    seed_pairs: list[tuple[int, int]],
    unaligned1: list[int],
    unaligned2: list[int],
    sim_th: float = 0.0,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """One bootstrapping round: find new high-confidence pairs, expand the seed set.

    Ports OpenEA's ``augment``/``augment_neighborhood``: cosine similarity
    between every still-unaligned KG1 entity and every still-unaligned KG2
    entity, [find_new_pairs][linkingtk.algorithms.ea._iptranse_training.find_new_pairs]
    (= OpenEA's own ``find_alignment(sim_mat, sim_th, k=1)``, row-wise
    top-1 plus threshold) for one-sided candidate matches, then a
    column-collision resolution step (keep only the highest-similarity
    match per KG2 target) standing in for OpenEA's separate, stateful
    ``update_labeled_alignment_x``/``_y`` editing passes -- once a pair is
    accepted here it's added to the seed set permanently, rather than
    remaining revisable in later rounds the way OpenEA's own labeled-
    alignment dict does. A simpler, one-directional approximation of the
    same "avoid many-to-one collisions" property, not a full port.

    Args:
        embeddings: ``(num_entities, dim)`` current entity embeddings
            (any consistent representation -- L2-normalized here before
            comparing).
        seed_pairs: Current seed set (both original and previously
            bootstrapped pairs).
        unaligned1: KG1 entity ids not yet in ``seed_pairs``.
        unaligned2: KG2 entity ids not yet in ``seed_pairs``.
        sim_th: Minimum cosine similarity for a match to count. OpenEA's
            published value is ``0.0`` (no real floor -- reciprocal/
            collision resolution is the operative filter, same posture as
            [find_mutual_pairs][linkingtk.algorithms.ea._iptranse_training.find_mutual_pairs]'s
            docstring).

    Returns:
        ``(new_seed_pairs, remaining_unaligned1, remaining_unaligned2)``.
        ``new_seed_pairs`` is ``seed_pairs`` plus every pair accepted this
        round; the two remaining lists have those entities removed.
    """
    if not unaligned1 or not unaligned2:
        return seed_pairs, unaligned1, unaligned2

    embeds1 = embeddings[unaligned1]
    embeds2 = embeddings[unaligned2]
    embeds1 = embeds1 / np.clip(np.linalg.norm(embeds1, axis=1, keepdims=True), 1e-12, None)
    embeds2 = embeds2 / np.clip(np.linalg.norm(embeds2, axis=1, keepdims=True), 1e-12, None)
    sim_mat = embeds1 @ embeds2.T

    matches = find_new_pairs(sim_mat, sim_th)
    best_by_col: dict[int, tuple[int, float]] = {}
    for row, col, score in matches:
        if col not in best_by_col or score > best_by_col[col][1]:
            best_by_col[col] = (row, score)

    new_pairs = [(unaligned1[row], unaligned2[col]) for col, (row, _score) in best_by_col.items()]
    new_ids1 = {pair[0] for pair in new_pairs}
    new_ids2 = {pair[1] for pair in new_pairs}
    remaining1 = [entity for entity in unaligned1 if entity not in new_ids1]
    remaining2 = [entity for entity in unaligned2 if entity not in new_ids2]
    return seed_pairs + new_pairs, remaining1, remaining2

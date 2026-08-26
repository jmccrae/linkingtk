"""node2vec structural embeddings for
[SimpleHHEALinker][linkingtk.algorithms.ea.simple_hhea.SimpleHHEALinker].

Ports the biased-random-walk structural-embedding stage of
https://github.com/DataArcTech/Simple-HHEA
(`feature_perprocessing/preproccess.py` + `longterm/{main.py,node2vec.py}`
+ `get_deep_emb.py`) using the `node2vec` PyPI package -- a maintained,
tested implementation of the same Grover & Leskovec biased-walk algorithm
the reference hand-rolls in `longterm/node2vec.py` -- rather than
hand-porting its alias-sampling walk code (see
[[feedback_prefer_real_deps_over_handrolled]]).

Two deliberate simplifications vs. the reference (flagged in #62's plan,
not silent):

- Walks are entity-only, with no relation-id interleaving. The reference
  splices relation ids into each walk (entity-relation-entity-...) before
  handing it to Word2Vec, but the paper text itself never mentions this
  and it isn't ablated separately -- treated here as a reference-code-only
  embellishment on top of "a biased random walk method", not the core
  contribution.
- The "merge known-aligned entities into one graph node before walking"
  trick (the reference's `node2same`) is driven directly by the
  `train_pairs` [SimpleHHEALinker.fit][linkingtk.algorithms.ea.simple_hhea.SimpleHHEALinker.fit]
  already receives as `ground_truth`, instead of reverse-engineering the
  reference's `ref_ent_ids[:1500]` file-position slice. Each pair's
  *target* id is remapped onto its *source* id directly (one fewer moving
  part than the reference's brand-new shared id per pair) -- functionally
  the same "one shared node" outcome.

One addition the reference doesn't have at all, found necessary against
the real ICEWS-WIKI dataset (not a toy/synthetic graph): its combined
graph has an extremely power-law-skewed degree distribution (median node
degree 18, but the single highest-degree node has 273,317 edges --
confirmed by direct measurement, not assumed). `node2vec`'s (and the
reference's own hand-rolled `longterm/node2vec.py`'s) alias-table
transition-probability precompute is `O(degree)` per edge, so a hub node
alone costs `O(degree^2)` -- tens of billions of operations for that one
node, which never finished in practice (confirmed: killed after several
CPU-bound minutes stuck in exactly that precompute step, well before any
walk was even simulated).
[cap_node_degree][linkingtk.algorithms.ea._simple_hhea_structure.cap_node_degree]
randomly subsamples any node's incident edges down to `max_degree` before
handing the graph to `Node2Vec` -- a standard, well-precedented mitigation
for random-walk embeddings on scale-free graphs (not something the
reference needed to consider, or that a toy-scale unit-test fixture would
ever surface).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from linkingtk.exceptions import OptionalDependencyError
from linkingtk.utils.graph import Triple

if TYPE_CHECKING:
    import numpy.typing as npt


def _merge_map(train_pairs: list[tuple[str, str]]) -> dict[str, str]:
    """Target id -> source id, for each known-aligned training pair.

    Collapses both sides of a known alignment onto one shared graph node
    (the source id) before random walks are simulated.
    """
    return dict(train_pairs)


def cap_node_degree(
    edges: list[tuple[str, str]], max_degree: int, rng: np.random.Generator
) -> list[tuple[str, str]]:
    """Randomly subsamples `edges` so no node ends up with more than `max_degree`.

    Shuffles `edges` (so which ones survive isn't biased toward triple
    order/relation type), then greedily keeps each edge only while both
    endpoints are still under `max_degree` -- a node already at the cap
    loses only its "excess" edges, not a uniform fraction of all of them,
    so a rare hub node is trimmed while everything else is left alone.
    """
    order = rng.permutation(len(edges))
    counts: dict[str, int] = {}
    kept: list[tuple[str, str]] = []
    for index in order:
        u, v = edges[index]
        if counts.get(u, 0) >= max_degree or counts.get(v, 0) >= max_degree:
            continue
        kept.append((u, v))
        counts[u] = counts.get(u, 0) + 1
        counts[v] = counts.get(v, 0) + 1
    return kept


def build_structure_embeddings(
    entity_ids: list[str],
    triples: list[Triple],
    train_pairs: list[tuple[str, str]],
    *,
    dimensions: int = 64,
    walk_length: int = 80,
    num_walks: int = 10,
    p: float = 1e-100,
    q: float = 1.0,
    window: int = 10,
    epochs: int = 5,
    workers: int = 1,
    max_degree: int = 1000,
    random_state: int | None = None,
) -> dict[str, npt.NDArray[np.floating[Any]]]:
    """One structural embedding per ``entity_ids``, via a biased random walk + skip-gram.

    ``p``/``q``/``walk_length``/``num_walks``/``window``/``epochs`` default
    to the reference's own values (`p=1e-100` effectively forbids
    immediately backtracking to the previous node; `q=1` is neutral
    BFS/DFS balance). Entities with no edges at all in `triples` fall back
    to a random vector -- ports `get_deep_emb.py`'s own ``except:
    np.random.random_sample(...)`` fallback for the same case.

    Args:
        max_degree: Caps every node's neighbor count before walking (see
            [cap_node_degree][linkingtk.algorithms.ea._simple_hhea_structure.cap_node_degree])
            -- needed for real event-KG-scale graphs (e.g. ICEWS-WIKI),
            whose degree distribution is extreme enough that node2vec's
            transition-probability precompute is otherwise intractable.
            Irrelevant at toy/unit-test scale.

    Raises:
        OptionalDependencyError: If `node2vec` isn't installed.
    """
    try:
        import networkx as nx
        from node2vec import Node2Vec
    except ImportError as exc:
        raise OptionalDependencyError("SimpleHHEALinker", "kge") from exc

    merge_map = _merge_map(train_pairs)

    def representative(entity_id: str) -> str:
        return merge_map.get(entity_id, entity_id)

    edges: set[tuple[str, str]] = set()
    for subject, _, obj in triples:
        u, v = representative(subject), representative(obj)
        if u != v:
            edges.add((u, v) if u < v else (v, u))
    # Sorted, not `list(edges)` directly -- a plain set's iteration order
    # depends on PYTHONHASHSEED (randomized per process by default), which
    # would silently make `cap_node_degree`'s rng.permutation select
    # different *edges* run to run even for the same `random_state`
    # (confirmed: this was flaking a fully-seeded unit test before being
    # sorted here).
    rng = np.random.default_rng(random_state)
    capped_edges = cap_node_degree(sorted(edges), max_degree, rng)

    graph = nx.Graph()
    graph.add_edges_from(capped_edges)

    embeddings: dict[str, npt.NDArray[np.floating[Any]]] = {}
    if graph.number_of_nodes() == 0:
        for entity_id in entity_ids:
            embeddings[entity_id] = rng.random(dimensions).astype(np.float32)
        return embeddings

    node2vec = Node2Vec(
        graph,
        dimensions=dimensions,
        walk_length=walk_length,
        num_walks=num_walks,
        p=p,
        q=q,
        workers=workers,
        quiet=True,
        seed=random_state,
    )
    model = node2vec.fit(sg=1, window=window, min_count=0, epochs=epochs)

    for entity_id in entity_ids:
        node = representative(entity_id)
        if node in model.wv:
            embeddings[entity_id] = np.asarray(model.wv[node], dtype=np.float32)
        else:
            embeddings[entity_id] = rng.random(dimensions).astype(np.float32)
    return embeddings

"""node2vec structural embeddings for
[SimpleHHEALinker][linkingtk.algorithms.ea.simple_hhea.SimpleHHEALinker].

Ports the biased-random-walk structural-embedding stage of
https://github.com/DataArcTech/Simple-HHEA
(`feature_perprocessing/preproccess.py` + `longterm/{main.py,node2vec.py}`
+ `get_deep_emb.py`) directly from its own hand-rolled alias-sampling walk
code in `longterm/node2vec.py` -- **not** the third-party `node2vec` PyPI
package this module first used.

That first attempt was a real, confirmed bug, not just a fidelity gap:
`longterm/node2vec.py`'s `get_alias_edge` uses a **non-standard** formula
(`weight * p` for the "return to the previous node" term -- the file
even has the textbook `weight / p` commented out directly above it, a
deliberate inversion of the Grover & Leskovec convention). The `node2vec`
PyPI package implements the *standard* (divide-by-`p`) formula, so
copying the reference's literal `p=1e-100` into it did the *opposite* of
the intended "never immediately backtrack" -- confirmed directly: sampled
walks degenerated into pure two-node oscillation (`['d','c','d','c',
...]`). Retuning `p` for the standard package's own convention (`p=100`)
produced genuinely exploratory walks, but scored *far* worse on the real
ICEWS-WIKI benchmark than the degenerate ones (Hits@1 0.037-0.041 vs.
0.262, across several configurations, including 10x more Word2Vec epochs
and a 10x harder degree cap -- ruling out undertraining and hub-node
dilution as the cause). The reference's actual formula isn't equivalent
to either end of the standard package's p/q dial: at `q=1`, its `elif`
branch (a neighbor that's *also* adjacent to the previous node) gets
zeroed out too, alongside the return term -- a third walk character the
standard package has no way to express. Hand-porting the exact formula
below, rather than continuing to search for an equivalent
parameterization, is the only way to reproduce it.

Ported near-verbatim: `_alias_setup`/`_alias_draw` (the classic
alias-method sampling utilities), `_precompute_transition_probs`/
`_get_alias_edge`/`_node2vec_walk`/`_simulate_walks` (the reference's
`Graph.preprocess_transition_probs`/`get_alias_edge`/`node2vec_walk`/
`simulate_walks`). Differences from the reference's own code, all
mechanical, none behavioral:

- `_alias_draw` takes an explicit `np.random.Generator` instead of
  reading numpy's global random state (this module already seeds
  everything else that way).
- `np.int` (removed in modern numpy) becomes `np.int64`.
- Each node's sorted neighbor list is computed once and cached
  (`_sorted_neighbors`), not recomputed by a fresh `sorted(G.neighbors(...))`
  call on every single walk step and every `get_alias_edge` call like the
  reference does -- pure performance (the ordering, and therefore every
  alias table's meaning, is identical either way; a real-graph hub node
  still at the `max_degree` cap would otherwise be re-sorted on every one
  of its many visits).

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
degree 18, but the single highest-degree node -- "United States", ICEWS
event data's most common actor -- has 273,317 edges; confirmed by direct
measurement, not assumed). The alias-table transition-probability
precompute is `O(degree)` per edge, so a hub node alone costs
`O(degree^2)` -- tens of billions of operations for that one node, which
never finished in practice with the original (uncapped) graph (confirmed:
killed after several CPU-bound minutes stuck in exactly that precompute
step, well before any walk was even simulated).
[cap_node_degree][linkingtk.algorithms.ea._simple_hhea_structure.cap_node_degree]
randomly subsamples any node's incident edges down to `max_degree` before
walking -- a standard, well-precedented mitigation for random-walk
embeddings on scale-free graphs (not something the reference needed to
consider, or that a toy-scale unit-test fixture would ever surface).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from linkingtk.exceptions import OptionalDependencyError
from linkingtk.utils.graph import Triple

if TYPE_CHECKING:
    import networkx as nx
    import numpy.typing as npt

AliasTable = tuple["npt.NDArray[np.int64]", "npt.NDArray[np.float64]"]


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


def _alias_setup(probs: list[float]) -> AliasTable:
    """Ports `alias_setup`: precomputes O(1)-draw tables for a discrete distribution."""
    size = len(probs)
    prob_table = np.zeros(size)
    alias_table = np.zeros(size, dtype=np.int64)

    smaller = []
    larger = []
    for index, prob in enumerate(probs):
        prob_table[index] = size * prob
        if prob_table[index] < 1.0:
            smaller.append(index)
        else:
            larger.append(index)

    while smaller and larger:
        small = smaller.pop()
        large = larger.pop()
        alias_table[small] = large
        prob_table[large] = prob_table[large] + prob_table[small] - 1.0
        if prob_table[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return alias_table, prob_table


def _alias_draw(table: AliasTable, rng: np.random.Generator) -> int:
    """Ports `alias_draw`: one O(1) draw from a table built by `_alias_setup`."""
    alias_table, prob_table = table
    index = int(rng.random() * len(alias_table))
    if rng.random() < prob_table[index]:
        return index
    return int(alias_table[index])


def _get_alias_edge(
    graph: nx.Graph,
    sorted_neighbors: dict[str, list[str]],
    src: str,
    dst: str,
    p: float,
    q: float,
) -> AliasTable:
    """Ports `get_alias_edge` -- the reference's own (non-standard) transition-weight formula.

    For a step arriving at `dst` from `src`, each of `dst`'s neighbors
    gets weight ``* p`` if it's `src` itself (return), ``* (1 - q)`` if
    it's also a neighbor of `src` (redundant, already-nearby territory),
    or ``* q`` otherwise (genuinely new territory).
    """
    unnormalized_probs = []
    for neighbor in sorted_neighbors[dst]:
        weight = graph[dst][neighbor].get("weight", 1.0)
        if neighbor == src:
            unnormalized_probs.append(weight * p)
        elif graph.has_edge(neighbor, src):
            unnormalized_probs.append(weight * (1 - q))
        else:
            unnormalized_probs.append(weight * q)
    norm_const = sum(unnormalized_probs)
    normalized_probs = [prob / norm_const for prob in unnormalized_probs]
    return _alias_setup(normalized_probs)


def _precompute_transition_probs(
    graph: nx.Graph, sorted_neighbors: dict[str, list[str]], p: float, q: float
) -> tuple[dict[str, AliasTable], dict[tuple[str, str], AliasTable]]:
    """Ports `preprocess_transition_probs`."""
    alias_nodes: dict[str, AliasTable] = {}
    for node in graph.nodes():
        unnormalized_probs = [
            graph[node][neighbor].get("weight", 1.0) for neighbor in sorted_neighbors[node]
        ]
        norm_const = sum(unnormalized_probs)
        normalized_probs = [prob / norm_const for prob in unnormalized_probs]
        alias_nodes[node] = _alias_setup(normalized_probs)

    alias_edges: dict[tuple[str, str], AliasTable] = {}
    for src, dst in graph.edges():
        alias_edges[src, dst] = _get_alias_edge(graph, sorted_neighbors, src, dst, p, q)
        alias_edges[dst, src] = _get_alias_edge(graph, sorted_neighbors, dst, src, p, q)

    return alias_nodes, alias_edges


def _node2vec_walk(
    sorted_neighbors: dict[str, list[str]],
    alias_nodes: dict[str, AliasTable],
    alias_edges: dict[tuple[str, str], AliasTable],
    walk_length: int,
    start_node: str,
    rng: np.random.Generator,
) -> list[str]:
    """Ports `node2vec_walk`."""
    walk = [start_node]
    while len(walk) < walk_length:
        current = walk[-1]
        neighbors = sorted_neighbors[current]
        if not neighbors:
            break
        if len(walk) == 1:
            walk.append(neighbors[_alias_draw(alias_nodes[current], rng)])
        else:
            previous = walk[-2]
            walk.append(neighbors[_alias_draw(alias_edges[previous, current], rng)])
    return walk


def _simulate_walks(
    graph: nx.Graph,
    sorted_neighbors: dict[str, list[str]],
    alias_nodes: dict[str, AliasTable],
    alias_edges: dict[tuple[str, str], AliasTable],
    num_walks: int,
    walk_length: int,
    rng: np.random.Generator,
) -> list[list[str]]:
    """Ports `simulate_walks`."""
    walks = []
    nodes = list(graph.nodes())
    for _ in range(num_walks):
        rng.shuffle(nodes)
        for node in nodes:
            walks.append(
                _node2vec_walk(sorted_neighbors, alias_nodes, alias_edges, walk_length, node, rng)
            )
    return walks


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
    to the reference's own values -- see this module's docstring for why
    `p=1e-100` (a tiny, near-zero value) is correct *here*, ported from
    the reference's own non-standard alias-edge formula (`weight * p`,
    not `weight / p`), unlike the standard `node2vec` package's
    convention this module used at first. Entities with no edges at all
    in `triples` fall back to a random vector -- ports `get_deep_emb.py`'s
    own ``except: np.random.random_sample(...)`` fallback for the same
    case.

    Args:
        workers: Worker threads for gensim's `Word2Vec` fit (real
            in-process threading, not separate OS processes -- safe to
            raise). Walk simulation itself is always single-threaded (the
            reference's own `simulate_walks` is too, absent an explicit
            `--workers` override it doesn't use for `longterm/main.py`'s
            default single-process invocation).
        max_degree: Caps every node's neighbor count before walking (see
            [cap_node_degree][linkingtk.algorithms.ea._simple_hhea_structure.cap_node_degree])
            -- needed for real event-KG-scale graphs (e.g. ICEWS-WIKI),
            whose degree distribution is extreme enough that the
            transition-probability precompute is otherwise intractable.
            Irrelevant at toy/unit-test scale.

    Raises:
        OptionalDependencyError: If `networkx`/`gensim` aren't installed.
    """
    try:
        import networkx as nx
        from gensim.models import Word2Vec
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
    graph.add_edges_from(capped_edges, weight=1.0)

    embeddings: dict[str, npt.NDArray[np.floating[Any]]] = {}
    if graph.number_of_nodes() == 0:
        for entity_id in entity_ids:
            embeddings[entity_id] = rng.random(dimensions).astype(np.float32)
        return embeddings

    sorted_neighbors = {node: sorted(graph.neighbors(node)) for node in graph.nodes()}
    alias_nodes, alias_edges = _precompute_transition_probs(graph, sorted_neighbors, p, q)
    walks = _simulate_walks(
        graph, sorted_neighbors, alias_nodes, alias_edges, num_walks, walk_length, rng
    )

    model = Word2Vec(
        walks,
        vector_size=dimensions,
        window=window,
        min_count=0,
        sg=1,
        epochs=epochs,
        workers=workers,
        seed=random_state if random_state is not None else 1,
    )

    for entity_id in entity_ids:
        node = representative(entity_id)
        if node in model.wv:
            embeddings[entity_id] = np.asarray(model.wv[node], dtype=np.float32)
        else:
            embeddings[entity_id] = rng.random(dimensions).astype(np.float32)
    return embeddings

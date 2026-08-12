"""Graph utilities wrapping optional NetworkX and RDFLib dependencies, plus
pykeen-``TriplesFactory``-compatible id mapping and train/test splitting
that work without pykeen installed.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Union

import numpy as np
from sklearn.model_selection import train_test_split as _sk_train_test_split

if TYPE_CHECKING:
    import networkx as nx
    import numpy.typing as npt
    import rdflib

Triple = tuple[str, str, str]
Graph = Union[list[Triple], "nx.Graph", "rdflib.Graph", None]


def to_triples(graph: Graph, *, relation_attr: str = "relation") -> list[Triple]:
    """Normalize a supported graph representation into a plain triple list.

    Args:
        graph: A list of ``(subject, predicate, object)`` triples, a
            ``networkx.Graph`` (or ``DiGraph``/``MultiGraph``/``MultiDiGraph``),
            an ``rdflib.Graph``, or ``None``.
        relation_attr: For ``networkx`` graphs only: the edge-data key to
            read each triple's relation/predicate label from (e.g. set via
            ``G.add_edge(u, v, relation="author_of")``). When an edge lacks
            this attribute, the triple falls back to the literal relation
            ``"related_to"``. Ignored for ``rdflib.Graph`` and plain-list
            inputs, which already carry real predicates.

    Returns:
        The graph as a list of triples. Empty if ``graph`` is ``None``.

    Raises:
        TypeError: If ``graph`` is not a triple list, ``None``, or an
            instance of an installed NetworkX/RDFLib graph type.
    """
    if graph is None:
        return []
    if isinstance(graph, list):
        return graph

    try:
        import rdflib

        if isinstance(graph, rdflib.Graph):
            return [(str(s), str(p), str(o)) for s, p, o in graph]
    except ImportError:
        pass

    try:
        import networkx as nx

        if isinstance(graph, nx.Graph):
            return [
                (str(u), str(data.get(relation_attr, "related_to")), str(v))
                for u, v, data in graph.edges(data=True)
            ]
    except ImportError:
        pass

    raise TypeError(f"Unsupported graph type: {type(graph)!r}")


def build_id_mappings(triples: list[Triple]) -> tuple[dict[str, int], dict[str, int]]:
    """Build pykeen-``TriplesFactory``-compatible entity/relation id mappings.

    Mirrors what ``pykeen.triples.TriplesFactory.from_labeled_triples`` builds
    internally when no explicit mapping is given: entities are the sorted
    union of all subjects and objects, relations are the sorted set of
    predicates, each enumerated from ``0``. Sorting makes the mapping
    deterministic regardless of triple order/duplication.

    Build this from the *complete* triple set (e.g. before splitting via
    [train_test_split_triples][linkingtk.utils.graph.train_test_split_triples])
    so that a train split and a test split can share one consistent id
    space -- see that function's docstring.

    Args:
        triples: Triples to derive ids from, e.g. from
            [to_triples][linkingtk.utils.graph.to_triples].

    Returns:
        ``(entity_to_id, relation_to_id)``. Both are ``{}`` if ``triples``
        is empty.
    """
    entities = sorted({s for s, _, _ in triples} | {o for _, _, o in triples})
    relations = sorted({p for _, p, _ in triples})
    return (
        {label: index for index, label in enumerate(entities)},
        {label: index for index, label in enumerate(relations)},
    )


def map_triples_to_ids(
    triples: list[Triple],
    entity_to_id: Mapping[str, int],
    relation_to_id: Mapping[str, int],
) -> npt.NDArray[np.int64]:
    """Map label triples to a pykeen-``TriplesFactory``-ready id array.

    Args:
        triples: Triples to map.
        entity_to_id: Subject/object label -> id, e.g. from
            [build_id_mappings][linkingtk.utils.graph.build_id_mappings].
        relation_to_id: Predicate label -> id.

    Returns:
        An ``(n, 3)`` ``int64`` array of ``(head_id, relation_id, tail_id)``
        rows, in the same column order pykeen's ``TriplesFactory.mapped_triples``
        uses -- pass it straight through as that constructor's
        ``mapped_triples`` argument. Shape is ``(0, 3)`` if ``triples`` is
        empty.

    Raises:
        KeyError: If a triple references a label absent from
            ``entity_to_id``/``relation_to_id``.
    """
    if not triples:
        return np.empty((0, 3), dtype=np.int64)
    return np.array(
        [(entity_to_id[s], relation_to_id[p], entity_to_id[o]) for s, p, o in triples],
        dtype=np.int64,
    )


def train_test_split_triples(
    triples: list[Triple],
    test_size: float = 0.2,
    random_state: int | None = None,
) -> tuple[list[Triple], list[Triple]]:
    """Randomly split triples into train/test lists.

    A thin wrapper around ``sklearn.model_selection.train_test_split`` (no
    pykeen dependency, so this works without the ``kge`` extra installed).
    Unlike ``pykeen.triples.TriplesFactory.split``, this does **not**
    guarantee every entity/relation appears in the train split -- it's a
    plain uniform-random row split over ``triples``, so an entity that
    happens to have all its triples land in the test partition will be
    absent from train (and vice versa). Callers that need that coverage
    guarantee should build a ``TriplesFactory`` and call its own
    ``.split()`` instead.

    Build id mappings (via
    [build_id_mappings][linkingtk.utils.graph.build_id_mappings]) from the
    *full* ``triples`` list passed here, not from the returned splits
    independently, so both splits share one consistent id space -- e.g.::

        train, test = train_test_split_triples(triples, test_size=0.2, random_state=0)
        entity_to_id, relation_to_id = build_id_mappings(triples)
        train_ids = map_triples_to_ids(train, entity_to_id, relation_to_id)
        test_ids = map_triples_to_ids(test, entity_to_id, relation_to_id)

    Self-loops (``subject == object``) and multi-edges/duplicate triples
    (e.g. from a ``networkx.MultiDiGraph`` via
    [to_triples][linkingtk.utils.graph.to_triples]) are treated as
    ordinary, independent rows -- no deduplication or entity/relation-aware
    grouping is performed, so two triples that share an entity pair can
    land on opposite sides of the split.

    Args:
        triples: Triples to split.
        test_size: Fraction of ``triples`` assigned to the test split
            (sklearn's own ``test_size`` semantics).
        random_state: Seed for reproducible splitting; ``None`` splits
            non-deterministically (sklearn's default).

    Returns:
        ``(train_triples, test_triples)``. ``([], [])`` if ``triples`` is
        empty.

    Raises:
        ValueError: Propagated from sklearn if ``test_size``/the number of
            triples can't produce a non-empty train and test split (e.g.
            fewer than 2 triples).
    """
    if not triples:
        return [], []
    train, test = _sk_train_test_split(triples, test_size=test_size, random_state=random_state)
    return train, test

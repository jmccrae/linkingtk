"""Graph utilities wrapping optional NetworkX and RDFLib dependencies."""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    import networkx as nx
    import rdflib

Triple = tuple[str, str, str]
Graph = Union[list[Triple], "nx.Graph", "rdflib.Graph", None]


def to_triples(graph: Graph) -> list[Triple]:
    """Normalize a supported graph representation into a plain triple list.

    Args:
        graph: A list of ``(subject, predicate, object)`` triples, a
            ``networkx.Graph``, an ``rdflib.Graph``, or ``None``.

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
            return [(str(u), "related_to", str(v)) for u, v in graph.edges()]
    except ImportError:
        pass

    raise TypeError(f"Unsupported graph type: {type(graph)!r}")

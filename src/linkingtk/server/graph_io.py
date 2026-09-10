"""Parse an uploaded supporting graph into a `linkingtk.utils.graph.Graph`.

Supports raw inline triples, any RDF serialization `rdflib` can auto-detect
(turtle, N-Triples, RDF/XML, JSON-LD, ...), and NetworkX's GraphML, GEXF,
and edge-list file formats. The result is handed to
[BaseLinker.link][linkingtk.algorithms.base.BaseLinker.link] as-is --
[to_triples][linkingtk.utils.graph.to_triples] normalizes it from there, so
this module does no triple conversion of its own.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from linkingtk.exceptions import OptionalDependencyError
from linkingtk.utils.graph import Graph

_NETWORKX_FORMATS = {"graphml", "gexf", "edgelist"}
SUPPORTED_FORMATS = {"triples", "rdf", *_NETWORKX_FORMATS}


def parse_graph(data: bytes, fmt: str) -> Graph:
    """Parse uploaded graph bytes into a `linkingtk.utils.graph.Graph`.

    Args:
        data: The raw uploaded file/body content.
        fmt: One of ``"triples"`` (inline JSON list of ``[s, p, o]``
            triples), ``"rdf"`` (any rdflib-parseable serialization), or
            ``"graphml"``/``"gexf"``/``"edgelist"``.

    Raises:
        ValueError: If ``fmt`` isn't one of the supported values, or the
            content can't be parsed as that format.
    """
    if fmt == "triples":
        try:
            triples = json.loads(data)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON for graph_format='triples': {exc}") from exc
        return [(s, p, o) for s, p, o in triples]

    if fmt == "rdf":
        try:
            import rdflib
        except ImportError as exc:
            raise OptionalDependencyError("graph_format='rdf'", "graph") from exc

        graph = rdflib.Graph()
        try:
            graph.parse(data=data.decode("utf-8"))
        except Exception as exc:  # rdflib raises many different parser-specific errors
            raise ValueError(f"Could not parse RDF graph: {exc}") from exc
        return graph

    if fmt in _NETWORKX_FORMATS:
        try:
            import networkx as nx
        except ImportError as exc:
            raise OptionalDependencyError(f"graph_format={fmt!r}", "graph") from exc

        reader = {
            "graphml": nx.read_graphml,
            "gexf": nx.read_gexf,
            "edgelist": nx.read_edgelist,
        }[fmt]
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / f"graph.{fmt}"
            tmp_path.write_bytes(data)
            try:
                return reader(tmp_path)
            except Exception as exc:  # networkx raises many different parser-specific errors
                raise ValueError(f"Could not parse {fmt} graph: {exc}") from exc

    raise ValueError(
        f"Unsupported graph_format {fmt!r}; expected one of {sorted(SUPPORTED_FORMATS)}"
    )

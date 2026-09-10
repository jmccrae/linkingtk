import json

import pytest
from fastapi.testclient import TestClient

from linkingtk.algorithms.base import BaseLinker
from linkingtk.core.entity import Entity
from linkingtk.core.result import AlignmentResult
from linkingtk.core.source import EntitySource
from linkingtk.server import create_app


class _FakeLinker(BaseLinker):
    """Records what it's called with and returns a fixed result."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def link(self, dataset1, dataset2, graph=None, blocking=None):
        self.calls.append((dataset1, dataset2, graph))
        return [AlignmentResult(source_id="a1", target_id="b1", score=0.9)]


class _FakeSource(EntitySource):
    def __init__(self) -> None:
        self._entities = [Entity(id="b1", labels=["cat"])]

    def search(self, query: str, top_k: int = 10) -> list[Entity]:
        return self._entities[:top_k]

    def get(self, entity_id: str) -> Entity | None:
        return next((e for e in self._entities if e.id == entity_id), None)


@pytest.fixture
def linker() -> _FakeLinker:
    return _FakeLinker()


@pytest.fixture
def source() -> _FakeSource:
    return _FakeSource()


@pytest.fixture
def client(linker: _FakeLinker, source: _FakeSource) -> TestClient:
    app = create_app(linkers={"fake": linker}, sources={"fake-source": source})
    return TestClient(app)


def _dataset1_body(**extra) -> dict:
    body = {"dataset1": [{"id": "a1", "labels": ["cat"]}]}
    body.update(extra)
    return body


def test_health(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_list_linkers(client: TestClient) -> None:
    assert client.get("/linkers").json() == ["fake"]


def test_list_sources(client: TestClient) -> None:
    assert client.get("/sources").json() == ["fake-source"]


def test_link_with_inline_entities(client: TestClient, linker: _FakeLinker) -> None:
    body = _dataset1_body(dataset2_entities=[{"id": "b1", "labels": ["dog"]}])
    response = client.post("/link/fake", data={"request": json.dumps(body)})

    assert response.status_code == 200
    assert response.json() == [
        {"source_id": "a1", "target_id": "b1", "score": 0.9, "alternatives": []}
    ]
    dataset1, dataset2, graph = linker.calls[0]
    assert dataset1 == [Entity(id="a1", labels=["cat"])]
    assert dataset2 == [Entity(id="b1", labels=["dog"])]
    assert graph is None


def test_link_with_named_source(
    client: TestClient, linker: _FakeLinker, source: _FakeSource
) -> None:
    body = _dataset1_body(dataset2_source="fake-source")
    response = client.post("/link/fake", data={"request": json.dumps(body)})

    assert response.status_code == 200
    _, dataset2, _ = linker.calls[0]
    assert dataset2 is source


def test_link_with_inline_triples_graph(client: TestClient, linker: _FakeLinker) -> None:
    body = _dataset1_body(dataset2_entities=[])
    triples = json.dumps([["s", "p", "o"]]).encode()

    response = client.post(
        "/link/fake",
        data={"request": json.dumps(body), "graph_format": "triples"},
        files={"graph": ("graph.json", triples, "application/json")},
    )

    assert response.status_code == 200
    _, _, graph = linker.calls[0]
    assert graph == [("s", "p", "o")]


def test_link_with_rdf_graph(client: TestClient, linker: _FakeLinker) -> None:
    body = _dataset1_body(dataset2_entities=[])
    turtle = b"""
    @prefix ex: <http://example.org/> .
    ex:s ex:p ex:o .
    """

    response = client.post(
        "/link/fake",
        data={"request": json.dumps(body), "graph_format": "rdf"},
        files={"graph": ("graph.ttl", turtle, "text/turtle")},
    )

    assert response.status_code == 200
    _, _, graph = linker.calls[0]
    from linkingtk.utils.graph import to_triples

    assert to_triples(graph) == [
        ("http://example.org/s", "http://example.org/p", "http://example.org/o")
    ]


def test_link_with_graphml_graph(client: TestClient, linker: _FakeLinker) -> None:
    body = _dataset1_body(dataset2_entities=[])
    graphml = b"""<?xml version="1.0" encoding="UTF-8"?>
    <graphml xmlns="http://graphml.graphdrawing.org/xmlns">
      <graph edgedefault="directed">
        <node id="s"/>
        <node id="o"/>
        <edge source="s" target="o"/>
      </graph>
    </graphml>
    """

    response = client.post(
        "/link/fake",
        data={"request": json.dumps(body), "graph_format": "graphml"},
        files={"graph": ("graph.graphml", graphml, "application/xml")},
    )

    assert response.status_code == 200
    _, _, graph = linker.calls[0]
    from linkingtk.utils.graph import to_triples

    assert to_triples(graph) == [("s", "related_to", "o")]


def test_unknown_linker_returns_404(client: TestClient) -> None:
    body = _dataset1_body(dataset2_entities=[])
    response = client.post("/link/nope", data={"request": json.dumps(body)})
    assert response.status_code == 404


def test_unknown_source_returns_404(client: TestClient) -> None:
    body = _dataset1_body(dataset2_source="nope")
    response = client.post("/link/fake", data={"request": json.dumps(body)})
    assert response.status_code == 404


def test_bad_graph_format_returns_400(client: TestClient) -> None:
    body = _dataset1_body(dataset2_entities=[])
    response = client.post(
        "/link/fake",
        data={"request": json.dumps(body), "graph_format": "bogus"},
        files={"graph": ("graph.bin", b"data", "application/octet-stream")},
    )
    assert response.status_code == 400

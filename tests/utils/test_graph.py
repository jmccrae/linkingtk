import networkx as nx
import numpy as np
import pytest
import rdflib

from linkingtk.utils.graph import (
    build_id_mappings,
    map_triples_to_ids,
    to_triples,
    train_test_split_triples,
)


def test_to_triples_none() -> None:
    assert to_triples(None) == []


def test_to_triples_list_passthrough() -> None:
    triples = [("a", "rel", "b")]
    assert to_triples(triples) == triples


def test_to_triples_rejects_unsupported_type() -> None:
    with pytest.raises(TypeError):
        to_triples(object())  # type: ignore[arg-type]


class TestToTriplesNetworkX:
    def test_digraph_default_relation_fallback(self) -> None:
        graph = nx.DiGraph()
        graph.add_edge("a", "b")
        assert to_triples(graph) == [("a", "related_to", "b")]

    def test_relation_attr_used_when_present(self) -> None:
        graph = nx.DiGraph()
        graph.add_edge("a", "b", relation="author_of")
        graph.add_edge("b", "c")
        assert to_triples(graph) == [
            ("a", "author_of", "b"),
            ("b", "related_to", "c"),
        ]

    def test_custom_relation_attr_name(self) -> None:
        graph = nx.DiGraph()
        graph.add_edge("a", "b", label="knows")
        assert to_triples(graph, relation_attr="label") == [("a", "knows", "b")]

    def test_self_loop_preserved(self) -> None:
        graph = nx.DiGraph()
        graph.add_edge("a", "a", relation="self_ref")
        assert to_triples(graph) == [("a", "self_ref", "a")]

    def test_multidigraph_parallel_edges_preserved(self) -> None:
        graph = nx.MultiDiGraph()
        graph.add_edge("a", "b", relation="author_of")
        graph.add_edge("a", "b", relation="influenced_by")
        assert sorted(to_triples(graph)) == sorted(
            [("a", "author_of", "b"), ("a", "influenced_by", "b")]
        )


class TestToTriplesRDFLib:
    def test_rdflib_graph(self) -> None:
        graph = rdflib.Graph()
        s, p, o = rdflib.URIRef("urn:a"), rdflib.URIRef("urn:rel"), rdflib.URIRef("urn:b")
        graph.add((s, p, o))
        assert to_triples(graph) == [(str(s), str(p), str(o))]


class TestBuildIdMappings:
    def test_empty(self) -> None:
        assert build_id_mappings([]) == ({}, {})

    def test_sorted_deterministic_mapping(self) -> None:
        triples = [("b", "r2", "c"), ("a", "r1", "b")]
        entity_to_id, relation_to_id = build_id_mappings(triples)
        assert entity_to_id == {"a": 0, "b": 1, "c": 2}
        assert relation_to_id == {"r1": 0, "r2": 1}

    def test_self_loop_entity_counted_once(self) -> None:
        entity_to_id, _ = build_id_mappings([("a", "r", "a")])
        assert entity_to_id == {"a": 0}


class TestMapTriplesToIds:
    def test_empty(self) -> None:
        result = map_triples_to_ids([], {}, {})
        assert result.shape == (0, 3)
        assert result.dtype == np.int64

    def test_basic_mapping(self) -> None:
        triples = [("a", "r1", "b"), ("b", "r2", "a")]
        entity_to_id, relation_to_id = build_id_mappings(triples)
        result = map_triples_to_ids(triples, entity_to_id, relation_to_id)
        assert result.tolist() == [[0, 0, 1], [1, 1, 0]]

    def test_unknown_label_raises_keyerror(self) -> None:
        with pytest.raises(KeyError):
            map_triples_to_ids([("a", "r", "b")], {}, {})


class TestTrainTestSplitTriples:
    _TRIPLES = [(f"e{i}", "r", f"e{i + 1}") for i in range(10)]

    def test_empty(self) -> None:
        assert train_test_split_triples([]) == ([], [])

    def test_split_is_a_disjoint_full_partition(self) -> None:
        train, test = train_test_split_triples(self._TRIPLES, test_size=0.3, random_state=0)
        assert len(train) + len(test) == len(self._TRIPLES)
        assert set(train).isdisjoint(test)
        assert set(train) | set(test) == set(self._TRIPLES)
        assert len(test) == 3

    def test_deterministic_with_random_state(self) -> None:
        first = train_test_split_triples(self._TRIPLES, test_size=0.3, random_state=0)
        second = train_test_split_triples(self._TRIPLES, test_size=0.3, random_state=0)
        assert first == second

    def test_too_few_triples_raises(self) -> None:
        with pytest.raises(ValueError):
            train_test_split_triples([("a", "r", "b")])

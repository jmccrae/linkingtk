import pytest

from linkingtk.utils.graph import to_triples


def test_to_triples_none() -> None:
    assert to_triples(None) == []


def test_to_triples_list_passthrough() -> None:
    triples = [("a", "rel", "b")]
    assert to_triples(triples) == triples


def test_to_triples_rejects_unsupported_type() -> None:
    with pytest.raises(TypeError):
        to_triples(object())  # type: ignore[arg-type]

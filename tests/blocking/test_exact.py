from linkingtk.blocking.exact import ExactMatch
from linkingtk.core.entity import Entity


def test_exact_match_finds_shared_label() -> None:
    dataset1 = [Entity(id="a1", labels=["cat"]), Entity(id="a2", labels=["dog"])]
    dataset2 = [Entity(id="b1", labels=[("cat", "en")]), Entity(id="b2", labels=["fish"])]

    pairs = ExactMatch().candidate_pairs(dataset1, dataset2)

    assert [(e1.id, e2.id) for e1, e2 in pairs] == [("a1", "b1")]


def test_exact_match_no_candidates() -> None:
    dataset1 = [Entity(id="a1", labels=["cat"])]
    dataset2 = [Entity(id="b1", labels=["dog"])]

    assert ExactMatch().candidate_pairs(dataset1, dataset2) == []


def test_exact_match_deduplicates_pairs() -> None:
    dataset1 = [Entity(id="a1", labels=["cat", "kitty"])]
    dataset2 = [Entity(id="b1", labels=["cat", "kitty"])]

    pairs = ExactMatch().candidate_pairs(dataset1, dataset2)

    assert len(pairs) == 1

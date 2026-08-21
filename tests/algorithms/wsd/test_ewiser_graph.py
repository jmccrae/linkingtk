from __future__ import annotations

import sys
import types

import pytest

torch = pytest.importorskip("torch")

from linkingtk.algorithms.wsd._ewiser_graph import build_relation_adjacency  # noqa: E402
from linkingtk.algorithms.wsd._ewiser_vocab import SenseVocabulary  # noqa: E402


class _FakeWnError(Exception):
    pass


class _FakeSense:
    def __init__(self, synset: _FakeSynset, derivation: list[_FakeSense] | None = None) -> None:
        self._synset = synset
        self._derivation = derivation or []

    def synset(self) -> _FakeSynset:
        return self._synset

    def relations(self, relation_type: str) -> dict[str, list[_FakeSense]]:
        if relation_type == "derivation" and self._derivation:
            return {"derivation": self._derivation}
        return {}


class _FakeSynset:
    def __init__(
        self,
        id: str,
        hypernym: list[_FakeSynset] | None = None,
        similar: list[_FakeSynset] | None = None,
        verb_group: list[_FakeSynset] | None = None,
        senses: list[_FakeSense] | None = None,
    ) -> None:
        self.id = id
        self._by_type = {
            "hypernym": hypernym or [],
            "similar": similar or [],
            "verb_group": verb_group or [],
        }
        self._senses = senses or []

    def relations(self, *types: str) -> dict[str, list[_FakeSynset]]:
        result = {}
        for relation_type in types:
            values = self._by_type.get(relation_type, [])
            if values:
                result[relation_type] = values
        return result

    def senses(self) -> list[_FakeSense]:
        return self._senses


def _fake_wn_module(synsets_by_id: dict[str, _FakeSynset]) -> types.ModuleType:
    module = types.ModuleType("wn")

    def synset(id: str, *, lexicon: str | None = None) -> _FakeSynset:
        found = synsets_by_id.get(id)
        if found is None:
            raise module.Error(f"no such synset: {id}")  # type: ignore[attr-defined]
        return found

    module.synset = synset  # type: ignore[attr-defined]
    module.Error = _FakeWnError  # type: ignore[attr-defined]
    return module


def _dense(adjacency: torch.Tensor) -> torch.Tensor:
    return adjacency.to_dense()


class TestHypernymDirection:
    def test_propagates_from_hypernym_into_hyponym_only(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        hypernym = _FakeSynset("s-hyper")
        hyponym = _FakeSynset("s-hypo", hypernym=[hypernym])
        module = _fake_wn_module({"s-hyper": hypernym, "s-hypo": hyponym})
        monkeypatch.setitem(sys.modules, "wn", module)
        vocabulary = SenseVocabulary(["s-hypo", "s-hyper"])

        adjacency = _dense(build_relation_adjacency(vocabulary))

        hypo_idx, hyper_idx = 0, 1
        assert adjacency[hypo_idx, hyper_idx] > 0
        assert adjacency[hyper_idx, hypo_idx] == 0


class TestSymmetricRelations:
    def test_similar_is_bidirectional(self, monkeypatch: pytest.MonkeyPatch) -> None:
        a = _FakeSynset("s-a")
        b = _FakeSynset("s-b")
        a._by_type["similar"] = [b]
        b._by_type["similar"] = [a]
        module = _fake_wn_module({"s-a": a, "s-b": b})
        monkeypatch.setitem(sys.modules, "wn", module)
        vocabulary = SenseVocabulary(["s-a", "s-b"])

        adjacency = _dense(build_relation_adjacency(vocabulary))

        assert adjacency[0, 1] > 0
        assert adjacency[1, 0] > 0

    def test_verb_group_is_bidirectional(self, monkeypatch: pytest.MonkeyPatch) -> None:
        a = _FakeSynset("s-a")
        b = _FakeSynset("s-b")
        a._by_type["verb_group"] = [b]
        b._by_type["verb_group"] = [a]
        module = _fake_wn_module({"s-a": a, "s-b": b})
        monkeypatch.setitem(sys.modules, "wn", module)
        vocabulary = SenseVocabulary(["s-a", "s-b"])

        adjacency = _dense(build_relation_adjacency(vocabulary))

        assert adjacency[0, 1] > 0
        assert adjacency[1, 0] > 0

    def test_derivation_roundtrips_through_senses_and_is_bidirectional(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # wn stores derivation reciprocally (both senses list each other) --
        # confirmed directly against a real lexicon -- so the fixture
        # declares both directions explicitly, the way real data would.
        a = _FakeSynset("s-a")
        b = _FakeSynset("s-b")
        sense_a = _FakeSense(a)
        sense_b = _FakeSense(b)
        sense_a._derivation = [sense_b]
        sense_b._derivation = [sense_a]
        a._senses = [sense_a]
        b._senses = [sense_b]
        module = _fake_wn_module({"s-a": a, "s-b": b})
        monkeypatch.setitem(sys.modules, "wn", module)
        vocabulary = SenseVocabulary(["s-a", "s-b"])

        adjacency = _dense(build_relation_adjacency(vocabulary))

        assert adjacency[0, 1] > 0
        assert adjacency[1, 0] > 0


class TestTopKTruncation:
    def test_normalizes_over_all_predecessors_then_truncates(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # child has 4 hypernym predecessors; p1 gets a second (similar) edge
        # so its raw incoming weight is 2 vs. 1 for p2/p3/p4. Normalized
        # over all 4 (total=5): p1=0.4, p2=p3=p4=0.2 each. With
        # max_incoming=2, only p1 (0.4) and the first-inserted 0.2 tie (p2)
        # should survive -- and the surviving weights should NOT be
        # renormalized to sum to 1 (0.4 + 0.2 = 0.6, not 1.0).
        p1, p2, p3, p4 = (_FakeSynset(f"s-p{i}") for i in range(1, 5))
        child = _FakeSynset("s-child", hypernym=[p1, p2, p3, p4], similar=[p1])
        p1._by_type["similar"] = [child]
        module = _fake_wn_module({"s-child": child, "s-p1": p1, "s-p2": p2, "s-p3": p3, "s-p4": p4})
        monkeypatch.setitem(sys.modules, "wn", module)
        vocabulary = SenseVocabulary(["s-child", "s-p1", "s-p2", "s-p3", "s-p4"])

        adjacency = _dense(build_relation_adjacency(vocabulary, max_incoming=2))

        child_row = adjacency[0]
        assert child_row[1].item() == pytest.approx(0.4)
        assert child_row[2].item() == pytest.approx(0.2)
        assert child_row[3].item() == 0.0
        assert child_row[4].item() == 0.0
        assert child_row.sum().item() == pytest.approx(0.6)


class TestUnresolvedTargets:
    def test_relation_target_outside_vocabulary_is_dropped_not_an_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        outside = _FakeSynset("s-outside")
        hyponym = _FakeSynset("s-hypo", hypernym=[outside])
        module = _fake_wn_module({"s-hypo": hyponym, "s-outside": outside})
        monkeypatch.setitem(sys.modules, "wn", module)
        vocabulary = SenseVocabulary(["s-hypo"])

        adjacency = _dense(build_relation_adjacency(vocabulary))

        assert adjacency.sum().item() == 0.0

    def test_reserved_slot_is_skipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        module = _fake_wn_module({})
        monkeypatch.setitem(sys.modules, "wn", module)
        vocabulary = SenseVocabulary([None, "s-a"])

        adjacency = _dense(build_relation_adjacency(vocabulary))

        assert adjacency.shape == (2, 2)
        assert adjacency.sum().item() == 0.0

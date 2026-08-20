from __future__ import annotations

import sys
import types

import pytest

from linkingtk.sources.wn import WnEntitySource


class _FakeWnError(Exception):
    pass


class _FakeLexicon:
    def __init__(self, language: str) -> None:
        self.language = language


class _FakeWord:
    def __init__(self, lemma: str, language: str) -> None:
        self._lemma = lemma
        self._language = language

    def lemma(self) -> str:
        return self._lemma

    def lexicon(self) -> _FakeLexicon:
        return _FakeLexicon(self._language)


class _FakeSynset:
    def __init__(self, id: str, words: list[_FakeWord], definition: str | None) -> None:
        self.id = id
        self._words = words
        self._definition = definition

    def words(self) -> list[_FakeWord]:
        return self._words

    def lemmas(self) -> list[str]:
        return [word.lemma() for word in self._words]

    def definition(self) -> str | None:
        return self._definition


def _fake_wn_module(
    synsets_by_query: dict[str, list[_FakeSynset]],
    synsets_by_id: dict[str, _FakeSynset],
) -> types.ModuleType:
    module = types.ModuleType("wn")

    def synsets(
        form: str | None = None,
        pos: str | None = None,
        ili: str | None = None,
        *,
        lexicon: str | None = None,
        lang: str | None = None,
    ) -> list[_FakeSynset]:
        return synsets_by_query.get(form or "", [])

    def synset(id: str, *, lexicon: str | None = None, lang: str | None = None) -> _FakeSynset:
        found = synsets_by_id.get(id)
        if found is None:
            raise module.Error(f"no such synset: {id}")  # type: ignore[attr-defined]
        return found

    module.synsets = synsets  # type: ignore[attr-defined]
    module.synset = synset  # type: ignore[attr-defined]
    module.Error = _FakeWnError  # type: ignore[attr-defined]
    return module


_BANK_FINANCE = _FakeSynset(
    id="oewn-bank.n.01",
    words=[_FakeWord("bank", "en")],
    definition="a financial institution",
)
_BANK_RIVER = _FakeSynset(
    id="oewn-bank.n.02",
    words=[_FakeWord("bank", "en"), _FakeWord("riverbank", "en")],
    definition="the land alongside a river",
)


@pytest.fixture
def fake_wn(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    module = _fake_wn_module(
        synsets_by_query={"bank": [_BANK_FINANCE, _BANK_RIVER]},
        synsets_by_id={"oewn-bank.n.01": _BANK_FINANCE, "oewn-bank.n.02": _BANK_RIVER},
    )
    monkeypatch.setitem(sys.modules, "wn", module)
    return module


class TestSearch:
    def test_maps_synsets_to_entities(self, fake_wn: types.ModuleType) -> None:
        source = WnEntitySource(lang="en")

        results = source.search("bank")

        assert [e.id for e in results] == ["oewn-bank.n.01", "oewn-bank.n.02"]
        assert results[0].labels == ["bank"]
        assert results[0].description == "a financial institution"

    def test_no_matches_returns_empty_list(self, fake_wn: types.ModuleType) -> None:
        source = WnEntitySource(lang="en")

        assert source.search("nonexistent") == []

    def test_top_k_caps_results(self, fake_wn: types.ModuleType) -> None:
        source = WnEntitySource(lang="en")

        results = source.search("bank", top_k=1)

        assert len(results) == 1

    def test_multilingual_labels_get_language_tags(self, fake_wn: types.ModuleType) -> None:
        source = WnEntitySource()  # lang=None

        results = source.search("bank")

        assert results[1].labels == [("bank", "en"), ("riverbank", "en")]


class TestGet:
    def test_returns_matching_entity(self, fake_wn: types.ModuleType) -> None:
        source = WnEntitySource(lang="en")

        entity = source.get("oewn-bank.n.01")

        assert entity is not None
        assert entity.id == "oewn-bank.n.01"
        assert entity.description == "a financial institution"

    def test_unknown_id_returns_none(self, fake_wn: types.ModuleType) -> None:
        source = WnEntitySource(lang="en")

        assert source.get("no-such-synset") is None

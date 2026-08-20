from __future__ import annotations

import sys
import types

import pytest

from linkingtk.sources.wn import WnEntitySource, sensekey_to_synset_id


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


class _FakeSense:
    def __init__(self, identifier: str, synset: _FakeSynset) -> None:
        self._identifier = identifier
        self._synset = synset

    def metadata(self) -> dict[str, str]:
        return {"identifier": self._identifier}

    def synset(self) -> _FakeSynset:
        return self._synset


def _fake_wn_module(
    synsets_by_query: dict[str, list[_FakeSynset]],
    synsets_by_id: dict[str, _FakeSynset],
    senses_by_lemma: dict[str, list[_FakeSense]] | None = None,
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

    def senses(
        form: str | None = None, pos: str | None = None, *, lexicon: str | None = None
    ) -> list[_FakeSense]:
        return (senses_by_lemma or {}).get(form or "", [])

    module.synsets = synsets  # type: ignore[attr-defined]
    module.synset = synset  # type: ignore[attr-defined]
    module.senses = senses  # type: ignore[attr-defined]
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


class TestSensekeyToSynsetId:
    def test_matches_by_identifier_metadata(self, monkeypatch: pytest.MonkeyPatch) -> None:
        group_top = _FakeSynset(id="omw-en-00031264-n", words=[], definition=None)
        group_other = _FakeSynset(id="omw-en-14621446-n", words=[], definition=None)
        module = _fake_wn_module(
            synsets_by_query={},
            synsets_by_id={},
            senses_by_lemma={
                "group": [
                    _FakeSense("group%1:03:00::", group_top),
                    _FakeSense("group%1:07:00::", group_other),
                ]
            },
        )
        monkeypatch.setitem(sys.modules, "wn", module)

        assert sensekey_to_synset_id("group%1:03:00::") == "omw-en-00031264-n"
        assert sensekey_to_synset_id("group%1:07:00::") == "omw-en-14621446-n"

    def test_no_matching_sense_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        module = _fake_wn_module(
            synsets_by_query={},
            synsets_by_id={},
            senses_by_lemma={"group": [_FakeSense("group%1:03:00::", _FakeSynset("x", [], None))]},
        )
        monkeypatch.setitem(sys.modules, "wn", module)

        assert sensekey_to_synset_id("group%1:99:00::") is None

    def test_unknown_lemma_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        module = _fake_wn_module(synsets_by_query={}, synsets_by_id={})
        monkeypatch.setitem(sys.modules, "wn", module)

        assert sensekey_to_synset_id("nonexistent%1:03:00::") is None

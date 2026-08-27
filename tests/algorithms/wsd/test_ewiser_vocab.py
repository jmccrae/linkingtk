from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from linkingtk.algorithms.wsd._ewiser_vocab import SenseVocabulary


class _FakeWnError(Exception):
    pass


class _FakeSynset:
    def __init__(self, id: str) -> None:
        self.id = id


def _fake_wn_module(synsets_by_id: dict[str, _FakeSynset]) -> types.ModuleType:
    module = types.ModuleType("wn")

    def synset(id: str, *, lexicon: str | None = None, lang: str | None = None) -> _FakeSynset:
        found = synsets_by_id.get(id)
        if found is None:
            raise module.Error(f"no such synset: {id}")  # type: ignore[attr-defined]
        return found

    module.synset = synset  # type: ignore[attr-defined]
    module.Error = _FakeWnError  # type: ignore[attr-defined]
    return module


@pytest.fixture
def fake_wn(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    module = _fake_wn_module(
        {
            "omw-en-02084071-n": _FakeSynset("omw-en-02084071-n"),
            "omw-en-00007846-n": _FakeSynset("omw-en-00007846-n"),
        }
    )
    monkeypatch.setitem(sys.modules, "wn", module)
    return module


class TestFromOffsetsFile:
    def test_prepends_nspecial_reserved_slots(
        self, fake_wn: types.ModuleType, tmp_path: Path
    ) -> None:
        offsets_file = tmp_path / "offsets.txt"
        offsets_file.write_text("wn:02084071n 10742\n")

        vocabulary = SenseVocabulary.from_offsets_file(offsets_file, nspecial=4)

        assert len(vocabulary) == 5
        for index in range(4):
            assert vocabulary.synset_id_for(index) is None
        assert vocabulary.synset_id_for(4) == "omw-en-02084071-n"

    def test_resolves_offsets_to_synset_ids(
        self, fake_wn: types.ModuleType, tmp_path: Path
    ) -> None:
        offsets_file = tmp_path / "offsets.txt"
        offsets_file.write_text("wn:02084071n 10742\nwn:00007846n 6909\n")

        vocabulary = SenseVocabulary.from_offsets_file(offsets_file, nspecial=0)

        assert vocabulary.index_for("omw-en-02084071-n") == 0
        assert vocabulary.index_for("omw-en-00007846-n") == 1

    def test_non_offset_dummy_token_becomes_reserved_slot(
        self, fake_wn: types.ModuleType, tmp_path: Path
    ) -> None:
        offsets_file = tmp_path / "offsets.txt"
        offsets_file.write_text("wn:02084071n 10742\nmadeupword0000 0\n")

        vocabulary = SenseVocabulary.from_offsets_file(offsets_file, nspecial=0)

        assert vocabulary.synset_id_for(0) == "omw-en-02084071-n"
        assert vocabulary.synset_id_for(1) is None

    def test_unresolvable_offset_becomes_reserved_slot(
        self, fake_wn: types.ModuleType, tmp_path: Path
    ) -> None:
        offsets_file = tmp_path / "offsets.txt"
        offsets_file.write_text("wn:99999999n 0\n")

        vocabulary = SenseVocabulary.from_offsets_file(offsets_file, nspecial=0)

        assert vocabulary.synset_id_for(0) is None

    def test_records_the_lexicon_it_was_built_from(
        self, fake_wn: types.ModuleType, tmp_path: Path
    ) -> None:
        offsets_file = tmp_path / "offsets.txt"
        offsets_file.write_text("wn:02084071n 10742\n")

        vocabulary = SenseVocabulary.from_offsets_file(offsets_file, lexicon="omw-en:1.4")

        assert vocabulary.lexicon == "omw-en:1.4"


class TestFromWn:
    def test_deduplicates_and_sorts(self) -> None:
        vocabulary = SenseVocabulary.from_wn(
            ["omw-en-00007846-n", "omw-en-02084071-n", "omw-en-00007846-n"], nspecial=4
        )

        assert len(vocabulary) == 6
        assert vocabulary.synset_id_for(4) == "omw-en-00007846-n"
        assert vocabulary.synset_id_for(5) == "omw-en-02084071-n"

    def test_reserved_slots_have_no_index(self) -> None:
        vocabulary = SenseVocabulary.from_wn(["omw-en-02084071-n"], nspecial=2)

        assert vocabulary.index_for("omw-en-02084071-n") == 2

    def test_has_no_lexicon(self) -> None:
        vocabulary = SenseVocabulary.from_wn(["omw-en-02084071-n"])

        assert vocabulary.lexicon is None


class TestIndexForSynsetIdFor:
    def test_unknown_synset_id_returns_none(self) -> None:
        vocabulary = SenseVocabulary.from_wn(["omw-en-02084071-n"], nspecial=0)

        assert vocabulary.index_for("omw-en-nonexistent-n") is None

    def test_round_trips(self) -> None:
        vocabulary = SenseVocabulary.from_wn(["omw-en-02084071-n", "omw-en-00007846-n"], nspecial=1)

        for index in range(len(vocabulary)):
            synset_id = vocabulary.synset_id_for(index)
            if synset_id is not None:
                assert vocabulary.index_for(synset_id) == index

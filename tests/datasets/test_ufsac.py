from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from linkingtk.datasets.ufsac import UfsacDataset
from linkingtk.sources.wn import WnEntitySource

_XML = """<?xml version="1.0" encoding="UTF-8"?>
<corpus>
  <document id="d001">
    <paragraph>
      <sentence id="s1">
        <word surface_form="The" lemma="the" pos="DT"/>
        <word surface_form="group" lemma="group" pos="NN" wn30_key="group%1:03:00::"/>
        <word surface_form="met" lemma="meet" pos="VBD" wn30_key="meet%2:41:00::"/>
        <word surface_form="." lemma="." pos="."/>
      </sentence>
    </paragraph>
  </document>
</corpus>
"""


class _FakeSynset:
    def __init__(self, id: str) -> None:
        self.id = id


class _FakeSense:
    def __init__(self, identifier: str, synset_id: str) -> None:
        self._identifier = identifier
        self._synset = _FakeSynset(synset_id)

    def metadata(self) -> dict[str, str]:
        return {"identifier": self._identifier}

    def synset(self) -> _FakeSynset:
        return self._synset


_SENSES_BY_LEMMA = {
    "group": [_FakeSense("group%1:03:00::", "omw-en-00031264-n")],
    "meet": [_FakeSense("meet%2:41:00::", "omw-en-01234567-v")],
}


@pytest.fixture(autouse=True)
def fake_wn(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    module = types.ModuleType("wn")

    def senses(form: str | None = None, *, lexicon: str | None = None) -> list[_FakeSense]:
        return _SENSES_BY_LEMMA.get(form or "", [])

    module.senses = senses  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "wn", module)
    return module


@pytest.fixture
def fixture_source(tmp_path: Path) -> str:
    path = tmp_path / "semcor.xml"
    path.write_text(_XML)
    return str(path)


class TestLoad:
    def test_parses_words_with_sense_keys_as_mentions(self, fixture_source: str) -> None:
        mentions, senses, ground_truth = UfsacDataset(source=fixture_source).load()

        assert {m.labels[0] for m in mentions} == {"group", "met"}
        assert isinstance(senses, WnEntitySource)

    def test_reconstructs_sentence_context_by_joining_surface_forms(
        self, fixture_source: str
    ) -> None:
        mentions, _senses, _gt = UfsacDataset(source=fixture_source).load()

        group_mention = next(m for m in mentions if m.labels == ["group"])
        text, start, end = group_mention.context
        assert text[start:end] == "group"
        assert text == "The group met ."

    def test_ground_truth_resolves_sense_keys_to_synset_ids(self, fixture_source: str) -> None:
        _mentions, _senses, ground_truth = UfsacDataset(source=fixture_source).load()

        gt_by_mention = dict(ground_truth)
        mention_id = "ufsac:d001:s1:1"
        assert gt_by_mention[mention_id] == "omw-en-00031264-n"

    def test_unresolvable_sense_key_drops_the_mention(self, tmp_path: Path) -> None:
        xml = _XML.replace('wn30_key="group%1:03:00::"', 'wn30_key="group%9:99:99::"')
        path = tmp_path / "corpus.xml"
        path.write_text(xml)

        mentions, _senses, ground_truth = UfsacDataset(source=str(path)).load()

        assert "group" not in {m.labels[0] for m in mentions}
        assert "ufsac:d001:s1:1" not in dict(ground_truth)

    def test_lexicon_is_forwarded_to_the_entity_source(self, fixture_source: str) -> None:
        _mentions, senses, _gt = UfsacDataset(source=fixture_source, lexicon="omw-en:2.0").load()

        assert senses.lexicon == "omw-en:2.0"


class TestXzCompression:
    def test_decompresses_xz_suffixed_source(self, tmp_path: Path) -> None:
        import lzma

        path = tmp_path / "semcor.xml.xz"
        path.write_bytes(lzma.compress(_XML.encode()))

        mentions, _senses, _gt = UfsacDataset(source=str(path)).load()

        assert {m.labels[0] for m in mentions} == {"group", "met"}

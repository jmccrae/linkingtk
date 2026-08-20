from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import yaml

from linkingtk.datasets.semcor import SemCorDataset
from linkingtk.sources.wn import WnEntitySource

_DOC_A = {
    "sOM7": {
        "text": "The bank is closed.",
        "tokens": [[0, 3], [4, 8], [9, 11], [12, 18], [18, 19]],
        "lemmas": ["the", "bank", "be", "close", "."],
        "oewn_key": [[1, "oewn-08420278-n"]],
    },
    "0PBR": {
        "text": "It reopens tomorrow.",
        "tokens": [[0, 2], [3, 10], [11, 19], [19, 20]],
        "lemmas": ["it", "reopen", "tomorrow", "."],
        "oewn_key": [[1, "oewn-00296178-v"]],
    },
}
_DOC_B = {
    "z8MI": {
        "text": "A crane lifted the beam.",
        "tokens": [[0, 1], [2, 7], [8, 14], [15, 18], [19, 23], [23, 24]],
        "lemmas": ["a", "crane", "lift", "the", "beam", "."],
        "oewn_key": [[1, "oewn-03327841-n"]],
    },
}


@pytest.fixture(autouse=True)
def fake_wn(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "wn", types.ModuleType("wn"))


@pytest.fixture
def fixture_source(tmp_path: Path) -> str:
    press = tmp_path / "data" / "press_reportage"
    fiction = tmp_path / "data" / "fiction_general"
    press.mkdir(parents=True)
    fiction.mkdir(parents=True)
    (press / "br-a01.yaml").write_text(yaml.safe_dump(_DOC_A))
    (fiction / "br-k01.yaml").write_text(yaml.safe_dump(_DOC_B))
    return str(tmp_path)


class TestLoad:
    def test_parses_every_sense_tagged_token_across_all_documents(
        self, fixture_source: str
    ) -> None:
        mentions, senses, ground_truth = SemCorDataset(source=fixture_source).load()

        assert len(mentions) == 3
        assert len(ground_truth) == 3
        assert isinstance(senses, WnEntitySource)

    def test_mention_carries_surface_text_and_char_span_context(self, fixture_source: str) -> None:
        mentions, _senses, _gt = SemCorDataset(source=fixture_source).load()

        bank_mention = next(m for m in mentions if m.labels == ["bank"])
        assert bank_mention.context == ("The bank is closed.", 4, 8)

    def test_labels_use_lemma_not_inflected_surface_form(self, fixture_source: str) -> None:
        mentions, _senses, _gt = SemCorDataset(source=fixture_source).load()

        reopens_mention = next(m for m in mentions if m.labels == ["reopen"])
        text, start, end = reopens_mention.context
        assert text[start:end] == "reopens"

    def test_ground_truth_maps_mention_id_to_oewn_synset_id(self, fixture_source: str) -> None:
        _mentions, _senses, ground_truth = SemCorDataset(source=fixture_source).load()

        gt_by_mention = dict(ground_truth)
        bank_mention_id = "semcor:press_reportage/br-a01:sOM7:1"
        assert gt_by_mention[bank_mention_id] == "oewn-08420278-n"

    def test_categories_filters_to_named_subdirectories(self, fixture_source: str) -> None:
        mentions, _senses, _gt = SemCorDataset(
            source=fixture_source, categories=["fiction_general"]
        ).load()

        assert len(mentions) == 1
        assert mentions[0].labels == ["crane"]

    def test_lexicon_is_forwarded_to_the_entity_source(self, fixture_source: str) -> None:
        _mentions, senses, _gt = SemCorDataset(source=fixture_source, lexicon="oewn:2020").load()

        assert senses.lexicon == "oewn:2020"

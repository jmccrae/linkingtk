from pathlib import Path

import pandas as pd
import pytest

from linkingtk.datasets.aida_conll import AidaConllDataset, fetch_wikipedia_extracts

# One tiny document per split, each with one real (non-NIL) mention plus one
# NIL mention (qid=None) -- confirms NIL mentions are skipped entirely, matching
# AIDA-CoNLL's own "InKB" convention (see module docstring).
_ROWS_BY_SPLIT = {
    "train": {
        "document_id": 1,
        "text": "Paris is nice. NASA launched a rocket.",
        "entities": [
            {"start": 0, "end": 5, "tag": "LOC", "pageid": 1, "qid": 90, "title": "Paris"},
            {"start": 16, "end": 20, "tag": "ORG", "pageid": None, "qid": None, "title": None},
        ],
    },
    "validation": {
        "document_id": 2,
        "text": "Berlin is the capital of Germany.",
        "entities": [
            {"start": 0, "end": 6, "tag": "LOC", "pageid": 2, "qid": 3787, "title": "Berlin"},
        ],
    },
    "test": {
        "document_id": 3,
        "text": "Rome_Airport is closed today.",
        "entities": [
            {
                "start": 0,
                "end": 12,
                "tag": "LOC",
                "pageid": 3,
                "qid": 220,
                "title": "Rome_(city)",
            },
        ],
    },
}


def _fake_fetcher(titles: list[str]) -> dict[str, str]:
    return {title: f"description of {title}" for title in titles}


def _raising_fetcher(titles: list[str]) -> dict[str, str]:
    raise AssertionError("description_fetcher should not be called by load_splits()")


@pytest.fixture
def fixture_source(tmp_path: Path) -> str:
    for split, row in _ROWS_BY_SPLIT.items():
        pd.DataFrame([row]).to_parquet(tmp_path / f"{split}.parquet")
    return str(tmp_path)


class TestLoad:
    def test_mentions_exclude_nil_entities(self, fixture_source: str) -> None:
        mentions, _kb, _ground_truth = AidaConllDataset(
            source=fixture_source, description_fetcher=_fake_fetcher
        ).load()

        assert len(mentions) == 3  # one real mention per split, NIL excluded
        assert {"mention:1:0:5", "mention:2:0:6", "mention:3:0:12"} == {m.id for m in mentions}

    def test_mention_context_and_label(self, fixture_source: str) -> None:
        mentions, _kb, _ground_truth = AidaConllDataset(
            source=fixture_source, description_fetcher=_fake_fetcher
        ).load()

        paris = next(m for m in mentions if m.id == "mention:1:0:5")
        assert paris.labels == ["Paris"]
        assert paris.context == ("Paris is nice. NASA launched a rocket.", 0, 5)

    def test_kb_entities_labeled_and_described(self, fixture_source: str) -> None:
        _mentions, kb, _ground_truth = AidaConllDataset(
            source=fixture_source, description_fetcher=_fake_fetcher
        ).load()

        by_id = {e.id: e for e in kb}
        assert set(by_id) == {"Q90", "Q3787", "Q220"}
        assert by_id["Q220"].labels == ["Rome (city)"]  # underscores humanized
        assert by_id["Q220"].description == "description of Rome_(city)"

    def test_ground_truth_covers_all_splits(self, fixture_source: str) -> None:
        _mentions, _kb, ground_truth = AidaConllDataset(
            source=fixture_source, description_fetcher=_fake_fetcher
        ).load()

        assert set(ground_truth) == {
            ("mention:1:0:5", "Q90"),
            ("mention:2:0:6", "Q3787"),
            ("mention:3:0:12", "Q220"),
        }


class TestLoadSplits:
    def test_returns_train_test_val_order(self, fixture_source: str) -> None:
        train, test, val = AidaConllDataset(
            source=fixture_source, description_fetcher=_raising_fetcher
        ).load_splits()

        assert train == [("mention:1:0:5", "Q90")]
        assert test == [("mention:3:0:12", "Q220")]
        assert val == [("mention:2:0:6", "Q3787")]

    def test_does_not_fetch_descriptions(self, fixture_source: str) -> None:
        # _raising_fetcher would fail the test if called -- load_splits() only
        # needs mention/entity ids, not full KB Entity objects with descriptions.
        AidaConllDataset(source=fixture_source, description_fetcher=_raising_fetcher).load_splits()


class TestFetchWikipediaExtracts:
    def test_empty_titles_returns_empty(self) -> None:
        assert fetch_wikipedia_extracts([]) == {}

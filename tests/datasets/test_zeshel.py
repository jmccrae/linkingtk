from pathlib import Path

import pandas as pd
import pytest

from linkingtk.datasets.zeshel import ZeshelDataset

# Two domains ("alpha" train-only, "beta" test-only) so KB-filtering-by-
# domain is actually exercised -- a KB entity from a domain that never
# appears in any mention split must not show up in load()'s kb list.
_MENTION_ROWS_BY_SPLIT = {
    "train": [
        {
            "subset": "alpha",
            "id": "doc1",
            "text": "Paris is the capital of France.",
            "entities": [{"start": 0, "end": 5, "label": ["Q90"]}],
        }
    ],
    "validation": [
        {
            "subset": "alpha",
            "id": "doc2",
            "text": "Berlin is nice.",
            "entities": [{"start": 0, "end": 6, "label": ["Q3787"]}],
        }
    ],
    "test": [
        {
            "subset": "beta",
            "id": "doc3",
            "text": "Rome has old ruins.",
            "entities": [
                {"start": 0, "end": 4, "label": ["Q220"]},
                {"start": 0, "end": 0, "label": []},  # no label -- skipped
            ],
        }
    ],
}

_KB_ROWS = [
    {"id": "Q90", "subset": "alpha", "name": "Paris", "description": "capital of France"},
    {"id": "Q3787", "subset": "alpha", "name": "Berlin", "description": "capital of Germany"},
    {"id": "Q220", "subset": "beta", "name": "Rome", "description": "capital of Italy"},
    # never referenced by any mention split -- must be filtered out of load()'s kb.
    {"id": "Q64", "subset": "gamma", "name": "Unused", "description": "not in any split"},
]


@pytest.fixture
def dataset(tmp_path: Path) -> ZeshelDataset:
    mentions_dir = tmp_path / "mentions"
    mentions_dir.mkdir()
    for split, rows in _MENTION_ROWS_BY_SPLIT.items():
        pd.DataFrame(rows).to_parquet(mentions_dir / f"{split}.parquet")

    kb_dir = tmp_path / "kb"
    kb_dir.mkdir()
    pd.DataFrame(_KB_ROWS).to_parquet(kb_dir / "kb.parquet")

    return ZeshelDataset(mentions_source=str(mentions_dir), kb_source=str(kb_dir))


class TestLoad:
    def test_mentions_carry_context_domain_and_skip_unlabeled(self, dataset: ZeshelDataset) -> None:
        mentions, _kb, _ground_truth = dataset.load()

        assert {m.id for m in mentions} == {
            "mention:doc1:0:5",
            "mention:doc2:0:6",
            "mention:doc3:0:4",
        }
        paris = next(m for m in mentions if m.id == "mention:doc1:0:5")
        assert paris.labels == ["Paris"]
        assert paris.context == ("Paris is the capital of France.", 0, 5)
        assert paris.properties == {"domain": "alpha"}

    def test_ground_truth_covers_all_splits(self, dataset: ZeshelDataset) -> None:
        _mentions, _kb, ground_truth = dataset.load()

        assert set(ground_truth) == {
            ("mention:doc1:0:5", "Q90"),
            ("mention:doc2:0:6", "Q3787"),
            ("mention:doc3:0:4", "Q220"),
        }

    def test_kb_filtered_to_domains_present_in_mentions(self, dataset: ZeshelDataset) -> None:
        _mentions, kb, _ground_truth = dataset.load()

        assert {e.id for e in kb} == {"Q90", "Q3787", "Q220"}  # "Q64" (domain gamma) excluded
        rome = next(e for e in kb if e.id == "Q220")
        assert rome.labels == ["Rome"]
        assert rome.description == "capital of Italy"
        assert rome.properties == {"domain": "beta"}


class TestLoadSplits:
    def test_returns_train_test_val_order(self, dataset: ZeshelDataset) -> None:
        train, test, val = dataset.load_splits()

        assert train == [("mention:doc1:0:5", "Q90")]
        assert test == [("mention:doc3:0:4", "Q220")]
        assert val == [("mention:doc2:0:6", "Q3787")]

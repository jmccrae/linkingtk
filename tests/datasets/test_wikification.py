import zipfile
from pathlib import Path

import pytest

from linkingtk.datasets.wikification import (
    Ace2004Dataset,
    AquaintDataset,
    MsnbcDataset,
    WikipediaSampleDataset,
)

_FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "wikification_toy"


def _zip_url(tmp_path: Path) -> str:
    """Zip fixtures/wikification_toy's whole tree, preserving relative paths."""
    zip_path = tmp_path / "data.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        for file in _FIXTURES_DIR.rglob("*"):
            if file.is_file():
                archive.write(file, arcname=str(file.relative_to(_FIXTURES_DIR)))
    return f"file://{zip_path}"


def _fake_fetcher(titles: list[str]) -> dict[str, str]:
    return {title: f"description of {title}" for title in titles}


def _raising_fetcher(titles: list[str]) -> dict[str, str]:
    raise AssertionError("description_fetcher should not be called by load_splits()")


@pytest.fixture
def zip_url(tmp_path: Path) -> str:
    return _zip_url(tmp_path)


class TestMsnbcDataset:
    def test_skips_nil_mentions(self, zip_url: str) -> None:
        mentions, _kb, ground_truth = MsnbcDataset(
            zip_url=zip_url, description_fetcher=_fake_fetcher
        ).load()

        assert len(mentions) == 1
        assert len(ground_truth) == 1

    def test_mention_context_and_label(self, zip_url: str) -> None:
        mentions, _kb, _ground_truth = MsnbcDataset(
            zip_url=zip_url, description_fetcher=_fake_fetcher
        ).load()

        paris = mentions[0]
        assert paris.id == "mention:Doc1.txt:0:5"
        assert paris.labels == ["Paris"]
        assert paris.context == ("Paris is nice. NASA launched a rocket.", 0, 5)

    def test_kb_entity_from_wikipedia_url(self, zip_url: str) -> None:
        _mentions, kb, ground_truth = MsnbcDataset(
            zip_url=zip_url, description_fetcher=_fake_fetcher
        ).load()

        assert ground_truth == [("mention:Doc1.txt:0:5", "Paris")]
        by_id = {e.id: e for e in kb}
        assert set(by_id) == {"Paris"}
        assert by_id["Paris"].labels == ["Paris"]
        assert by_id["Paris"].description == "description of Paris"


class TestAce2004Dataset:
    def test_loads_from_its_own_zip_subfolder(self, zip_url: str) -> None:
        mentions, kb, ground_truth = Ace2004Dataset(
            zip_url=zip_url, description_fetcher=_fake_fetcher
        ).load()

        assert [m.id for m in mentions] == ["mention:AceDoc1:0:6"]
        assert ground_truth == [("mention:AceDoc1:0:6", "Berlin")]
        assert {e.id for e in kb} == {"Berlin"}


class TestAquaintDataset:
    def test_loads_from_its_own_zip_subfolder(self, zip_url: str) -> None:
        mentions, kb, ground_truth = AquaintDataset(
            zip_url=zip_url, description_fetcher=_fake_fetcher
        ).load()

        assert [m.id for m in mentions] == ["mention:AquaintDoc1:0:4"]
        assert ground_truth == [("mention:AquaintDoc1:0:4", "Rome")]
        assert {e.id for e in kb} == {"Rome"}


class TestWikipediaSampleDataset:
    def test_load_concatenates_train_and_test(self, zip_url: str) -> None:
        mentions, kb, ground_truth = WikipediaSampleDataset(
            zip_url=zip_url, description_fetcher=_fake_fetcher, max_train_documents=None
        ).load()

        assert {m.id for m in mentions} == {
            "mention:TrainDoc1:0:6",
            "mention:TrainDoc2:0:6",
            "mention:TestDoc1:0:5",
        }
        assert set(ground_truth) == {
            ("mention:TrainDoc1:0:6", "London"),
            ("mention:TrainDoc2:0:6", "Madrid"),
            ("mention:TestDoc1:0:5", "Tokyo"),
        }
        assert {e.id for e in kb} == {"London", "Madrid", "Tokyo"}

    def test_bare_title_chosen_annotation_used_directly(self, zip_url: str) -> None:
        # Unlike Msnbc/Ace2004/Aquaint's full Wikipedia URLs, WikipediaSample's
        # ChosenAnnotation is already a bare title -- confirms _entity_title()
        # passes it through unchanged rather than treating it as a URL.
        _mentions, kb, _ground_truth = WikipediaSampleDataset(
            zip_url=zip_url, description_fetcher=_fake_fetcher, max_train_documents=None
        ).load()

        by_id = {e.id: e for e in kb}
        assert by_id["London"].labels == ["London"]

    def test_max_train_documents_caps_train_split(self, zip_url: str) -> None:
        mentions, _kb, _ground_truth = WikipediaSampleDataset(
            zip_url=zip_url, description_fetcher=_fake_fetcher, max_train_documents=1
        ).load()

        train_mention_ids = {m.id for m in mentions if m.id.startswith("mention:Train")}
        assert train_mention_ids == {"mention:TrainDoc1:0:6"}


class TestWikipediaSampleDatasetLoadSplits:
    def test_returns_train_test_val_order(self, zip_url: str) -> None:
        train, test, val = WikipediaSampleDataset(
            zip_url=zip_url, description_fetcher=_raising_fetcher, max_train_documents=None
        ).load_splits()

        assert set(train) == {
            ("mention:TrainDoc1:0:6", "London"),
            ("mention:TrainDoc2:0:6", "Madrid"),
        }
        assert test == [("mention:TestDoc1:0:5", "Tokyo")]
        assert val == []

    def test_does_not_fetch_descriptions(self, zip_url: str) -> None:
        # _raising_fetcher would fail the test if called.
        WikipediaSampleDataset(
            zip_url=zip_url, description_fetcher=_raising_fetcher, max_train_documents=None
        ).load_splits()

    def test_max_train_documents_applies_to_splits_too(self, zip_url: str) -> None:
        train, _test, _val = WikipediaSampleDataset(
            zip_url=zip_url, description_fetcher=_raising_fetcher, max_train_documents=1
        ).load_splits()

        assert train == [("mention:TrainDoc1:0:6", "London")]

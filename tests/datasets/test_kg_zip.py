import zipfile
from pathlib import Path
from typing import ClassVar

from linkingtk.datasets.dbp15k import DBP15KFrEnDataset, DBP15KJaEnDataset, DBP15KZhEnDataset
from linkingtk.datasets.icews import IcewsWikiDataset
from linkingtk.datasets.kg_zip import _KGZipDataset
from linkingtk.datasets.openea import (
    DbpediaWikidata15KDataset,
    DbpediaYago15KDataset,
    EnDe15KDataset,
    EnFr15KDataset,
)

_FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


def _zip_url(tmp_path: Path, fixture_dir: str, folder: str) -> str:
    """Zip a fixtures/<fixture_dir> directory's files under <folder>/ in a tmp zip."""
    zip_path = tmp_path / "data.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        for file in (_FIXTURES_DIR / fixture_dir).iterdir():
            archive.write(file, arcname=f"{folder}/{file.name}")
    return f"file://{zip_path}"


class _ToyDataset(_KGZipDataset):
    """Test-only concrete subclass: DBP15K/OpenEA-style single ill_ent_ids file."""

    _folder = "toy"
    _ground_truth_files: ClassVar[tuple[str, ...]] = ("ill_ent_ids",)
    _train_ground_truth_file = "sup_ent_ids"
    _test_ground_truth_file = "ref_ent_ids"
    _val_ground_truth_file = "val_ent_ids"


class _ToyIcewsStyleDataset(_KGZipDataset):
    """Test-only concrete subclass: ICEWS-style split ref/sup pairs + 5-column triples."""

    _folder = "icews_toy"
    _ground_truth_files: ClassVar[tuple[str, ...]] = ("ref_pairs", "sup_pairs")
    _train_ground_truth_file = "sup_pairs"
    _test_ground_truth_file = "ref_pairs"


class TestEntitiesAndLabels:
    def test_labels_from_uri_local_name(self, tmp_path: Path) -> None:
        dataset = _ToyDataset(zip_url=_zip_url(tmp_path, "kg_zip_toy", "toy"))
        entities1, entities2, _ = dataset.load()

        assert [e.labels for e in entities1] == [["Cat"], ["Dog"], ["Bird"]]
        assert [e.labels for e in entities2] == [["Cat"], ["Dog"], ["Fish"]]

    def test_entity_ids_are_namespaced_by_dataset_and_side(self, tmp_path: Path) -> None:
        dataset = _ToyDataset(zip_url=_zip_url(tmp_path, "kg_zip_toy", "toy"))
        entities1, entities2, _ = dataset.load()

        assert [e.id for e in entities1] == ["toy:1:0", "toy:1:1", "toy:1:2"]
        assert [e.id for e in entities2] == ["toy:2:100", "toy:2:101", "toy:2:102"]


class TestGroundTruth:
    def test_single_ground_truth_file_drops_dangling_pairs(self, tmp_path: Path) -> None:
        dataset = _ToyDataset(zip_url=_zip_url(tmp_path, "kg_zip_toy", "toy"))
        _, _, ground_truth = dataset.load()

        # ill_ent_ids has a third pair (2 -> 999) whose target isn't in ent_ids_2.
        assert ground_truth == [("toy:1:0", "toy:2:100"), ("toy:1:1", "toy:2:101")]

    def test_multiple_ground_truth_files_are_concatenated(self, tmp_path: Path) -> None:
        dataset = _ToyIcewsStyleDataset(zip_url=_zip_url(tmp_path, "kg_zip_icews_toy", "icews_toy"))
        _, _, ground_truth = dataset.load()

        assert ground_truth == [
            ("icews_toy:1:0", "icews_toy:2:50"),
            ("icews_toy:1:1", "icews_toy:2:51"),
        ]


class TestSplits:
    def test_dbp15k_openea_style_train_test_val_split(self, tmp_path: Path) -> None:
        dataset = _ToyDataset(zip_url=_zip_url(tmp_path, "kg_zip_toy", "toy"))
        train, test, val = dataset.load_splits()

        assert train == [("toy:1:0", "toy:2:100")]
        assert test == [("toy:1:1", "toy:2:101")]
        # val_ent_ids' only pair (2 -> 999) is dangling, like ill_ent_ids' third pair.
        assert val == []

    def test_icews_style_train_test_split_has_no_val(self, tmp_path: Path) -> None:
        dataset = _ToyIcewsStyleDataset(zip_url=_zip_url(tmp_path, "kg_zip_icews_toy", "icews_toy"))
        train, test, val = dataset.load_splits()

        assert train == [("icews_toy:1:1", "icews_toy:2:51")]
        assert test == [("icews_toy:1:0", "icews_toy:2:50")]
        assert val == []


class TestGraphs:
    def test_triples_are_namespaced_like_entity_ids(self, tmp_path: Path) -> None:
        dataset = _ToyDataset(zip_url=_zip_url(tmp_path, "kg_zip_toy", "toy"))
        graph1, graph2 = dataset.load_graphs()

        assert graph1 == [("toy:1:0", "10", "toy:1:1"), ("toy:1:1", "11", "toy:1:2")]
        assert graph2 == [("toy:2:100", "20", "toy:2:101")]

    def test_extra_trailing_columns_are_ignored(self, tmp_path: Path) -> None:
        dataset = _ToyIcewsStyleDataset(zip_url=_zip_url(tmp_path, "kg_zip_icews_toy", "icews_toy"))
        graph1, graph2 = dataset.load_graphs()

        assert graph1 == [("icews_toy:1:0", "5", "icews_toy:1:1")]
        assert graph2 == [("icews_toy:2:50", "6", "icews_toy:2:51")]


class TestConcreteDatasetWiring:
    def test_dbp15k_variants_point_at_distinct_folders(self) -> None:
        assert DBP15KZhEnDataset()._folder == "data/zh_en"
        assert DBP15KJaEnDataset()._folder == "data/ja_en"
        assert DBP15KFrEnDataset()._folder == "data/fr_en"

    def test_openea_variants_point_at_distinct_folders(self) -> None:
        assert EnFr15KDataset()._folder == "data/en_fr_15k_V1"
        assert EnDe15KDataset()._folder == "data/en_de_15k_V1"
        assert DbpediaWikidata15KDataset()._folder == "data/dbp_wd_15k_V1"
        assert DbpediaYago15KDataset()._folder == "data/dbp_yg_15k_V1"

    def test_dbp15k_and_openea_share_the_entmatcher_zip(self) -> None:
        zip_url = DBP15KZhEnDataset()._zip_url
        assert zip_url == EnFr15KDataset()._zip_url
        assert "EntMatcher" in zip_url

    def test_icews_wiki_uses_ref_and_sup_pairs(self) -> None:
        dataset = IcewsWikiDataset()
        assert dataset._folder == "icews_wiki"
        assert dataset._ground_truth_files == ("ref_pairs", "sup_pairs")

    def test_dbp15k_and_openea_variants_expose_native_splits(self) -> None:
        for dataset in (
            DBP15KZhEnDataset(),
            DBP15KJaEnDataset(),
            DBP15KFrEnDataset(),
            EnFr15KDataset(),
            EnDe15KDataset(),
            DbpediaWikidata15KDataset(),
            DbpediaYago15KDataset(),
        ):
            assert dataset._train_ground_truth_file == "sup_ent_ids"
            assert dataset._test_ground_truth_file == "ref_ent_ids"
            assert dataset._val_ground_truth_file == "val_ent_ids"

    def test_icews_wiki_exposes_native_split_with_no_val(self) -> None:
        dataset = IcewsWikiDataset()
        assert dataset._train_ground_truth_file == "sup_pairs"
        assert dataset._test_ground_truth_file == "ref_pairs"
        assert dataset._val_ground_truth_file is None

    def test_zip_url_override_takes_precedence(self, tmp_path: Path) -> None:
        override = _zip_url(tmp_path, "kg_zip_toy", "toy")
        assert DBP15KZhEnDataset(zip_url=override).zip_url == override

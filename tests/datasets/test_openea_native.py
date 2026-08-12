import zipfile
from pathlib import Path

from linkingtk.datasets.openea_native import (
    DbpediaWikidata15KAttrDataset,
    DbpediaYago15KAttrDataset,
    EnDe15KAttrDataset,
    EnFr15KAttrDataset,
    _OpenEANativeDataset,
)

_FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "openea_native_toy"


class _ToyDataset(_OpenEANativeDataset):
    """Test-only concrete subclass pointed at the fixture zip."""

    _dataset_name = "toy_attr"


def _zip_url(tmp_path: Path) -> str:
    zip_path = tmp_path / "data.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        for file in _FIXTURES_DIR.iterdir():
            archive.write(file, arcname=file.name)
    return f"file://{zip_path}"


class TestEntitiesAndLabels:
    def test_entities_derived_from_rel_triples_union(self, tmp_path: Path) -> None:
        dataset = _ToyDataset(zip_url=_zip_url(tmp_path))
        entities1, entities2, _ = dataset.load()

        assert [e.id for e in entities1] == ["toy_attr:1:0", "toy_attr:1:1", "toy_attr:1:2"]
        assert [e.labels for e in entities1] == [["A"], ["B"], ["C"]]
        assert [e.id for e in entities2] == ["toy_attr:2:0", "toy_attr:2:1", "toy_attr:2:2"]
        assert [e.labels for e in entities2] == [["X"], ["Y"], ["Z"]]


class TestGroundTruth:
    def test_load_concatenates_all_three_link_files(self, tmp_path: Path) -> None:
        dataset = _ToyDataset(zip_url=_zip_url(tmp_path))
        _, _, ground_truth = dataset.load()

        # test_links' second pair (D -> Z) is dangling: D never appears in
        # rel_triples_1, so it's dropped.
        assert ground_truth == [
            ("toy_attr:1:0", "toy_attr:2:0"),
            ("toy_attr:1:1", "toy_attr:2:1"),
            ("toy_attr:1:2", "toy_attr:2:2"),
        ]


class TestSplits:
    def test_load_splits_matches_link_files(self, tmp_path: Path) -> None:
        dataset = _ToyDataset(zip_url=_zip_url(tmp_path))
        train, test, val = dataset.load_splits()

        assert train == [("toy_attr:1:0", "toy_attr:2:0")]
        assert val == [("toy_attr:1:1", "toy_attr:2:1")]
        assert test == [("toy_attr:1:2", "toy_attr:2:2")]


class TestGraphs:
    def test_triples_are_namespaced_like_entity_ids(self, tmp_path: Path) -> None:
        dataset = _ToyDataset(zip_url=_zip_url(tmp_path))
        graph1, graph2 = dataset.load_graphs()

        assert graph1 == [
            ("toy_attr:1:0", "http://ex.org/rel/next", "toy_attr:1:1"),
            ("toy_attr:1:1", "http://ex.org/rel/next", "toy_attr:1:2"),
        ]
        assert graph2 == [
            ("toy_attr:2:0", "http://ex.org/rel/next", "toy_attr:2:1"),
            ("toy_attr:2:1", "http://ex.org/rel/next", "toy_attr:2:2"),
        ]


class TestAttributeTriples:
    def test_attribute_triples_are_parsed_and_namespaced(self, tmp_path: Path) -> None:
        dataset = _ToyDataset(zip_url=_zip_url(tmp_path))
        attrs1, attrs2 = dataset.load_attribute_triples()

        # attr_triples_1's third row (subject D) is dropped -- D never
        # appears in rel_triples_1.
        assert attrs1 == [
            ("toy_attr:1:0", "http://ex.org/attr/pop", "1234"),
            ("toy_attr:1:1", "http://ex.org/attr/name", "Beta"),
        ]
        assert attrs2 == [("toy_attr:2:0", "http://ex.org/attr/pop", "5678")]

    def test_typed_literal_and_plain_value_both_parsed(self, tmp_path: Path) -> None:
        dataset = _ToyDataset(zip_url=_zip_url(tmp_path))
        attrs1, _ = dataset.load_attribute_triples()

        typed_value = next(v for _, p, v in attrs1 if p == "http://ex.org/attr/pop")
        plain_value = next(v for _, p, v in attrs1 if p == "http://ex.org/attr/name")
        assert typed_value == "1234"
        assert plain_value == "Beta"


class TestConcreteDatasetWiring:
    def test_variants_point_at_distinct_datasets(self) -> None:
        assert EnFr15KAttrDataset()._dataset_name == "en_fr_15k_v1_attr"
        assert EnDe15KAttrDataset()._dataset_name == "en_de_15k_v1_attr"
        assert DbpediaWikidata15KAttrDataset()._dataset_name == "dbp_wd_15k_v1_attr"
        assert DbpediaYago15KAttrDataset()._dataset_name == "dbp_yg_15k_v1_attr"

    def test_variants_point_at_distinct_hosts(self) -> None:
        urls = {
            EnFr15KAttrDataset()._zip_url,
            EnDe15KAttrDataset()._zip_url,
            DbpediaWikidata15KAttrDataset()._zip_url,
            DbpediaYago15KAttrDataset()._zip_url,
        }
        assert len(urls) == 4
        assert all("huggingface.co/datasets/matchbench" in url for url in urls)

    def test_zip_url_override_takes_precedence(self, tmp_path: Path) -> None:
        override = _zip_url(tmp_path)
        assert EnFr15KAttrDataset(zip_url=override).zip_url == override

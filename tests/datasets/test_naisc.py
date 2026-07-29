from pathlib import Path

from linkingtk.datasets.naisc import AnatomyDataset, ConferenceDataset

_FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "naisc_toy"


def _local_dataset() -> ConferenceDataset:
    return ConferenceDataset(base_url=f"file://{_FIXTURES_DIR}")


def test_loads_classes_with_labels() -> None:
    left, right, _ = _local_dataset().load()

    left_by_id = {entity.id: entity for entity in left}
    assert left_by_id["http://left#Person"].labels == ["Person"]
    assert left_by_id["http://left#Paper"].labels == ["Paper"]
    # No <rdfs:label> in the fixture -> falls back to a humanized local name.
    assert left_by_id["http://left#Program_Committee"].labels == ["Program Committee"]

    right_ids = {entity.id for entity in right}
    assert right_ids == {
        "http://right#Person",
        "http://right#Document",
        "http://right#ProgramCommittee",
    }


def test_excludes_object_properties_from_entities() -> None:
    left, right, _ = _local_dataset().load()
    left_ids = {entity.id for entity in left}
    right_ids = {entity.id for entity in right}

    assert "http://left#hasReview" not in left_ids
    assert "http://right#reviews" not in right_ids


def test_ground_truth_filters_dangling_and_property_only_pairs() -> None:
    _, _, ground_truth = _local_dataset().load()

    assert set(ground_truth) == {
        ("http://left#Person", "http://right#Person"),
        ("http://left#Paper", "http://right#Document"),
        ("http://left#Program_Committee", "http://right#ProgramCommittee"),
    }


def test_conference_dataset_default_base_url() -> None:
    assert ConferenceDataset().base_url.endswith("/datasets/conference")


def test_anatomy_dataset_default_base_url() -> None:
    assert AnatomyDataset().base_url.endswith("/datasets/anatomy")

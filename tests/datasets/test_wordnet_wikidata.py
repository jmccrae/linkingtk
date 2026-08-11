from pathlib import Path

from linkingtk.datasets.wordnet_wikidata import (
    WordNetWikidataLanguagesDataset,
    WordNetWikidataLocationsDataset,
    WordNetWikidataOrganismsDataset,
    WordNetWikidataOrganismsHardDataset,
)

_FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "wn_wd_toy"


def _local_dataset() -> WordNetWikidataLanguagesDataset:
    return WordNetWikidataLanguagesDataset(base_url=f"file://{_FIXTURES_DIR}")


class TestEntitiesAndLabels:
    def test_loads_entities_with_ids_and_labels(self) -> None:
        entities1, entities2, _ = _local_dataset().load()

        assert [e.id for e in entities1] == [
            "http://example.org/wn/cat",
            "http://example.org/wn/dog",
            "http://example.org/wn/unicorn",
        ]
        assert [e.id for e in entities2] == [
            "http://example.org/wd/cat",
            "http://example.org/wd/dog",
            "http://example.org/wd/fish",
        ]

    def test_semicolon_separated_labels_are_split(self) -> None:
        entities1, _, _ = _local_dataset().load()

        cat = next(e for e in entities1 if e.id == "http://example.org/wn/cat")
        assert cat.labels == ["cat", "feline"]

    def test_third_column_translation_duplicate_is_ignored(self) -> None:
        # s_labels' 3rd column is a same-language duplicate of the 2nd for
        # this dataset -- only the 2nd column should end up as the label.
        entities1, _, _ = _local_dataset().load()

        dog = next(e for e in entities1 if e.id == "http://example.org/wn/dog")
        assert dog.labels == ["dog"]


class TestGroundTruth:
    def test_drops_pairs_whose_target_is_missing_from_t_labels(self) -> None:
        _, _, ground_truth = _local_dataset().load()

        # ent_ILLs links "unicorn" to "dragon", which isn't in t_labels.
        assert ground_truth == [
            ("http://example.org/wn/cat", "http://example.org/wd/cat"),
            ("http://example.org/wn/dog", "http://example.org/wd/dog"),
        ]


class TestGraphs:
    def test_triples_are_passed_through_as_uris(self) -> None:
        graph1, graph2 = _local_dataset().load_graphs()

        assert graph1 == [
            (
                "http://example.org/wn/dog",
                "http://example.org/schema#hypernym",
                "http://example.org/wn/cat",
            )
        ]
        assert graph2 == [
            (
                "http://example.org/wd/dog",
                "http://www.wikidata.org/prop/direct/P279",
                "http://example.org/wd/cat",
            )
        ]


class TestSubsetWiring:
    def test_each_subclass_points_at_its_own_subset(self) -> None:
        assert WordNetWikidataLanguagesDataset().base_url.endswith("/languages/jape")
        assert WordNetWikidataLocationsDataset().base_url.endswith("/locations/jape")
        assert WordNetWikidataOrganismsDataset().base_url.endswith("/organisms/jape")
        assert WordNetWikidataOrganismsHardDataset().base_url.endswith("/organisms_hard/jape")

    def test_base_url_override_takes_precedence(self) -> None:
        override = f"file://{_FIXTURES_DIR}"
        assert WordNetWikidataOrganismsDataset(base_url=override).base_url == override

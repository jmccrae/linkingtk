import json
from pathlib import Path

import pytest

from linkingtk.datasets.lcquad2 import Lcquad2Dataset, _entity_qids, _find_span

# One train row and one test row. Q1's label ("Mahmoud Abbas") appears
# verbatim in "question"; Q2's label ("South Park") only appears in
# "paraphrased_question" (exercises the fallback); Q3's label ("Ebola
# hemorrhagic fever") appears in neither (a real failure mode -- the
# question uses different wording, "Ebola virus" -- so it must be skipped,
# not crash); Q4 is referenced twice by the same row (deduped to one
# mention, matching real repeated-reference SPARQL like the population
# query pattern in the module docstring).
_TRAIN_ROW = {
    "uid": 1,
    "question": "Who is the head of state of Mahmoud Abbas's country?",
    "paraphrased_question": "What country is Mahmoud Abbas the head of state of?",
    "sparql_wikidata": "select distinct ?sbj where { ?sbj wdt:P35 wd:Q1 . ?sbj wdt:P31 wd:Q3 }",
}
_TEST_ROW = {
    "uid": 2,
    "question": "not the label anywhere here",
    "paraphrased_question": "Which actress voices a role on South Park?",
    "sparql_wikidata": "select ?x where { wd:Q2 wdt:P725 ?x . wd:Q4 wdt:P17 wd:Q4 }",
}

_LABELS = {
    "Q1": ("Mahmoud Abbas", "President of the State of Palestine"),
    "Q2": ("South Park", "American animated sitcom"),
    "Q3": ("Ebola hemorrhagic fever", "viral disease"),
    "Q4": ("Freedonia", "fictional country"),
}


def _fake_fetcher(qids: list[str]) -> dict[str, tuple[str, str]]:
    return {qid: _LABELS.get(qid, ("", "")) for qid in qids}


def _raising_fetcher(qids: list[str]) -> dict[str, tuple[str, str]]:
    raise AssertionError("label_fetcher should not be called in this test")


@pytest.fixture
def fixture_source(tmp_path: Path) -> tuple[str, str]:
    train_path = tmp_path / "train.json"
    test_path = tmp_path / "test.json"
    train_path.write_text(json.dumps([_TRAIN_ROW]))
    test_path.write_text(json.dumps([_TEST_ROW]))
    return str(train_path), str(test_path)


class TestEntityQids:
    def test_extracts_distinct_qids_in_order(self) -> None:
        sparql = "select ?x where { wd:Q10 wdt:P31 wd:Q2 . wd:Q10 wdt:P17 wd:Q5 }"

        assert _entity_qids(sparql) == ["Q10", "Q2", "Q5"]

    def test_does_not_match_property_ids(self) -> None:
        assert _entity_qids("select ?x where { ?x wdt:P35 wd:Q1 }") == ["Q1"]


class TestFindSpan:
    def test_case_insensitive_match(self) -> None:
        assert _find_span("Who is BARACK obama", "Barack Obama") == (7, 19)

    def test_no_match_returns_none(self) -> None:
        assert _find_span("nothing here", "Paris") is None

    def test_empty_text_or_label_returns_none(self) -> None:
        assert _find_span("", "Paris") is None
        assert _find_span("Paris is nice", "") is None


class TestLoad:
    def test_mention_resolves_from_question_field(self, fixture_source: tuple[str, str]) -> None:
        train_source, test_source = fixture_source
        mentions, _kb, _ground_truth = Lcquad2Dataset(
            train_source=train_source, test_source=test_source, label_fetcher=_fake_fetcher
        ).load()

        mahmoud = next(m for m in mentions if m.id == "lcquad2:train:1:Q1")
        assert mahmoud.labels == ["Mahmoud Abbas"]
        assert mahmoud.context == (_TRAIN_ROW["question"], 28, 41)

    def test_mention_falls_back_to_paraphrased_question(
        self, fixture_source: tuple[str, str]
    ) -> None:
        train_source, test_source = fixture_source
        mentions, _kb, _ground_truth = Lcquad2Dataset(
            train_source=train_source, test_source=test_source, label_fetcher=_fake_fetcher
        ).load()

        south_park = next(m for m in mentions if m.id == "lcquad2:test:2:Q2")
        assert south_park.context == (_TEST_ROW["paraphrased_question"], 31, 41)

    def test_unresolvable_entity_is_skipped(self, fixture_source: tuple[str, str]) -> None:
        # Q3's label ("Ebola hemorrhagic fever") appears in neither of the
        # train row's question texts -- must be dropped, not crash.
        train_source, test_source = fixture_source
        mentions, _kb, ground_truth = Lcquad2Dataset(
            train_source=train_source, test_source=test_source, label_fetcher=_fake_fetcher
        ).load()

        assert not any(m.id == "lcquad2:train:1:Q3" for m in mentions)
        assert not any(target == "Q3" for _mid, target in ground_truth)

    def test_repeated_reference_in_one_row_yields_one_mention(
        self, fixture_source: tuple[str, str]
    ) -> None:
        # Q4 is referenced twice in the test row's SPARQL but has no
        # resolvable label match either way -- kept unresolved here on
        # purpose; separately confirms _entity_qids's own dedup (tested
        # above) means it's only ever considered once, not twice.
        train_source, test_source = fixture_source
        mentions, _kb, _ground_truth = Lcquad2Dataset(
            train_source=train_source, test_source=test_source, label_fetcher=_fake_fetcher
        ).load()

        assert sum(1 for m in mentions if m.id.endswith(":Q4")) == 0

    def test_kb_includes_every_resolved_qid_with_label_and_description(
        self, fixture_source: tuple[str, str]
    ) -> None:
        train_source, test_source = fixture_source
        _mentions, kb, _ground_truth = Lcquad2Dataset(
            train_source=train_source, test_source=test_source, label_fetcher=_fake_fetcher
        ).load()

        by_id = {e.id: e for e in kb}
        assert set(by_id) == {"Q1", "Q2"}  # Q3/Q4 never resolved, so never seen
        assert by_id["Q1"].labels == ["Mahmoud Abbas"]
        assert by_id["Q1"].description == "President of the State of Palestine"

    def test_ground_truth_pairs(self, fixture_source: tuple[str, str]) -> None:
        train_source, test_source = fixture_source
        _mentions, _kb, ground_truth = Lcquad2Dataset(
            train_source=train_source, test_source=test_source, label_fetcher=_fake_fetcher
        ).load()

        assert set(ground_truth) == {
            ("lcquad2:train:1:Q1", "Q1"),
            ("lcquad2:test:2:Q2", "Q2"),
        }


class TestLoadSplits:
    def test_returns_train_test_empty_val(self, fixture_source: tuple[str, str]) -> None:
        train_source, test_source = fixture_source
        train, test, val = Lcquad2Dataset(
            train_source=train_source, test_source=test_source, label_fetcher=_fake_fetcher
        ).load_splits()

        assert train == [("lcquad2:train:1:Q1", "Q1")]
        assert test == [("lcquad2:test:2:Q2", "Q2")]
        assert val == []

    def test_still_calls_label_fetcher(self, fixture_source: tuple[str, str]) -> None:
        # Unlike AIDA-CoNLL/Zeshel's load_splits(), this one can't skip the
        # label fetch -- mention span resolution itself depends on it. A
        # fetcher that raises would fail this test, confirming it's called.
        train_source, test_source = fixture_source
        with pytest.raises(AssertionError):
            Lcquad2Dataset(
                train_source=train_source, test_source=test_source, label_fetcher=_raising_fetcher
            ).load_splits()

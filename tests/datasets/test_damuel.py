from __future__ import annotations

import json
import lzma
import tarfile
from io import BytesIO
from pathlib import Path

import pytest

from linkingtk.datasets.damuel import DamuelDataset, _resolve_language_tar_url

_TEXT = "Alice met Bob in Paris."
_TOKENS = [
    {"start": 0, "end": 5, "lemma": "Alice"},  # 0
    {"start": 6, "end": 9, "lemma": "meet"},  # 1
    {"start": 10, "end": 13, "lemma": "Bob"},  # 2
    {"start": 14, "end": 16, "lemma": "in"},  # 3
    {"start": 17, "end": 22, "lemma": "Paris"},  # 4
    {"start": 22, "end": 23, "lemma": "."},  # 5
]

# Q1 (Alice) has a Wikipedia article with four links: a real "wiki" mention
# of Bob (Q2, present in this sample -> kept), an auto-detected "label"
# mention of Paris (Q3, present -> excluded only by the default origin
# filter), a "wiki"-origin mention pointing at a qid absent from this sample
# entirely (Q999 -> dropped regardless of origin), and a real "wiki" link
# with no "qid" key at all (a NIL-style unresolved link, confirmed to occur
# on real data -> dropped like AIDA-CoNLL's NIL mentions).
_ENTITY_ALICE = {
    "qid": "Q1",
    "lang": "en",
    "label": "Alice",
    "wiki": {
        "title": "Alice",
        "text": _TEXT,
        "tokens": _TOKENS,
        "links": [
            {"start": 2, "end": 3, "origin": "wiki", "title": "Bob", "qid": "Q2"},
            {"start": 4, "end": 5, "origin": "label", "title": "Paris", "qid": "Q3"},
            {"start": 0, "end": 1, "origin": "wiki", "title": "Nobody", "qid": "Q999"},
            {"start": 1, "end": 2, "origin": "wiki", "title": "Unresolved"},
        ],
    },
}
_ENTITY_BOB = {"qid": "Q2", "lang": "en", "label": "Bob"}
_ENTITY_PARIS = {
    "qid": "Q3",
    "lang": "en",
    "label": "Paris",
    "aliases": ["City of Light"],
    "description": "capital of France",
}
_ENTITY_NOBODY_SPECIAL = {"qid": "Q4", "lang": "en", "label": "Nobody Special"}
# Real data has entities with no "label" key at all -- only aliases, or only
# a description (confirmed on real Danish DaMuEL shards).
_ENTITY_NO_LABEL = {"qid": "Q5", "lang": "en", "description": "an entity with no label"}


def _xz_member(name: str, rows: list[dict]) -> tuple[str, bytes]:
    payload = "\n".join(json.dumps(row) for row in rows).encode()
    return name, lzma.compress(payload)


def _write_tar(path: Path, members: list[tuple[str, bytes]]) -> None:
    with tarfile.open(path, mode="w") as tar:
        for name, data in members:
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, BytesIO(data))


@pytest.fixture
def fixture_source(tmp_path: Path) -> str:
    # Two parts, deliberately in this order, so max_parts=1 tests can assert
    # only the first part's entities got loaded.
    part0 = _xz_member("./damuel_1.0_en/part-00000.xz", [_ENTITY_ALICE, _ENTITY_BOB])
    part1 = _xz_member(
        "./damuel_1.0_en/part-00001.xz",
        [_ENTITY_PARIS, _ENTITY_NOBODY_SPECIAL, _ENTITY_NO_LABEL],
    )
    path = tmp_path / "damuel_1.0_en.tar"
    _write_tar(path, [part0, part1])
    return str(path)


class TestLoad:
    def test_default_origin_keeps_only_real_wiki_mentions(self, fixture_source: str) -> None:
        mentions, _kb, ground_truth = DamuelDataset(source=fixture_source, max_parts=2).load()

        assert {m.id for m in mentions} == {"damuel:Q1:10:13"}
        assert ground_truth == [("damuel:Q1:10:13", "Q2")]

    def test_mention_context_and_label(self, fixture_source: str) -> None:
        mentions, _kb, _ground_truth = DamuelDataset(source=fixture_source, max_parts=2).load()

        bob_mention = mentions[0]
        assert bob_mention.labels == ["Bob"]
        assert bob_mention.context == (_TEXT, 10, 13)

    def test_kb_includes_entities_without_wiki_pages(self, fixture_source: str) -> None:
        _mentions, kb, _ground_truth = DamuelDataset(source=fixture_source, max_parts=2).load()

        by_id = {e.id: e for e in kb}
        assert set(by_id) == {"Q1", "Q2", "Q3", "Q4", "Q5"}
        assert by_id["Q2"].labels == ["Bob"]
        assert by_id["Q2"].description is None

    def test_kb_entity_with_no_label_key_falls_back_to_empty_labels(
        self, fixture_source: str
    ) -> None:
        _mentions, kb, _ground_truth = DamuelDataset(source=fixture_source, max_parts=2).load()

        no_label = next(e for e in kb if e.id == "Q5")
        assert no_label.labels == []
        assert no_label.description == "an entity with no label"

    def test_kb_entity_carries_aliases_and_description(self, fixture_source: str) -> None:
        _mentions, kb, _ground_truth = DamuelDataset(source=fixture_source, max_parts=2).load()

        paris = next(e for e in kb if e.id == "Q3")
        assert paris.labels == ["Paris", "City of Light"]
        assert paris.description == "capital of France"

    def test_link_with_no_qid_key_is_dropped(self, fixture_source: str) -> None:
        # A real "wiki"-origin link with no "qid" key at all (unresolved,
        # like a NIL mention) must not crash and must not appear.
        mentions, _kb, _ground_truth = DamuelDataset(source=fixture_source, max_parts=2).load()

        assert "damuel:Q1:6:9" not in {m.id for m in mentions}

    def test_target_missing_from_sample_drops_the_mention(self, fixture_source: str) -> None:
        # Q999 is never present in this fixture at all, even though the
        # Alice->Q999 link's origin ("wiki") would otherwise qualify.
        mentions, _kb, ground_truth = DamuelDataset(source=fixture_source, max_parts=2).load()

        assert "Q999" not in {tid for _mid, tid in ground_truth}
        assert not any(m.id.startswith("damuel:Q1:0:5") for m in mentions)

    def test_widening_mention_origins_includes_more_mentions(self, fixture_source: str) -> None:
        mentions, _kb, ground_truth = DamuelDataset(
            source=fixture_source, max_parts=2, mention_origins=("wiki", "label")
        ).load()

        assert {m.id for m in mentions} == {"damuel:Q1:10:13", "damuel:Q1:17:22"}
        assert ("damuel:Q1:17:22", "Q3") in ground_truth

    def test_max_parts_limits_how_many_shards_are_read(self, fixture_source: str) -> None:
        _mentions, kb, _ground_truth = DamuelDataset(source=fixture_source, max_parts=1).load()

        # Only part-00000's entities (Q1, Q2) should have been read.
        assert {e.id for e in kb} == {"Q1", "Q2"}


class TestResolveLanguageTarUrl:
    def test_resolves_bitstream_content_url_for_language(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        responses = [
            json.dumps(
                {
                    "_embedded": {
                        "bundles": [
                            {"name": "LICENSE", "_links": {"bitstreams": {"href": "unused"}}},
                            {
                                "name": "ORIGINAL",
                                "_links": {"bitstreams": {"href": "https://example/bitstreams"}},
                            },
                        ]
                    }
                }
            ).encode(),
            json.dumps(
                {
                    "_embedded": {
                        "bitstreams": [
                            {
                                "name": "damuel_1.0_af.tar",
                                "_links": {"content": {"href": "https://example/af/content"}},
                            },
                            {
                                "name": "damuel_1.0_en.tar",
                                "_links": {"content": {"href": "https://example/en/content"}},
                            },
                        ]
                    }
                }
            ).encode(),
        ]
        calls = iter(responses)
        monkeypatch.setattr(
            "linkingtk.datasets.damuel.fetch_cached",
            lambda url, cache_dir=None: next(calls),
        )

        url = _resolve_language_tar_url("en")

        assert url == "https://example/en/content"

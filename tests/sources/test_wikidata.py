from __future__ import annotations

import gzip
import io
import json
import pickle
import sys
import types
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from linkingtk.core import Entity
from linkingtk.sources.vector_index import VectorIndexEntitySource
from linkingtk.sources.wikidata import (
    _USER_AGENT,
    WikidataDumpEntities,
    WikidataEntitySource,
    _instance_of_qids,
    _is_url,
)


class _FakeResponse:
    def __init__(self, payload: dict[str, Any], status_ok: bool = True) -> None:
        self._payload = payload
        self.status_ok = status_ok

    def raise_for_status(self) -> None:
        if not self.status_ok:
            raise RuntimeError("HTTP error")

    def json(self) -> dict[str, Any]:
        return self._payload


class _FakeSession:
    def __init__(self) -> None:
        # Mirrors requests.Session()'s own default User-Agent -- see
        # test_wikipedia.py's _FakeSession for why this matters.
        self.headers: dict[str, str] = {"User-Agent": "python-requests/2.99.0"}
        self.calls: list[dict[str, Any]] = []
        self._search_response: dict[str, Any] = {"search": []}
        self._get_response: dict[str, Any] = {"entities": {}}

    def get(self, url: str, params: dict[str, Any], timeout: float) -> _FakeResponse:
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        if params.get("action") == "wbsearchentities":
            return _FakeResponse(self._search_response)
        return _FakeResponse(self._get_response)


def _fake_requests_module() -> types.ModuleType:
    module = types.ModuleType("requests")
    module.Session = _FakeSession  # type: ignore[attr-defined]
    return module


@pytest.fixture(autouse=True)
def fake_requests(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    module = _fake_requests_module()
    monkeypatch.setitem(sys.modules, "requests", module)
    return module


class TestSearch:
    def test_maps_hits_to_entities(self) -> None:
        session = _FakeSession()
        session._search_response = {
            "search": [
                {"id": "Q90", "label": "Paris", "description": "capital of France"},
                {"id": "Q830149", "label": "Paris", "description": "city in Texas"},
            ]
        }
        source = WikidataEntitySource(session=session)

        results = source.search("Paris")

        assert [e.id for e in results] == ["Q90", "Q830149"]
        assert results[0].labels == ["Paris"]
        assert results[0].description == "capital of France"

    def test_no_matches_returns_empty_list(self) -> None:
        source = WikidataEntitySource(session=_FakeSession())

        assert source.search("nonexistent") == []

    def test_top_k_forwarded_as_limit(self) -> None:
        session = _FakeSession()
        source = WikidataEntitySource(session=session)

        source.search("Paris", top_k=3)

        assert session.calls[0]["params"]["limit"] == 3

    def test_hit_without_label_is_skipped(self) -> None:
        session = _FakeSession()
        session._search_response = {"search": [{"id": "Q1", "description": "no label field"}]}
        source = WikidataEntitySource(session=session)

        assert source.search("x") == []


class TestGet:
    def test_returns_matching_entity(self) -> None:
        session = _FakeSession()
        session._get_response = {
            "entities": {
                "Q30": {
                    "id": "Q30",
                    "labels": {"en": {"language": "en", "value": "United States"}},
                    "descriptions": {"en": {"language": "en", "value": "country in North America"}},
                    "claims": {},
                }
            }
        }
        source = WikidataEntitySource(session=session)

        entity = source.get("Q30")

        assert entity is not None
        assert entity.id == "Q30"
        assert entity.labels == ["United States"]
        assert entity.description == "country in North America"
        assert entity.properties == {}

    def test_surfaces_instance_of_into_properties(self) -> None:
        session = _FakeSession()
        session._get_response = {
            "entities": {
                "Q5": {
                    "id": "Q5",
                    "labels": {"en": {"language": "en", "value": "human"}},
                    "descriptions": {},
                    "claims": {
                        "P31": [
                            {
                                "mainsnak": {
                                    "datavalue": {"value": {"id": "Q123"}},
                                }
                            },
                            {
                                "mainsnak": {
                                    "datavalue": {"value": {"id": "Q456"}},
                                }
                            },
                        ]
                    },
                }
            }
        }
        source = WikidataEntitySource(session=session)

        entity = source.get("Q5")

        assert entity is not None
        assert entity.properties == {"instance_of": "Q123 Q456"}

    def test_missing_entity_marker_returns_none(self) -> None:
        session = _FakeSession()
        session._get_response = {"entities": {"Q999999999": {"id": "Q999999999", "missing": ""}}}
        source = WikidataEntitySource(session=session)

        assert source.get("Q999999999") is None

    def test_top_level_api_error_returns_none(self) -> None:
        # Wikidata reports an out-of-range QID as a top-level error rather
        # than an "entities" payload with a "missing" marker -- confirmed
        # directly against the live API.
        session = _FakeSession()
        session._get_response = {"error": {"code": "no-such-entity"}}
        source = WikidataEntitySource(session=session)

        assert source.get("Q999999999999") is None

    def test_entity_without_requested_language_label_returns_none(self) -> None:
        session = _FakeSession()
        session._get_response = {
            "entities": {"Q42": {"id": "Q42", "labels": {}, "descriptions": {}, "claims": {}}}
        }
        source = WikidataEntitySource(session=session)

        assert source.get("Q42") is None


class TestInstanceOfQids:
    def test_extracts_qids_from_p31_claims(self) -> None:
        claims = {
            "P31": [
                {"mainsnak": {"datavalue": {"value": {"id": "Q5"}}}},
                {"mainsnak": {"datavalue": {"value": {"id": "Q123"}}}},
            ]
        }
        assert _instance_of_qids(claims) == ["Q5", "Q123"]

    def test_no_p31_claim_returns_empty_list(self) -> None:
        assert _instance_of_qids({}) == []

    def test_non_value_snak_is_skipped(self) -> None:
        claims = {"P31": [{"mainsnak": {"snaktype": "novalue"}}]}
        assert _instance_of_qids(claims) == []


def _dump_text(rows: list[dict[str, Any]]) -> str:
    """Line-delimited-JSON-array text, the shape of Wikidata's real dump."""
    lines = ["["]
    for i, row in enumerate(rows):
        suffix = "," if i < len(rows) - 1 else ""
        lines.append(json.dumps(row) + suffix)
    lines.append("]")
    return "\n".join(lines) + "\n"


_DUMP_ROWS = [
    {
        "type": "item",
        "id": "Q90",
        "labels": {"en": {"language": "en", "value": "Paris"}},
        "descriptions": {"en": {"language": "en", "value": "capital of France"}},
        "aliases": {},
        "claims": {},
    },
    {
        "type": "item",
        "id": "Q5",
        "labels": {"en": {"language": "en", "value": "human"}},
        "descriptions": {},
        "aliases": {"en": [{"language": "en", "value": "Homo sapiens"}]},
        "claims": {"P31": [{"mainsnak": {"datavalue": {"value": {"id": "Q123"}}}}]},
    },
    {
        "type": "property",
        "id": "P31",
        "labels": {"en": {"language": "en", "value": "instance of"}},
    },
    {
        "type": "item",
        "id": "Q999",
        "labels": {},
        "descriptions": {},
        "aliases": {},
        "claims": {},
    },
]


class TestWikidataDumpEntities:
    def test_reads_items_and_skips_non_items_and_unlabeled(self, tmp_path: Path) -> None:
        dump_path = tmp_path / "dump.json"
        dump_path.write_text(_dump_text(_DUMP_ROWS), encoding="utf-8")

        entities = list(WikidataDumpEntities(dump_path, progress=False))

        # P31 (a property, not an item) and Q999 (no "en" label) are both skipped.
        assert [e.id for e in entities] == ["Q90", "Q5"]

    def test_aliases_are_folded_into_labels(self, tmp_path: Path) -> None:
        dump_path = tmp_path / "dump.json"
        dump_path.write_text(_dump_text(_DUMP_ROWS), encoding="utf-8")

        entities = {e.id: e for e in WikidataDumpEntities(dump_path, progress=False)}

        assert entities["Q5"].labels == ["human", "Homo sapiens"]
        assert entities["Q5"].properties == {"instance_of": "Q123"}

    def test_gzip_dump_is_read_transparently(self, tmp_path: Path) -> None:
        dump_path = tmp_path / "dump.json.gz"
        with gzip.open(dump_path, "wt", encoding="utf-8") as f:
            f.write(_dump_text(_DUMP_ROWS))

        entities = list(WikidataDumpEntities(dump_path, progress=False))

        assert [e.id for e in entities] == ["Q90", "Q5"]

    def test_limit_stops_early(self, tmp_path: Path) -> None:
        dump_path = tmp_path / "dump.json"
        dump_path.write_text(_dump_text(_DUMP_ROWS), encoding="utf-8")

        entities = list(WikidataDumpEntities(dump_path, limit=1, progress=False))

        assert [e.id for e in entities] == ["Q90"]

    def test_reiterable_yields_same_result_twice(self, tmp_path: Path) -> None:
        dump_path = tmp_path / "dump.json"
        dump_path.write_text(_dump_text(_DUMP_ROWS), encoding="utf-8")
        dump = WikidataDumpEntities(dump_path, progress=False)

        assert [e.id for e in dump] == [e.id for e in dump]

    def test_malformed_json_line_is_skipped(self, tmp_path: Path) -> None:
        dump_path = tmp_path / "dump.json"
        dump_path.write_text(
            "[\n" + json.dumps(_DUMP_ROWS[0]) + ",\nnot valid json,\n"
            + json.dumps(_DUMP_ROWS[1]) + "\n]\n",
            encoding="utf-8",
        )

        entities = list(WikidataDumpEntities(dump_path, progress=False))

        assert [e.id for e in entities] == ["Q90", "Q5"]

    def test_builds_a_vector_index_end_to_end(
        self, tmp_path: Path, fake_faiss: types.ModuleType
    ) -> None:
        dump_path = tmp_path / "dump.json"
        dump_path.write_text(_dump_text(_DUMP_ROWS), encoding="utf-8")

        index = VectorIndexEntitySource.build(
            WikidataDumpEntities(dump_path, progress=False),
            _FakeEmbedder(),
            tmp_path / "idx",
            reduced_dim=2,
        )

        assert [e.id for e in index.search("Paris", top_k=1)] == ["Q90"]


class TestIsUrl:
    def test_http_url_is_detected(self) -> None:
        assert _is_url("http://example.com/dump.json") is True

    def test_https_url_is_detected(self) -> None:
        assert _is_url("https://example.com/dump.json.gz") is True

    def test_local_path_is_not_a_url(self) -> None:
        assert _is_url("/tmp/dump.json") is False
        assert _is_url("dump.json") is False


class TestWikidataDumpEntitiesUrl:
    def test_reads_from_plain_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        content = _dump_text(_DUMP_ROWS).encode("utf-8")
        requests: list[urllib.request.Request] = []

        def fake_urlopen(request: urllib.request.Request, *args: Any, **kwargs: Any) -> io.BytesIO:
            requests.append(request)
            return io.BytesIO(content)

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

        entities = list(WikidataDumpEntities("http://example.com/dump.json", progress=False))

        assert [e.id for e in entities] == ["Q90", "Q5"]
        assert requests[0].full_url == "http://example.com/dump.json"
        headers = {k.lower(): v for k, v in requests[0].header_items()}
        assert headers["user-agent"] == _USER_AGENT

    def test_reads_from_gzip_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        compressed = gzip.compress(_dump_text(_DUMP_ROWS).encode("utf-8"))

        def fake_urlopen(request: urllib.request.Request, *args: Any, **kwargs: Any) -> io.BytesIO:
            return io.BytesIO(compressed)

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

        entities = list(
            WikidataDumpEntities("https://example.com/dump.json.gz", progress=False)
        )

        assert [e.id for e in entities] == ["Q90", "Q5"]

    def test_reiterating_url_source_requests_again(self, monkeypatch: pytest.MonkeyPatch) -> None:
        content = _dump_text(_DUMP_ROWS).encode("utf-8")
        call_count = 0

        def fake_urlopen(request: urllib.request.Request, *args: Any, **kwargs: Any) -> io.BytesIO:
            nonlocal call_count
            call_count += 1
            return io.BytesIO(content)

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
        dump = WikidataDumpEntities("http://example.com/dump.json", progress=False)

        list(dump)
        list(dump)

        assert call_count == 2


class TestConstruction:
    def test_self_created_session_gets_descriptive_user_agent(self) -> None:
        source = WikidataEntitySource()

        assert "linkingtk" in source._session.headers["User-Agent"]

    def test_caller_supplied_session_headers_are_untouched(self) -> None:
        session = _FakeSession()
        session.headers["User-Agent"] = "custom-agent/1.0"

        WikidataEntitySource(session=session)

        assert session.headers["User-Agent"] == "custom-agent/1.0"


class _FakeEmbedder:
    def __init__(self, dim: int = 26) -> None:
        self.dim = dim

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for char in text.lower():
                if "a" <= char <= "z":
                    vectors[row, ord(char) - ord("a")] += 1.0
        return vectors


class _FakeIndexFlatIP:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.vectors: list[np.ndarray] = []

    def add(self, vectors: np.ndarray) -> None:
        self.vectors.extend(vectors)

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        n_queries = queries.shape[0]
        if not self.vectors:
            return (
                np.zeros((n_queries, k), dtype=np.float32),
                -np.ones((n_queries, k), dtype=np.int64),
            )
        matrix = np.stack(self.vectors)
        scores = queries @ matrix.T
        k_eff = min(k, matrix.shape[0])
        order = np.argsort(-scores, axis=1)[:, :k_eff]
        top_scores = np.take_along_axis(scores, order, axis=1)
        if k_eff < k:
            pad = k - k_eff
            order = np.concatenate([order, -np.ones((n_queries, pad), dtype=np.int64)], axis=1)
            top_scores = np.concatenate(
                [top_scores, np.zeros((n_queries, pad), dtype=np.float32)], axis=1
            )
        return top_scores.astype(np.float32), order.astype(np.int64)


def _write_index(index: _FakeIndexFlatIP, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump({"dim": index.dim, "vectors": index.vectors}, f)


def _read_index(path: str) -> _FakeIndexFlatIP:
    with open(path, "rb") as f:
        data = pickle.load(f)
    index = _FakeIndexFlatIP(data["dim"])
    index.vectors = data["vectors"]
    return index


@pytest.fixture
def fake_faiss(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    module = types.ModuleType("faiss")
    module.IndexFlatIP = _FakeIndexFlatIP  # type: ignore[attr-defined]
    module.write_index = _write_index  # type: ignore[attr-defined]
    module.read_index = _read_index  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "faiss", module)
    return module


class TestVectorIndexBackend:
    def test_search_delegates_to_vector_index_without_any_network_call(
        self, tmp_path: Path, fake_faiss: types.ModuleType
    ) -> None:
        entities = [Entity(id="Q90", labels=["Paris"], description="capital of France")]
        index = VectorIndexEntitySource.build(
            entities, _FakeEmbedder(), tmp_path / "idx", reduced_dim=None
        )
        session = _FakeSession()
        source = WikidataEntitySource(session=session, vector_index=index)

        results = source.search("Paris")

        assert [e.id for e in results] == ["Q90"]
        assert session.calls == []

    def test_get_prefers_vector_index_over_live_call(
        self, tmp_path: Path, fake_faiss: types.ModuleType
    ) -> None:
        entities = [Entity(id="Q90", labels=["Paris"], description="capital of France")]
        index = VectorIndexEntitySource.build(
            entities, _FakeEmbedder(), tmp_path / "idx", reduced_dim=None
        )
        session = _FakeSession()
        source = WikidataEntitySource(session=session, vector_index=index)

        entity = source.get("Q90")

        assert entity is not None
        assert entity.description == "capital of France"
        assert session.calls == []

    def test_get_falls_back_to_live_call_on_index_miss(
        self, tmp_path: Path, fake_faiss: types.ModuleType
    ) -> None:
        index = VectorIndexEntitySource.build(
            [], _FakeEmbedder(), tmp_path / "idx", reduced_dim=None
        )
        session = _FakeSession()
        session._get_response = {
            "entities": {
                "Q30": {
                    "id": "Q30",
                    "labels": {"en": {"language": "en", "value": "United States"}},
                    "descriptions": {},
                    "claims": {},
                }
            }
        }
        source = WikidataEntitySource(session=session, vector_index=index)

        entity = source.get("Q30")

        assert entity is not None
        assert entity.labels == ["United States"]
        assert len(session.calls) == 1

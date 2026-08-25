from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from linkingtk.sources.wikipedia import WikipediaEntitySource, _clean_snippet


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
        # Mirrors requests.Session()'s own behavior of pre-populating a
        # "python-requests/*" User-Agent -- this is what makes
        # `headers.setdefault(...)` the wrong way to apply our own (it
        # never fires, since the key already exists).
        self.headers: dict[str, str] = {"User-Agent": "python-requests/2.99.0"}
        self.calls: list[dict[str, Any]] = []
        self._search_response: dict[str, Any] = {"query": {"search": []}}
        self._get_response: dict[str, Any] = {"query": {"pages": {}}}

    def get(self, url: str, params: dict[str, Any], timeout: float) -> _FakeResponse:
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        if params.get("list") == "search":
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
            "query": {
                "search": [
                    {
                        "title": "Python (programming language)",
                        "pageid": 1,
                        "snippet": "<span class=\"searchmatch\">Python</span> is a language",
                    },
                    {"title": "Python (genus)", "pageid": 2, "snippet": "a genus of snakes"},
                ]
            }
        }
        source = WikipediaEntitySource(session=session)

        results = source.search("Python")

        assert [e.id for e in results] == ["Python (programming language)", "Python (genus)"]
        assert results[0].labels == ["Python (programming language)"]
        assert results[0].description == "Python is a language"

    def test_no_matches_returns_empty_list(self) -> None:
        session = _FakeSession()
        source = WikipediaEntitySource(session=session)

        assert source.search("nonexistent") == []

    def test_top_k_forwarded_as_srlimit(self) -> None:
        session = _FakeSession()
        source = WikipediaEntitySource(session=session)

        source.search("Python", top_k=3)

        assert session.calls[0]["params"]["srlimit"] == 3

    def test_html_entities_in_snippet_are_unescaped(self) -> None:
        session = _FakeSession()
        session._search_response = {
            "query": {
                "search": [
                    {"title": "AT&T", "pageid": 1, "snippet": "AT&amp;T is a <b>company</b>"}
                ]
            }
        }
        source = WikipediaEntitySource(session=session)

        results = source.search("AT&T")

        assert results[0].description == "AT&T is a company"


class TestGet:
    def test_returns_matching_entity(self) -> None:
        session = _FakeSession()
        session._get_response = {
            "query": {
                "pages": {
                    "1": {
                        "pageid": 1,
                        "title": "Albert Einstein",
                        "extract": "Albert Einstein was a theoretical physicist.",
                    }
                }
            }
        }
        source = WikipediaEntitySource(session=session)

        entity = source.get("Albert Einstein")

        assert entity is not None
        assert entity.id == "Albert Einstein"
        assert entity.description == "Albert Einstein was a theoretical physicist."

    def test_missing_page_returns_none(self) -> None:
        session = _FakeSession()
        session._get_response = {
            "query": {"pages": {"-1": {"title": "Nonexistent Page Xyz", "missing": ""}}}
        }
        source = WikipediaEntitySource(session=session)

        assert source.get("Nonexistent Page Xyz") is None


class TestConstruction:
    def test_self_created_session_gets_descriptive_user_agent(self) -> None:
        # requests.Session() pre-populates its own "python-requests/*"
        # User-Agent, which the live MediaWiki API rejects outright -- a
        # naive `headers.setdefault(...)` would never override it.
        source = WikipediaEntitySource()

        assert "linkingtk" in source._session.headers["User-Agent"]

    def test_caller_supplied_session_headers_are_untouched(self) -> None:
        session = _FakeSession()
        session.headers["User-Agent"] = "custom-agent/1.0"

        WikipediaEntitySource(session=session)

        assert session.headers["User-Agent"] == "custom-agent/1.0"

    def test_lang_selects_api_endpoint(self) -> None:
        session = _FakeSession()
        source = WikipediaEntitySource(lang="fr", session=session)

        source.search("chat")

        assert session.calls[0]["url"] == "https://fr.wikipedia.org/w/api.php"


class TestCleanSnippet:
    def test_strips_tags_and_unescapes_entities(self) -> None:
        assert _clean_snippet('<span class="searchmatch">A</span> &amp; B') == "A & B"

    def test_plain_text_is_unchanged(self) -> None:
        assert _clean_snippet("plain text") == "plain text"

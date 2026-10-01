from __future__ import annotations

import json
import os
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest

from linkingtk.algorithms.kev import KevClient, KevError, KevLinker, _windowed_context
from linkingtk.blocking.base import BlockingStrategy
from linkingtk.core.entity import Entity


class _AllPairs(BlockingStrategy):
    """Lets every pair through, used to isolate KevLinker's own behavior."""

    def candidate_pairs(
        self, dataset1: list[Entity], dataset2: list[Entity]
    ) -> list[tuple[Entity, Entity]]:
        return [(e1, e2) for e1 in dataset1 for e2 in dataset2]


class _FakeKevClient(KevClient):
    """Records every request; P(yes) is looked up by a keyword in the question text."""

    def __init__(self, p_yes_by_keyword: dict[str, float], fail: bool = False) -> None:
        super().__init__()
        self.p_yes_by_keyword = p_yes_by_keyword
        self.fail = fail
        self.calls: list[dict[str, Any]] = []

    def ask_noul(
        self,
        state: str,
        questions: dict[str, str],
        criteria: dict[str, str] | None = None,
    ) -> dict[str, float]:
        self.calls.append({"state": state, "questions": questions, "criteria": criteria})
        if self.fail:
            raise KevError("boom")
        return {
            qid: next((p for kw, p in self.p_yes_by_keyword.items() if kw in text), 0.0)
            for qid, text in questions.items()
        }

    def ask_choice(
        self, state: str, instructions: str, options: dict[str, str]
    ) -> dict[str, float]:
        self.calls.append({"state": state, "instructions": instructions, "options": options})
        if self.fail:
            raise KevError("boom")
        return {
            name: next((p for kw, p in self.p_yes_by_keyword.items() if kw in desc), 0.0)
            for name, desc in options.items()
        }


def _mention(text: str, surface: str, id: str = "m1", lemma: str | None = None) -> Entity:
    start = text.index(surface)
    return Entity(id=id, labels=[lemma or surface], context=(text, start, start + len(surface)))


class TestKevLinker:
    def test_ranks_candidates_by_p_yes(self) -> None:
        mention = _mention("He moved to Paris last year.", "Paris")
        kb = [
            Entity(id="Paris_Hilton", labels=["Paris Hilton"], description="socialite"),
            Entity(id="Paris", labels=["Paris"], description="capital of France"),
        ]
        client = _FakeKevClient({"capital": 0.9, "socialite": 0.2})

        [result] = KevLinker(client, task="el").link([mention], kb, blocking=_AllPairs())

        assert result.target_id == "Paris"
        assert result.alternatives == ["Paris_Hilton"]

    def test_one_state_with_one_question_per_candidate(self) -> None:
        mention = _mention("He moved to Paris last year.", "Paris")
        kb = [Entity(id=f"e{i}", labels=[f"E{i}"], description=f"d{i}") for i in range(3)]
        client = _FakeKevClient({})

        KevLinker(client, task="el").link([mention], kb, blocking=_AllPairs())

        [call] = client.calls
        assert 'Mention: "Paris"' in call["state"]
        assert "[[Paris]]" in call["state"]
        assert list(call["questions"]) == ["c0", "c1", "c2"]
        assert "E1: d1" in call["questions"]["c1"]
        assert call["criteria"] is not None and set(call["criteria"]) == {"true", "false"}

    def test_chunks_candidates_across_requests(self) -> None:
        mention = _mention("a bass guitar", "bass")
        kb = [Entity(id=f"e{i}", labels=["bass"], description=f"sense {i}") for i in range(5)]
        client = _FakeKevClient({"sense 3": 0.8})

        linker = KevLinker(client, task="wsd", questions_per_request=2)
        scores = linker.score_candidates([mention], kb, blocking=_AllPairs())

        assert [len(call["questions"]) for call in client.calls] == [2, 2, 1]
        assert dict(scores["m1"]) == {"e0": 0.0, "e1": 0.0, "e2": 0.0, "e3": 0.8, "e4": 0.0}

    def test_wsd_state_includes_lemma(self) -> None:
        mention = _mention("She caught a fish.", "caught", lemma="catch")
        sense = Entity(id="s1", labels=["catch"], description="take hold of")
        client = _FakeKevClient({})

        KevLinker(client, task="wsd").link([mention], [sense], blocking=_AllPairs())

        assert "Lemma: catch" in client.calls[0]["state"]
        assert 'word "caught"' in client.calls[0]["questions"]["c0"]

    def test_failed_request_scores_zero_instead_of_raising(self) -> None:
        mention = _mention("He moved to Paris.", "Paris")
        kb = [Entity(id="Paris", labels=["Paris"])]

        scores = KevLinker(_FakeKevClient({}, fail=True)).score_candidates(
            [mention], kb, blocking=_AllPairs()
        )

        assert scores == {"m1": [("Paris", 0.0)]}

    def test_choice_asks_one_question_over_all_candidates(self) -> None:
        mention = _mention("a bass guitar", "bass")
        kb = [Entity(id=f"e{i}", labels=["bass"], description=f"sense {i}") for i in range(5)]
        client = _FakeKevClient({"sense 3": 0.7, "sense 1": 0.2})

        linker = KevLinker(client, task="wsd", question_type="choice", questions_per_request=2)
        [result] = linker.link([mention], kb, blocking=_AllPairs())

        [call] = client.calls
        assert 'word "bass"' in call["instructions"]
        assert call["options"]["option 4"] == "bass: sense 3"
        assert len(call["options"]) == 5
        assert result.target_id == "e3"
        assert result.alternatives[0] == "e1"

    def test_choice_failure_scores_zero(self) -> None:
        mention = _mention("He moved to Paris.", "Paris")
        kb = [Entity(id="Paris", labels=["Paris"])]

        linker = KevLinker(_FakeKevClient({}, fail=True), question_type="choice")

        assert linker.score_candidates([mention], kb, blocking=_AllPairs()) == {
            "m1": [("Paris", 0.0)]
        }

    def test_rejects_non_positive_chunk_size(self) -> None:
        with pytest.raises(ValueError):
            KevLinker(_FakeKevClient({}), questions_per_request=0)


class TestWindowedContext:
    def test_trims_to_window_and_brackets_span(self) -> None:
        text = "one two three four five TARGET six seven eight nine ten"
        mention = _mention(text, "TARGET")

        assert _windowed_context(mention, 2) == "... four five [[TARGET]] six seven ..."

    def test_no_ellipsis_when_context_fits(self) -> None:
        mention = _mention("a TARGET b", "TARGET")

        assert _windowed_context(mention, 5) == "a [[TARGET]] b"


class _StubHandler(BaseHTTPRequestHandler):
    requests: list[dict[str, Any]] = []

    def do_POST(self) -> None:  # noqa: N802 -- BaseHTTPRequestHandler's API
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        type(self).requests.append(
            {"path": self.path, "body": body, "auth": self.headers.get("authorization")}
        )
        if body["state"] == "bad":
            self.send_response(422)
            self.end_headers()
            self.wfile.write(b'{"detail": "invalid"}')
            return
        answers = {
            qid: {"type": "choice", "probabilities": {"a": 0.6, "b": 0.4}}
            if q["type"] == "choice"
            else {"type": "noul", "noul": 0.25}
            for qid, q in body["questions"].items()
        }
        payload = json.dumps({"answers": answers, "usage": {}, "latency_ms": 1}).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: Any) -> None:
        pass


@pytest.fixture
def stub_server() -> Iterator[str]:
    _StubHandler.requests = []
    server = HTTPServer(("127.0.0.1", 0), _StubHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


class TestKevClient:
    def test_request_shape_and_response_parsing(self, stub_server: str) -> None:
        client = KevClient(stub_server, api_key="secret")

        p_yes = client.ask_noul("some text", {"a": "Is it?"}, {"true": "yes", "false": "no"})

        assert p_yes == {"a": 0.25}
        [request] = _StubHandler.requests
        assert request["path"] == "/v1/systemone"
        assert request["auth"] == "Bearer secret"
        assert request["body"] == {
            "state": "some text",
            "model": "kev-latest",
            "questions": {
                "a": {
                    "type": "noul",
                    "instructions": "Is it?",
                    "criteria": {"true": "yes", "false": "no"},
                }
            },
        }

    def test_choice_request_and_response(self, stub_server: str) -> None:
        p = KevClient(stub_server).ask_choice("text", "Which?", {"a": "first", "b": "second"})

        assert p == {"a": 0.6, "b": 0.4}
        [request] = _StubHandler.requests
        assert request["body"]["questions"] == {
            "q": {
                "type": "choice",
                "instructions": "Which?",
                "criteria": {"a": "first", "b": "second"},
            }
        }

    def test_http_error_raises_kev_error(self, stub_server: str) -> None:
        with pytest.raises(KevError, match="422"):
            KevClient(stub_server).ask_noul("bad", {"a": "Is it?"})

    def test_unreachable_server_raises_kev_error(self) -> None:
        with pytest.raises(KevError, match="kev.serve"):
            KevClient("http://127.0.0.1:1", timeout=2).health()


@pytest.mark.skipif("KEV_BASE_URL" not in os.environ, reason="needs a running kev.serve")
def test_live_server_prefers_correct_candidate() -> None:
    mention = _mention("The capital of France, Paris, hosted the Olympics.", "Paris")
    kb = [
        Entity(id="Paris", labels=["Paris"], description="capital city of France"),
        Entity(id="Paris_Hilton", labels=["Paris Hilton"], description="American socialite"),
    ]
    linker = KevLinker(KevClient(os.environ["KEV_BASE_URL"]), task="el")

    [result] = linker.link([mention], kb, blocking=_AllPairs())

    assert result.target_id == "Paris"

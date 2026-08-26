from __future__ import annotations

from typing import Any

from linkingtk.algorithms.llm import EA_PROMPT, EL_PROMPT, WSA_PROMPT, WSD_PROMPT, LlmBaseLinker
from linkingtk.algorithms.matching import GreedyMatcher, OptimalMatcher
from linkingtk.blocking.base import BlockingStrategy
from linkingtk.blocking.exact import ExactMatch
from linkingtk.core.entity import Entity
from linkingtk.core.source import EntitySource
from linkingtk.llm.client import LlmClient, LlmMessage


class _AllPairs(BlockingStrategy):
    """Lets every pair through, used to isolate LlmBaseLinker's own behavior."""

    def candidate_pairs(
        self, dataset1: list[Entity], dataset2: list[Entity]
    ) -> list[tuple[Entity, Entity]]:
        return [(e1, e2) for e1 in dataset1 for e2 in dataset2]


class _FakeSource(EntitySource):
    def __init__(self, entities: list[Entity]) -> None:
        self._entities = entities

    def search(self, query: str, top_k: int = 10) -> list[Entity]:
        matches = [entity for entity in self._entities if query in entity.labels]
        return matches[:top_k]

    def get(self, entity_id: str) -> Entity | None:
        for entity in self._entities:
            if entity.id == entity_id:
                return entity
        return None


class _FakeLlmClient(LlmClient):
    """Records every call and returns a canned response per source entity.

    `responses` maps a source entity id to the raw dict `complete_structured`
    should return for that call (looked up by the id embedded in the prompt's
    "Source entity" block, since the fake doesn't otherwise know which source
    entity a given call is for).
    """

    def __init__(self, responses: dict[str, dict[str, Any]]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    def complete(self, messages: list[LlmMessage], **kwargs: Any) -> str:
        raise NotImplementedError

    def complete_structured(
        self, messages: list[LlmMessage], *, schema: dict[str, Any], **kwargs: Any
    ) -> dict[str, Any]:
        self.calls.append({"messages": messages, "schema": schema, **kwargs})
        user_content = next(m.content for m in messages if m.role == "user")
        for source_id, response in self.responses.items():
            if f"(id={source_id})" in user_content:
                return response
        raise AssertionError(f"No canned response matches prompt:\n{user_content}")


def _entity(id: str, description: str | None = None, context: str | None = None) -> Entity:
    return Entity(id=id, labels=[id], description=description, context=context)


class TestPromptContent:
    def test_prompt_includes_source_and_every_candidate_id(self) -> None:
        source = _entity("mention1", context="the mention1 occurred here")
        candidates = [_entity("cand1", description="first"), _entity("cand2", description="second")]
        client = _FakeLlmClient(
            {"mention1": {"rankings": [{"candidate_id": "cand1", "score": 0.9}]}}
        )
        linker = LlmBaseLinker(client=client, task="el")

        linker.link([source], candidates, blocking=_AllPairs())

        [call] = client.calls
        user_content = next(m.content for m in call["messages"] if m.role == "user")
        assert "mention1" in user_content
        assert "the mention1 occurred here" in user_content
        assert "id=cand1" in user_content
        assert "id=cand2" in user_content
        assert "first" in user_content
        assert "second" in user_content
        assert call["schema"]["required"] == ["rankings"]

    def test_task_selects_instruction(self) -> None:
        source = _entity("s1")
        candidates = [_entity("c1")]
        for task, expected in [
            ("ea", EA_PROMPT.instruction),
            ("el", EL_PROMPT.instruction),
            ("wsd", WSD_PROMPT.instruction),
            ("wsa", WSA_PROMPT.instruction),
        ]:
            client = _FakeLlmClient({"s1": {"rankings": [{"candidate_id": "c1", "score": 1.0}]}})
            linker = LlmBaseLinker(client=client, task=task)  # type: ignore[arg-type]

            linker.link([source], candidates, blocking=_AllPairs())

            [call] = client.calls
            system_content = next(m.content for m in call["messages"] if m.role == "system")
            assert system_content == expected


class TestScoring:
    def test_ranks_by_returned_score(self) -> None:
        source = _entity("s1")
        candidates = [_entity("c1"), _entity("c2")]
        client = _FakeLlmClient(
            {
                "s1": {
                    "rankings": [
                        {"candidate_id": "c2", "score": 0.9},
                        {"candidate_id": "c1", "score": 0.1},
                    ]
                }
            }
        )
        linker = LlmBaseLinker(client=client)

        results = linker.link([source], candidates, blocking=_AllPairs())

        assert len(results) == 1
        assert results[0].target_id == "c2"
        assert results[0].alternatives == ["c1"]

    def test_unmentioned_candidate_gets_score_zero_not_dropped(self) -> None:
        source = _entity("s1")
        candidates = [_entity("c1"), _entity("c2")]
        client = _FakeLlmClient({"s1": {"rankings": [{"candidate_id": "c1", "score": 0.5}]}})
        linker = LlmBaseLinker(client=client)

        results = linker.link([source], candidates, blocking=_AllPairs())

        assert results[0].target_id == "c1"
        assert results[0].alternatives == ["c2"]

    def test_hallucinated_candidate_id_is_ignored(self) -> None:
        source = _entity("s1")
        candidates = [_entity("c1")]
        client = _FakeLlmClient(
            {
                "s1": {
                    "rankings": [
                        {"candidate_id": "c1", "score": 0.5},
                        {"candidate_id": "not-a-real-candidate", "score": 0.99},
                    ]
                }
            }
        )
        linker = LlmBaseLinker(client=client)

        results = linker.link([source], candidates, blocking=_AllPairs())

        assert len(results) == 1
        assert results[0].target_id == "c1"

    def test_source_with_no_candidates_from_blocking_gets_no_result_and_no_llm_call(
        self,
    ) -> None:
        source = _entity("s1")
        client = _FakeLlmClient({})
        linker = LlmBaseLinker(client=client)

        results = linker.link([source], [], blocking=ExactMatch())

        assert results == []
        assert client.calls == []

    def test_llm_call_failure_is_skipped_not_raised(self) -> None:
        source = _entity("s1")
        candidates = [_entity("c1")]

        class _FailingClient(_FakeLlmClient):
            def complete_structured(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
                raise RuntimeError("boom")

        linker = LlmBaseLinker(client=_FailingClient({}))

        results = linker.link([source], candidates, blocking=_AllPairs())

        assert results == []


class TestMatchingIntegration:
    def test_works_with_optimal_matcher(self) -> None:
        sources = [_entity("s1"), _entity("s2")]
        candidates = [_entity("c1"), _entity("c2")]
        client = _FakeLlmClient(
            {
                "s1": {
                    "rankings": [
                        {"candidate_id": "c1", "score": 0.9},
                        {"candidate_id": "c2", "score": 0.8},
                    ]
                },
                "s2": {
                    "rankings": [
                        {"candidate_id": "c1", "score": 0.7},
                        {"candidate_id": "c2", "score": 0.6},
                    ]
                },
            }
        )
        linker = LlmBaseLinker(client=client, matching=OptimalMatcher())

        results = linker.link(sources, candidates, blocking=_AllPairs())

        target_ids = {r.target_id for r in results}
        assert target_ids == {"c1", "c2"}

    def test_default_matcher_is_greedy(self) -> None:
        linker = LlmBaseLinker(client=_FakeLlmClient({}))
        assert isinstance(linker.matching, GreedyMatcher)


class TestEntitySourceEndToEnd:
    def test_links_against_an_entity_source_via_exact_match(self) -> None:
        source_entity = Entity(id="e1", labels=["big cat"])
        source_store = _FakeSource(
            [
                Entity(id="t1", labels=["big cat"], description="a large feline"),
                Entity(id="t2", labels=["small dog"], description="a small canine"),
            ]
        )
        client = _FakeLlmClient({"e1": {"rankings": [{"candidate_id": "t1", "score": 0.95}]}})
        linker = LlmBaseLinker(client=client, task="ea")

        results = linker.link([source_entity], source_store, blocking=ExactMatch())

        assert len(results) == 1
        assert results[0].target_id == "t1"
        assert results[0].score == 0.95

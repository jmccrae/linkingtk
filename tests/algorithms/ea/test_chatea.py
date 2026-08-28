from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from linkingtk.algorithms.ea._chatea_context import NeighborIndex
from linkingtk.algorithms.ea._chatea_prompts import resolve_dimensions
from linkingtk.algorithms.ea.chatea import ChatEALinker
from linkingtk.core.entity import Entity
from linkingtk.core.source import EntitySource
from linkingtk.exceptions import LinkingTKError
from linkingtk.llm.client import LlmClient, LlmMessage
from linkingtk.matchers import GreedyMatcher


class _FakeEmbeddingLinker:
    """Minimal fake `base_linker` -- toy vectors, not a real fitted linker."""

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors

    def source_embedding(self, entity_id: str) -> npt.NDArray[np.floating[Any]]:
        return np.array(self.vectors[entity_id], dtype=float)

    def target_embedding(self, entity_id: str) -> npt.NDArray[np.floating[Any]]:
        return np.array(self.vectors[entity_id], dtype=float)


class _FakeChatEAClient(LlmClient):
    """Dispatches canned `complete_structured` responses by schema shape.

    - A schema with a `good_enough` property -> the next queued rethink
      response.
    - A schema with a `description` property -> the description response
      whose entity name appears in the prompt.
    - Otherwise (a scoring call) -> the next queued response for whichever
      candidate id's `[Candidate Entity] = Entity(id=...` block appears in
      the prompt.
    """

    def __init__(
        self,
        scoring_responses: dict[str, list[dict[str, Any]]] | None = None,
        description_responses: dict[str, dict[str, Any]] | None = None,
        rethink_responses: list[dict[str, Any]] | None = None,
    ) -> None:
        self.scoring_responses = {k: list(v) for k, v in (scoring_responses or {}).items()}
        self.description_responses = description_responses or {}
        self.rethink_responses = list(rethink_responses or [])
        self.calls: list[dict[str, Any]] = []

    def complete(self, messages: list[LlmMessage], **kwargs: Any) -> str:
        raise NotImplementedError

    def complete_structured(
        self, messages: list[LlmMessage], *, schema: dict[str, Any], **kwargs: Any
    ) -> dict[str, Any]:
        self.calls.append({"messages": messages, "schema": schema, **kwargs})
        properties = schema.get("properties", {})
        user_content = next((m.content for m in reversed(messages) if m.role == "user"), "")

        if "good_enough" in properties:
            if self.rethink_responses:
                return self.rethink_responses.pop(0)
            return {"good_enough": False}

        if "description" in properties:
            for name, response in self.description_responses.items():
                if f"Entity: {name}" in user_content:
                    return response
            raise AssertionError(f"No canned description matches prompt:\n{user_content}")

        for candidate_id, queue in self.scoring_responses.items():
            if f"[Candidate Entity] = Entity(id={candidate_id!r}" in user_content:
                return queue.pop(0) if len(queue) > 1 else queue[0]
        raise AssertionError(f"No canned scoring response matches prompt:\n{user_content}")


def _entity(entity_id: str, name: str) -> Entity:
    return Entity(id=entity_id, labels=[name])


class TestNeighborIndex:
    def test_caps_neighbors_to_neigh_num(self) -> None:
        triples = [("a", "r1", "b"), ("a", "r2", "c"), ("a", "r3", "d")]
        index = NeighborIndex(
            triples,
            None,
            {"a": "A", "b": "B", "c": "C", "d": "D"},
            {"r1": "rel1", "r2": "rel2", "r3": "rel3"},
            neigh_num=2,
        )
        assert len(index.neighbors("a")) == 2

    def test_relation_name_falls_back_to_raw_id(self) -> None:
        index = NeighborIndex([("a", "unknown_rel", "b")], None, {"a": "A", "b": "B"}, {}, 10)
        [triple] = index.neighbors("a")
        assert triple[1] == "unknown_rel"

    def test_entity_with_no_triples_has_no_neighbors(self) -> None:
        index = NeighborIndex([("a", "r", "b")], None, {"a": "A", "b": "B"}, {}, 10)
        assert index.neighbors("isolated") == []


class TestResolveDimensions:
    def test_time_excluded_when_disabled(self) -> None:
        dims = resolve_dimensions(use_name=True, use_desc=False, use_struct=False, use_time=False)
        assert dims.keys == ("name",)
        assert dims.weights["name"] == 1.0

    def test_time_included_when_enabled(self) -> None:
        dims = resolve_dimensions(use_name=True, use_desc=False, use_struct=False, use_time=True)
        assert dims.keys == ("name", "time")

    def test_desc_excluded_without_name(self) -> None:
        dims = resolve_dimensions(use_name=False, use_desc=True, use_struct=False, use_time=False)
        assert "desc" not in dims.keys


class TestConfidenceShortcut:
    def test_skips_llm_entirely_when_base_scores_are_far_apart(self) -> None:
        base_linker = _FakeEmbeddingLinker({"s1": [1.0, 0.0], "c1": [1.0, 0.0], "c2": [0.0, 1.0]})
        client = _FakeChatEAClient()
        linker = ChatEALinker(base_linker=base_linker, client=client, threshold=0.5, use_desc=False)

        results = linker.link(
            [_entity("s1", "S1")], [_entity("c1", "C1"), _entity("c2", "C2")], graph=[]
        )

        assert client.calls == []
        assert len(results) == 1
        assert results[0].target_id == "c1"
        assert results[0].score == 1.0


class TestIterativeWidening:
    def test_widens_from_top1_to_top10_window_and_stops_early(self) -> None:
        base_linker = _FakeEmbeddingLinker(
            {
                "s1": [1.0, 0.0],
                "c1": [1.0, 0.1],
                "c2": [1.0, 0.2],
                "c3": [1.0, 0.3],
            }
        )
        client = _FakeChatEAClient(
            scoring_responses={
                "c1": [{"name_score": 2}, {"name_score": 3}],
                "c2": [{"name_score": 3}],
                "c3": [{"name_score": 5}],
            }
        )
        linker = ChatEALinker(
            base_linker=base_linker,
            client=client,
            threshold=0.5,
            use_desc=False,
            use_struct=False,
        )

        results = linker.link(
            [_entity("s1", "S1")],
            [_entity("c1", "C1"), _entity("c2", "C2"), _entity("c3", "C3")],
            graph=[],
        )

        scoring_calls = [c for c in client.calls if "name_score" in c["schema"]["properties"]]
        assert len(scoring_calls) == 4  # 1 (window=1) + 3 (window=10, all re-scored)

        assert len(results) == 1
        assert results[0].target_id == "c3"
        assert results[0].score == 1.0
        assert set(results[0].alternatives) == {"c1", "c2"}

    def test_ambiguous_score_triggers_rethink_call(self) -> None:
        base_linker = _FakeEmbeddingLinker({"s1": [1.0, 0.0], "c1": [1.0, 0.1]})
        client = _FakeChatEAClient(
            scoring_responses={"c1": [{"name_score": 3}]},
            rethink_responses=[{"good_enough": True}],
        )
        linker = ChatEALinker(
            base_linker=base_linker,
            client=client,
            threshold=0.5,
            use_desc=False,
            use_struct=False,
        )

        results = linker.link([_entity("s1", "S1")], [_entity("c1", "C1")], graph=[])

        rethink_calls = [c for c in client.calls if "good_enough" in c["schema"]["properties"]]
        scoring_calls = [c for c in client.calls if "name_score" in c["schema"]["properties"]]
        assert len(scoring_calls) == 1
        assert len(rethink_calls) == 1
        assert results[0].target_id == "c1"
        assert results[0].score == 0.5


class TestDescriptionGeneration:
    def test_generated_once_per_needed_entity_and_cached(self) -> None:
        base_linker = _FakeEmbeddingLinker({"s1": [1.0, 0.0], "c1": [1.0, 0.05], "c2": [1.0, 0.1]})
        client = _FakeChatEAClient(
            scoring_responses={
                "c1": [{"name_score": 5, "desc_score": 5}],
                "c2": [{"name_score": 1, "desc_score": 1}],
            },
            description_responses={
                "S1": {"description": "source desc"},
                "C1": {"description": "candidate one desc"},
                "C2": {"description": "candidate two desc"},
            },
        )
        linker = ChatEALinker(
            base_linker=base_linker, client=client, threshold=0.5, use_struct=False
        )

        linker.link([_entity("s1", "S1")], [_entity("c1", "C1"), _entity("c2", "C2")], graph=[])

        description_calls = [c for c in client.calls if "description" in c["schema"]["properties"]]
        assert len(description_calls) == 3


class TestMatching:
    def test_default_matcher_is_greedy(self) -> None:
        linker = ChatEALinker(base_linker=_FakeEmbeddingLinker({}), client=_FakeChatEAClient())
        assert isinstance(linker.matching, GreedyMatcher)


class TestEntitySourceRejected:
    def test_raises_when_dataset2_is_an_entity_source(self) -> None:
        class _EmptySource(EntitySource):
            def search(self, query: str, top_k: int = 10) -> list[Entity]:
                return []

            def get(self, entity_id: str) -> Entity | None:
                return None

        linker = ChatEALinker(base_linker=_FakeEmbeddingLinker({}), client=_FakeChatEAClient())
        with pytest.raises(LinkingTKError, match="EntitySource"):
            linker.link([_entity("s1", "S1")], _EmptySource(), graph=[])

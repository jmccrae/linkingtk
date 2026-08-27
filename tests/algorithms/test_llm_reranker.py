from __future__ import annotations

from typing import Any

from linkingtk.algorithms.base import DEFAULT_BLOCKING
from linkingtk.algorithms.llm_reranker import LlmRerankerLinker
from linkingtk.algorithms.matching import GreedyMatcher
from linkingtk.blocking.base import BlockingStrategy
from linkingtk.core.entity import Entity
from linkingtk.core.source import EntitySource
from linkingtk.llm.client import LlmClient, LlmMessage


class _AllPairs(BlockingStrategy):
    """Lets every pair through, used to isolate LlmRerankerLinker's own behavior."""

    def candidate_pairs(
        self, dataset1: list[Entity], dataset2: list[Entity]
    ) -> list[tuple[Entity, Entity]]:
        return [(e1, e2) for e1 in dataset1 for e2 in dataset2]


class _FakeBaseLinker:
    """A `CandidateScorer` returning canned scores, ignoring the real base method."""

    def __init__(self, scores_by_source: dict[str, list[tuple[str, float]]]) -> None:
        self.scores_by_source = scores_by_source
        self.calls = 0

    def score_candidates(
        self,
        dataset1: list[Entity],
        dataset2: list[Entity] | EntitySource,
        blocking: BlockingStrategy = DEFAULT_BLOCKING,
    ) -> dict[str, list[tuple[str, float]]]:
        self.calls += 1
        return self.scores_by_source


class _FakeLlmClient(LlmClient):
    """Records every call and returns a canned response per source entity."""

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


def _entity(id: str) -> Entity:
    return Entity(id=id, labels=[id])


class TestTopKTruncation:
    def test_only_top_k_candidates_are_sent_to_the_llm(self) -> None:
        source = _entity("s1")
        candidates = [_entity("c1"), _entity("c2"), _entity("c3"), _entity("c4")]
        base_linker = _FakeBaseLinker(
            {"s1": [("c1", 0.4), ("c2", 0.39), ("c3", 0.38), ("c4", 0.37)]}
        )
        client = _FakeLlmClient({"s1": {"rankings": [{"candidate_id": "c1", "score": 0.9}]}})
        linker = LlmRerankerLinker(base_linker=base_linker, client=client, top_k=2, threshold=1.0)

        linker.link([source], candidates, blocking=_AllPairs())

        [call] = client.calls
        user_content = next(m.content for m in call["messages"] if m.role == "user")
        assert "id=c1" in user_content
        assert "id=c2" in user_content
        assert "id=c3" not in user_content
        assert "id=c4" not in user_content


class TestConfidenceShortcut:
    def test_default_threshold_is_disabled_even_with_a_huge_score_gap(self) -> None:
        """Regression guard: base-linker score scale is arbitrary (e.g. a
        cross-encoder's unbounded raw logit margin), so an enabled-by-
        default absolute threshold would silently skip the LLM almost
        always for such linkers -- confirmed directly against #23's own
        WSD benchmark (GlossBertLinker), where a default of 0.5 skipped
        28/30 sampled entities."""
        source = _entity("s1")
        candidates = [_entity("c1"), _entity("c2")]
        base_linker = _FakeBaseLinker({"s1": [("c1", 100.0), ("c2", 0.1)]})
        client = _FakeLlmClient({"s1": {"rankings": [{"candidate_id": "c2", "score": 0.9}]}})
        linker = LlmRerankerLinker(base_linker=base_linker, client=client)

        results = linker.link([source], candidates, blocking=_AllPairs())

        assert len(client.calls) == 1
        assert results[0].target_id == "c2"

    def test_large_score_gap_skips_the_llm_call(self) -> None:
        source = _entity("s1")
        candidates = [_entity("c1"), _entity("c2")]
        base_linker = _FakeBaseLinker({"s1": [("c1", 0.9), ("c2", 0.1)]})
        client = _FakeLlmClient({})
        linker = LlmRerankerLinker(base_linker=base_linker, client=client, threshold=0.5)

        results = linker.link([source], candidates, blocking=_AllPairs())

        assert client.calls == []
        assert results[0].target_id == "c1"

    def test_small_score_gap_calls_the_llm(self) -> None:
        source = _entity("s1")
        candidates = [_entity("c1"), _entity("c2")]
        base_linker = _FakeBaseLinker({"s1": [("c1", 0.51), ("c2", 0.5)]})
        client = _FakeLlmClient({"s1": {"rankings": [{"candidate_id": "c2", "score": 0.9}]}})
        linker = LlmRerankerLinker(base_linker=base_linker, client=client, threshold=0.5)

        results = linker.link([source], candidates, blocking=_AllPairs())

        assert len(client.calls) == 1
        assert results[0].target_id == "c2"


class TestResponseMerging:
    def test_ranks_by_llm_returned_score(self) -> None:
        source = _entity("s1")
        candidates = [_entity("c1"), _entity("c2")]
        base_linker = _FakeBaseLinker({"s1": [("c1", 0.5), ("c2", 0.49)]})
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
        linker = LlmRerankerLinker(base_linker=base_linker, client=client, threshold=0.5)

        results = linker.link([source], candidates, blocking=_AllPairs())

        assert results[0].target_id == "c2"
        assert results[0].alternatives == ["c1"]

    def test_hallucinated_candidate_id_is_ignored(self) -> None:
        source = _entity("s1")
        candidates = [_entity("c1")]
        base_linker = _FakeBaseLinker({"s1": [("c1", 0.5)]})
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
        linker = LlmRerankerLinker(base_linker=base_linker, client=client, threshold=1.0)

        results = linker.link([source], candidates, blocking=_AllPairs())

        assert len(results) == 1
        assert results[0].target_id == "c1"


class TestLlmFailureFallback:
    def test_llm_call_failure_falls_back_to_base_ranking(self) -> None:
        source = _entity("s1")
        candidates = [_entity("c1"), _entity("c2")]
        base_linker = _FakeBaseLinker({"s1": [("c1", 0.5), ("c2", 0.49)]})

        class _FailingClient(_FakeLlmClient):
            def complete_structured(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
                raise RuntimeError("boom")

        linker = LlmRerankerLinker(
            base_linker=base_linker, client=_FailingClient({}), threshold=1.0
        )

        results = linker.link([source], candidates, blocking=_AllPairs())

        assert len(results) == 1
        assert results[0].target_id == "c1"


class TestMatchingIntegration:
    def test_no_base_candidates_gives_no_results_and_no_llm_call(self) -> None:
        source = _entity("s1")
        base_linker = _FakeBaseLinker({})
        client = _FakeLlmClient({})
        linker = LlmRerankerLinker(base_linker=base_linker, client=client)

        results = linker.link([source], [], blocking=_AllPairs())

        assert results == []
        assert client.calls == []

    def test_default_matcher_is_greedy(self) -> None:
        linker = LlmRerankerLinker(base_linker=_FakeBaseLinker({}), client=_FakeLlmClient({}))
        assert isinstance(linker.matching, GreedyMatcher)

"""General-purpose linker that prompts an LLM to decide/score links.

Usable across Entity Alignment, Entity Linking, Word Sense Disambiguation
and Word Sense Alignment -- selected via `task`, the same way
[StringSimilarityLinker][linkingtk.algorithms.string_similarity.StringSimilarityLinker]
is one general-purpose class rather than four task-specific subclasses.
Talks to an LLM through [LlmClient][linkingtk.llm.client.LlmClient] (#60),
so this module never imports `openai`/`anthropic` itself.
"""

from __future__ import annotations

import logging
from collections import defaultdict

from linkingtk.algorithms._llm_prompting import (
    _PROMPTS,
    _RESULT_SCHEMA,
    EA_PROMPT,
    EL_PROMPT,
    WSA_PROMPT,
    WSD_PROMPT,
    PromptTemplate,
    Task,
    _build_prompt,
    _resolve_candidate_id,
)
from linkingtk.algorithms.base import DEFAULT_BLOCKING, BaseLinker
from linkingtk.blocking.base import BlockingStrategy
from linkingtk.core.entity import Entity
from linkingtk.core.result import AlignmentResult
from linkingtk.core.source import EntitySource
from linkingtk.llm.client import LlmClient
from linkingtk.matchers import DEFAULT_MATCHER, Matcher
from linkingtk.utils.graph import Graph

logger = logging.getLogger("linkingtk")

__all__ = [
    "EA_PROMPT",
    "EL_PROMPT",
    "WSA_PROMPT",
    "WSD_PROMPT",
    "LlmBaseLinker",
    "PromptTemplate",
    "Task",
]


class LlmBaseLinker(BaseLinker):
    """Prompts an LLM to score/rank each source entity's blocked candidates.

    Unlike [StringSimilarityLinker][linkingtk.algorithms.string_similarity.StringSimilarityLinker],
    which scores each candidate pair independently, this linker makes one
    LLM call per *source* entity, showing all of its blocked candidates at
    once via [LlmClient.complete_structured][linkingtk.llm.client.LlmClient.complete_structured]
    -- cheaper than one call per pair, and gives the model real comparative
    context instead of siloed yes/no judgments.

    Args:
        client: An already-constructed [LlmClient][linkingtk.llm.client.LlmClient]
            (e.g. from [create_client][linkingtk.llm.client.create_client]).
            This class never imports `openai`/`anthropic` itself.
        task: Which [PromptTemplate][linkingtk.algorithms.llm.PromptTemplate]
            instruction to use -- `"ea"`, `"el"`, `"wsd"`, or `"wsa"`.
        matching: Strategy used to resolve scored candidates into final
            links, same as `StringSimilarityLinker`'s `matching` argument.
        max_tokens: Forwarded to `client.complete_structured` per call.
            Default 2048, not `complete_structured`'s own 1024 default --
            measured directly against a real WSD benchmark
            (`examples/llm_benchmark.py`) with wide (`top_k=50`) blocking:
            a source entity with many real candidates can need more than
            1024 tokens just to rank all of them, and 1024 produced
            truncated, unparseable JSON responses for exactly those
            higher-candidate-count entities (a systematic gap, not
            isolated noise). Widen further if using an even larger
            candidate width than 50.

    Note:
        `graph` is accepted for interface compliance but not used, same as
        `StringSimilarityLinker`. LLM calls are made sequentially, one per
        source entity with at least one blocked candidate.
    """

    def __init__(
        self,
        client: LlmClient,
        task: Task = "el",
        matching: Matcher = DEFAULT_MATCHER,
        max_tokens: int = 2048,
    ) -> None:
        self.client = client
        self.task = task
        self.matching = matching
        self.max_tokens = max_tokens

    def link(
        self,
        dataset1: list[Entity],
        dataset2: list[Entity] | EntitySource,
        graph: Graph = None,
        blocking: BlockingStrategy = DEFAULT_BLOCKING,
    ) -> list[AlignmentResult]:
        instruction = _PROMPTS[self.task].instruction
        pairs = blocking.candidate_pairs(dataset1, dataset2)

        sources_by_id = {entity.id: entity for entity in dataset1}
        candidates_by_id: dict[str, list[Entity]] = defaultdict(list)
        for entity1, entity2 in pairs:
            candidates_by_id[entity1.id].append(entity2)

        candidates_by_source: dict[str, list[tuple[str, float]]] = {}
        for source_id, candidates in candidates_by_id.items():
            messages = _build_prompt(instruction, sources_by_id[source_id], candidates)
            try:
                response = self.client.complete_structured(
                    messages, schema=_RESULT_SCHEMA, max_tokens=self.max_tokens
                )
            except Exception:
                logger.warning(
                    "LlmBaseLinker: LLM call failed for source entity %r, skipping",
                    source_id,
                    exc_info=True,
                )
                continue

            valid_ids = {candidate.id for candidate in candidates}
            scores: dict[str, float] = dict.fromkeys(valid_ids, 0.0)
            for ranking in response.get("rankings", []):
                raw_id, score = ranking.get("candidate_id"), ranking.get("score")
                candidate_id = _resolve_candidate_id(raw_id, valid_ids)
                if candidate_id is None:
                    logger.warning(
                        "LlmBaseLinker: ignoring hallucinated candidate id %r for source %r",
                        raw_id,
                        source_id,
                    )
                    continue
                scores[candidate_id] = float(score)

            if scores:
                candidates_by_source[source_id] = list(scores.items())

        return self.matching.match(candidates_by_source)

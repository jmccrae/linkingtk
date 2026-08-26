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
from dataclasses import dataclass
from typing import Any, Literal

from linkingtk.algorithms.base import DEFAULT_BLOCKING, BaseLinker
from linkingtk.algorithms.matching import DEFAULT_MATCHER, Matcher
from linkingtk.blocking.base import BlockingStrategy
from linkingtk.core.entity import Entity, context_text, description_text, label_texts
from linkingtk.core.result import AlignmentResult
from linkingtk.core.source import EntitySource
from linkingtk.llm.client import LlmClient, LlmMessage
from linkingtk.utils.graph import Graph

logger = logging.getLogger("linkingtk")

Task = Literal["ea", "el", "wsd", "wsa"]


@dataclass
class PromptTemplate:
    """A task's instruction to the LLM, prepended to a source entity + its candidates.

    Attributes:
        instruction: The system-message text describing the task and the
            expected judgment -- the only part that varies per task. Entity
            rendering itself (`_format_entity`) is shared across every task,
            since DESIGN.md's own per-task Entity-shape differences
            (`labels`/`description`/`context` populated differently per
            task) already fall out of "render whichever fields are
            non-empty" without per-task branching.
    """

    instruction: str


EA_PROMPT = PromptTemplate(
    instruction=(
        "You are aligning entities between two knowledge graphs. Given a "
        "source entity and a numbered list of candidate target entities, "
        "decide which candidates refer to the same real-world thing as the "
        "source entity."
    )
)
EL_PROMPT = PromptTemplate(
    instruction=(
        "You are linking a named-entity mention in text to entries in a "
        "knowledge base. Given the mention (with its surrounding context) "
        "and a numbered list of candidate knowledge-base entries, decide "
        "which candidates the mention refers to."
    )
)
WSD_PROMPT = PromptTemplate(
    instruction=(
        "You are disambiguating the sense of a word used in context. Given "
        "the word's context and a numbered list of candidate dictionary "
        "senses, decide which candidates match the intended sense."
    )
)
WSA_PROMPT = PromptTemplate(
    instruction=(
        "You are aligning senses between two dictionaries. Given a source "
        "dictionary sense and a numbered list of candidate senses from "
        "another dictionary, decide which candidates mean the same thing "
        "as the source sense."
    )
)

_PROMPTS: dict[Task, PromptTemplate] = {
    "ea": EA_PROMPT,
    "el": EL_PROMPT,
    "wsd": WSD_PROMPT,
    "wsa": WSA_PROMPT,
}

_RESULT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "rankings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "candidate_id": {"type": "string"},
                    "score": {"type": "number"},
                },
                "required": ["candidate_id", "score"],
            },
        },
    },
    "required": ["rankings"],
}


def _format_entity(entity: Entity) -> str:
    lines = [f"Labels: {', '.join(label_texts(entity))}"]
    if description_text(entity):
        lines.append(f"Description: {description_text(entity)}")
    if context_text(entity):
        lines.append(f"Context: {context_text(entity)}")
    return "\n".join(lines)


def _build_prompt(instruction: str, source: Entity, candidates: list[Entity]) -> list[LlmMessage]:
    candidate_blocks = "\n\n".join(
        f"Candidate {i} (id={candidate.id}):\n{_format_entity(candidate)}"
        for i, candidate in enumerate(candidates, start=1)
    )
    user_content = (
        f"Source entity (id={source.id}):\n{_format_entity(source)}\n\n"
        f"Candidates:\n{candidate_blocks}\n\n"
        "Return every candidate id you were given, each with a score in "
        "[0, 1] (higher = more likely correct), best match first."
    )
    return [
        LlmMessage(role="system", content=instruction),
        LlmMessage(role="user", content=user_content),
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
        max_tokens: int = 1024,
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
                candidate_id, score = ranking.get("candidate_id"), ranking.get("score")
                if candidate_id not in valid_ids:
                    logger.warning(
                        "LlmBaseLinker: ignoring hallucinated candidate id %r for source %r",
                        candidate_id,
                        source_id,
                    )
                    continue
                scores[candidate_id] = float(score)

            if scores:
                candidates_by_source[source_id] = list(scores.items())

        return self.matching.match(candidates_by_source)

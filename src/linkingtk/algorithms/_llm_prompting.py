"""Shared LLM-prompting building blocks for [LlmBaseLinker][linkingtk.algorithms.llm.LlmBaseLinker]
and [LlmRerankerLinker][linkingtk.algorithms.llm_reranker.LlmRerankerLinker].

Extracted from `llm.py` (#21) so #23's reranker reuses the exact same
per-task instructions, entity rendering, structured-output schema, and
hallucinated-id recovery, rather than a parallel copy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from linkingtk.core.entity import Entity, context_text, description_text, label_texts
from linkingtk.llm.client import LlmMessage

Task = Literal["ea", "el", "wsd", "wsa"]


@dataclass
class PromptTemplate:
    """A task's instruction to the LLM, prepended to a source entity + its candidates.

    Attributes:
        instruction: The system-message text describing the task and the
            expected judgment -- the only part that varies per task. Entity
            rendering itself (`_format_entity`) is shared across every task,
            since the per-task Entity-shape differences
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


def _resolve_candidate_id(raw_id: Any, valid_ids: set[str]) -> str | None:
    """Recover a real candidate id from a model's possibly-mangled response.

    Some models echo the prompt's own `"(id=...)"` rendering verbatim
    instead of extracting the bare id -- observed directly against real
    Ollama output (e.g. `"id=Atlanta_Falcons"` instead of
    `"Atlanta_Falcons"`). Returns `None` if `raw_id` doesn't resolve to a
    real candidate even after stripping that, meaning it's genuinely
    hallucinated rather than just reformatted.
    """
    if not isinstance(raw_id, str):
        return None
    if raw_id in valid_ids:
        return raw_id
    stripped = raw_id.removeprefix("id=").strip("'\"")
    return stripped if stripped in valid_ids else None

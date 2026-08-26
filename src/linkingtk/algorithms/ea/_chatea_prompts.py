"""Prompt/scoring-schema construction for
[ChatEALinker][linkingtk.algorithms.ea.chatea.ChatEALinker].

Ports `main_ChatEA.py`'s `init_fields`/`init_prompt` (the system prompt --
task framing plus a worked reasoning example) and `generate_prompt`/
`ask_for_accuracy` (the per-candidate reasoning prompt and the
"rethinking" confirmation prompt) -- adapted to score via
[LlmClient.complete_structured][linkingtk.llm.client.LlmClient.complete_structured]
against a fixed JSON schema instead of the reference's free-text +
regex-parsed response (`get_score`), matching
[LlmBaseLinker][linkingtk.algorithms.llm.LlmBaseLinker]'s own existing
convention in this repo.

The reference's `use_code`/class-definition text-assembly branching
(`init_prompt`'s combinatorial "how do I describe `self.tuples` in
pseudo-code depending on which flags are off" logic) is simplified here
to one plain description of whichever info types are enabled -- prompt-
text cosmetics for ablations this port isn't running. The actual
algorithm (candidate windowing, weighted scoring, thresholds,
rethinking) lives in
[_chatea_reasoning][linkingtk.algorithms.ea._chatea_reasoning] and is
ported faithfully.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from linkingtk.algorithms.ea._chatea_context import EntityContext, format_triple
from linkingtk.llm.client import LlmMessage

_BASE_WEIGHTS: dict[str, float] = {"name": 0.4, "desc": 0.3, "struct": 0.2, "time": 0.1}
_FIELD_LABELS: dict[str, str] = {
    "name": "NAME SIMILARITY",
    "desc": "PROBABILITY OF DESCRIPTION POINTING SAME ENTITY",
    "struct": "STRUCTURE SIMILARITY",
    "time": "TIME SIMILARITY",
}
_DIMENSION_ORDER = ("name", "desc", "struct", "time")

RETHINK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"good_enough": {"type": "boolean"}},
    "required": ["good_enough"],
}


@dataclass
class ScoringDimensions:
    """Which of the four score dimensions are active, and their (renormalized) weights."""

    keys: tuple[str, ...]
    weights: dict[str, float]


def resolve_dimensions(
    use_name: bool, use_desc: bool, use_struct: bool, use_time: bool
) -> ScoringDimensions:
    """Picks active dimensions and renormalizes `_BASE_WEIGHTS` over them.

    Mirrors `init_fields`'s `scale_weights = [w/sum(weights) for w in weights]`.
    The description dimension requires `use_name` too, same gating as the
    reference (`if use_name and use_desc and use_code`) -- "does this
    description point at the same entity as that name" presupposes a name
    was shown at all.
    """
    enabled = {
        "name": use_name,
        "desc": use_desc and use_name,
        "struct": use_struct,
        "time": use_time,
    }
    keys = tuple(k for k in _DIMENSION_ORDER if enabled[k])
    total = sum(_BASE_WEIGHTS[k] for k in keys) or 1.0
    weights = {k: _BASE_WEIGHTS[k] / total for k in keys}
    return ScoringDimensions(keys=keys, weights=weights)


def scoring_schema(dims: ScoringDimensions) -> dict[str, Any]:
    properties = {f"{k}_score": {"type": "integer", "minimum": 1, "maximum": 5} for k in dims.keys}
    return {"type": "object", "properties": properties, "required": list(properties.keys())}


def weighted_score(response: dict[str, Any], dims: ScoringDimensions) -> float:
    """Combines a scoring response into one score in ``[0, 1]``.

    Ports `eval_alignment_for_evaluate`'s ``score += weights[j] * (sims[j] - 1) * 0.25``.
    """
    total = 0.0
    for key in dims.keys:
        raw = response.get(f"{key}_score", 1)
        clipped = min(5, max(1, int(raw)))
        total += dims.weights[key] * (clipped - 1) * 0.25
    return round(total, 4)


def summarize_response(response: dict[str, Any], dims: ScoringDimensions) -> str:
    """A short text summary of a scoring response, used as the assistant's
    turn in the running chat history (mirrors the reference's own
    ``simple_response``, keeping history compact across many calls)."""
    parts = [f"{_FIELD_LABELS[k]} = {response.get(f'{k}_score', 1)} out of 5" for k in dims.keys]
    return ", ".join(parts) + "."


def build_system_prompt(dims: ScoringDimensions) -> str:
    """Ports `init_prompt`: task framing plus one worked reasoning example."""
    used_info = []
    if "name" in dims.keys:
        used_info.append("name")
    if "desc" in dims.keys:
        used_info.append("description")
    if "struct" in dims.keys:
        used_info.append("structure (neighboring knowledge tuples)")
    if "time" in dims.keys:
        used_info.append("temporal information")
    used_info.append("your own knowledge")

    example_steps = []
    if "name" in dims.keys:
        example_steps.append(
            "NAME SIMILARITY: 'Fudan University' and 'Fudan_University' are almost the "
            "same, so NAME SIMILARITY = 5 (very high)"
        )
    if "desc" in dims.keys:
        example_steps.append(
            "PROBABILITY OF DESCRIPTION POINTING SAME ENTITY: both descriptions describe "
            "a university in Shanghai founded in 1905, so this = 5 (very high)"
        )
    if "struct" in dims.keys:
        example_steps.append(
            "STRUCTURE SIMILARITY: 'China' is a common neighbor of both entities, so "
            "STRUCTURE SIMILARITY = 3 (medium)"
        )
    if "time" in dims.keys:
        example_steps.append(
            "TIME SIMILARITY: the candidate entity has no matching time information, so "
            "just assume TIME SIMILARITY = 2 (low)"
        )
    example = (
        "Worked example: Main Entity = Entity(id='8535', name='Fudan University'), "
        "Candidate Entity = Entity(id='24431', name='Fudan_University'). "
        "Reasoning step by step: " + "; ".join(example_steps) + "."
    )

    output_fields = ", ".join(f"[{_FIELD_LABELS[k]}]" for k in dims.keys)

    return (
        "You are a helpful assistant, helping align or match entities between two "
        f"knowledge graphs using {', '.join(used_info)}.\n\n"
        f"{example}\n\n"
        f"For every candidate entity, score {output_fields}, each an integer from "
        "1 (very low) to 5 (very high)."
    )


def build_candidate_prompt(
    main: EntityContext,
    candidate: EntityContext,
    candidate_names: list[str],
    ranked_so_far: list[tuple[str, float]],
    dims: ScoringDimensions,
) -> tuple[LlmMessage, str]:
    """Ports `generate_prompt`. Returns ``(full_message, simple_prompt_text)`` --
    `simple_prompt_text` is the compact form used for the running chat history
    (mirrors the reference's own ``prompt, simple_prompt`` pair)."""

    def render(entity: EntityContext) -> str:
        parts = [f"id={entity.entity_id!r}"]
        if "name" in dims.keys:
            parts.append(f"name={entity.name!r}")
        if "desc" in dims.keys:
            parts.append(f"description={entity.description!r}")
        if "struct" in dims.keys:
            triples_text = "; ".join(format_triple(t) for t in entity.neighbors)
            parts.append(f"tuples=[{triples_text}]")
        if "time" in dims.keys:
            time_text = "; ".join(f"{t[3] or '?'} to {t[4] or '?'}" for t in entity.neighbors)
            parts.append(f"time_info=[{time_text}]")
        return "Entity(" + ", ".join(parts) + ")"

    cand_list_text = ", ".join(candidate_names)
    ranked_text = ""
    if ranked_so_far:
        ranked_pairs = ", ".join(f"({name}, {score:.3f})" for name, score in ranked_so_far)
        ranked_text = f" Ranked so far (candidate, score): [{ranked_pairs}]."

    output_format = ", ".join(f"[{_FIELD_LABELS[k]}]" for k in dims.keys)
    full_content = (
        f"Candidate entities that may align with [Main Entity] {main.name}: "
        f"[{cand_list_text}].{ranked_text}\n"
        f"[Main Entity] = {render(main)}\n"
        f"[Candidate Entity] = {render(candidate)}\n"
        "Compared with other candidates, do [Main Entity] and [Candidate Entity] align "
        f"or match? Score {output_format} each as an integer 1-5."
    )
    simple_content = (
        f"Do [Main Entity] {main.name} and [Candidate Entity] {candidate.name} align or match?"
    )
    return LlmMessage(role="user", content=full_content), simple_content


def build_rethink_prompt(
    main_name: str, candidate_names: list[str], ranked: list[tuple[str, float, int]]
) -> LlmMessage:
    """Ports `ask_for_accuracy`'s prompt: "are these alignments good enough?"."""
    cand_list_text = ", ".join(candidate_names)
    aligned_text = ", ".join(
        f"(candidate={name}, score={score:.3f}, rank={rank})" for name, score, rank in ranked
    )
    content = (
        f"Candidate entities that may align with [Main Entity] {main_name}: [{cand_list_text}]. "
        f"Current ranked alignments: [{aligned_text}]. "
        "Compared with the candidate list, are these alignments good enough -- meaning the "
        "top-ranked candidate's score clears a reasonable threshold and is clearly higher than "
        "the rest? Answer with a boolean."
    )
    return LlmMessage(role="user", content=content)

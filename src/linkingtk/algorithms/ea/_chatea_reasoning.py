"""Per-source-entity iterative LLM re-ranking loop for
[ChatEALinker][linkingtk.algorithms.ea.chatea.ChatEALinker].

Ports `main_ChatEA.py`'s `eval_alignment_for_evaluate` -- the confidence
shortcut, widening candidate windows, weighted scoring, and the
stop/continue/rethink control flow -- **minus** its `ref_ent`/`base_rank`
ground-truth-peeking shortcut (the reference skips its LLM loop entirely
when the true target isn't even in the top-20 candidates, since no
re-ranking could ever reach it; that's pure benchmark-cost accounting
using ground truth only evaluation code has access to).
[BaseLinker.link][linkingtk.algorithms.base.BaseLinker.link]'s general
interface has no ground truth to peek at, so keeping that shortcut here
would be a real interface violation, not a faithful port -- its
cost-saving effect is instead reproduced at the benchmark-script layer,
where ground truth is legitimately available (see
`examples/chatea_ea.py`).

Structured JSON scoring (via
[_chatea_prompts][linkingtk.algorithms.ea._chatea_prompts]) replaces the
reference's regex-parsed free text, matching
[LlmBaseLinker][linkingtk.algorithms.llm.LlmBaseLinker]'s own convention.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from linkingtk.algorithms.ea._chatea_context import EntityContext
from linkingtk.algorithms.ea._chatea_prompts import (
    RETHINK_SCHEMA,
    ScoringDimensions,
    build_candidate_prompt,
    build_rethink_prompt,
    scoring_schema,
    summarize_response,
    weighted_score,
)
from linkingtk.llm.client import LlmClient, LlmMessage

logger = logging.getLogger("linkingtk")

SEARCH_WINDOWS: tuple[int, ...] = (1, 10, 20)


@dataclass
class ReasoningConfig:
    dims: ScoringDimensions
    threshold: float = 0.5
    history_len: int = 3
    search_windows: tuple[int, ...] = SEARCH_WINDOWS


def rerank_candidates(
    client: LlmClient,
    system_prompt: str,
    main: EntityContext,
    candidates: list[tuple[EntityContext, float]],
    config: ReasoningConfig,
) -> dict[str, float]:
    """Re-ranks `candidates` (base-method context + score, sorted descending
    by base score) for `main`, returning `entity_id -> final weighted score`
    for every candidate actually reasoned about by the LLM.

    Ports `eval_alignment_for_evaluate`. An empty `candidates` returns `{}`.
    """
    if not candidates:
        return {}

    if len(candidates) >= 2 and candidates[0][1] - candidates[1][1] > config.threshold:
        return {candidates[0][0].entity_id: 1.0}

    context_by_id = {ctx.entity_id: ctx for ctx, _ in candidates}
    candidate_names = [ctx.name for ctx, _ in candidates]
    system_messages = [LlmMessage(role="system", content=system_prompt)]
    chat_history: list[LlmMessage] = []
    scores: dict[str, float] = {}

    for window in config.search_windows:
        bounded_window = min(window, len(candidates))
        for candidate_ctx, _ in reversed(candidates[:bounded_window]):
            ranked_so_far = sorted(scores.items(), key=lambda item: -item[1])
            ranked_named = [(context_by_id[cid].name, score) for cid, score in ranked_so_far]
            full_message, simple_prompt = build_candidate_prompt(
                main, candidate_ctx, candidate_names, ranked_named, config.dims
            )
            messages = system_messages + chat_history + [full_message]
            try:
                response = client.complete_structured(
                    messages, schema=scoring_schema(config.dims), max_tokens=700
                )
            except Exception:
                logger.warning(
                    "ChatEALinker: scoring call failed for %r vs %r, skipping",
                    main.entity_id,
                    candidate_ctx.entity_id,
                    exc_info=True,
                )
                continue

            scores[candidate_ctx.entity_id] = weighted_score(response, config.dims)

            chat_history = chat_history + [
                LlmMessage(role="user", content=simple_prompt),
                LlmMessage(role="assistant", content=summarize_response(response, config.dims)),
            ]
            if len(chat_history) > config.history_len * 2:
                chat_history = chat_history[-config.history_len * 2 :]

        if not scores:
            continue

        ranked = sorted(scores.items(), key=lambda item: -item[1])
        top_id, top_score = ranked[0]
        if top_score < 0.4:
            good_enough = False
        elif top_score > 0.8:
            good_enough = True
        elif len(ranked) > 1 and top_score - ranked[1][1] > 0.5:
            good_enough = True
        else:
            good_enough = _ask_for_accuracy(
                client, system_messages, main, candidate_names, ranked, context_by_id
            )
        if good_enough:
            break

    return scores


def _ask_for_accuracy(
    client: LlmClient,
    system_messages: list[LlmMessage],
    main: EntityContext,
    candidate_names: list[str],
    ranked: list[tuple[str, float]],
    context_by_id: dict[str, EntityContext],
) -> bool:
    """Ports `ask_for_accuracy`: one more LLM call asking whether the
    current top-ranked alignment is confident enough to stop widening."""
    ranked_named = [
        (context_by_id[cid].name, score, rank) for rank, (cid, score) in enumerate(ranked)
    ]
    prompt = build_rethink_prompt(main.name, candidate_names, ranked_named)
    try:
        response = client.complete_structured(
            system_messages + [prompt], schema=RETHINK_SCHEMA, max_tokens=50
        )
    except Exception:
        logger.warning(
            "ChatEALinker: rethink call failed for %r, assuming not good enough",
            main.entity_id,
            exc_info=True,
        )
        return False
    return bool(response.get("good_enough", False))

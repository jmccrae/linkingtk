"""LLM re-ranking layer over an already-scored bi-encoder/cross-encoder candidate set.

Filed as #23: a cheap, already-trained
[ReFinEDLinker][linkingtk.algorithms.el.refined.ReFinEDLinker]/
[BlinkLinker][linkingtk.algorithms.el.blink.BlinkLinker]/
[GlossBertLinker][linkingtk.algorithms.wsd.glossbert.GlossBertLinker]/
[EwiserLinker][linkingtk.algorithms.wsd.ewiser.EwiserLinker]/
[EscLinker][linkingtk.algorithms.wsd.esc.EscLinker] narrows blocked
candidates to a scored top-k first; only that narrow window is then shown
to an LLM for re-ranking, unlike
[LlmBaseLinker][linkingtk.algorithms.llm.LlmBaseLinker] (#21), which sends
an LLM *every* blocked candidate with no first-stage narrowing. No
specific paper is being ported here (unlike
[ChatEALinker][linkingtk.algorithms.ea.chatea.ChatEALinker], #22) --
generalizes the same "retrieve cheap, then let an LLM re-rank only the
top-k" idea to Entity Linking and Word Sense Disambiguation.
"""

from __future__ import annotations

import logging
from typing import Protocol

from linkingtk.algorithms._llm_prompting import (
    _PROMPTS,
    _RESULT_SCHEMA,
    _build_prompt,
    _resolve_candidate_id,
)
from linkingtk.algorithms._llm_prompting import Task as Task  # re-exported
from linkingtk.algorithms.base import DEFAULT_BLOCKING, BaseLinker
from linkingtk.algorithms.matching import DEFAULT_MATCHER, Matcher
from linkingtk.blocking.base import BlockingStrategy
from linkingtk.core.entity import Entity
from linkingtk.core.result import AlignmentResult
from linkingtk.core.source import EntitySource
from linkingtk.llm.client import LlmClient
from linkingtk.utils.graph import Graph

logger = logging.getLogger("linkingtk")


class CandidateScorer(Protocol):
    """What `LlmRerankerLinker` needs from its `base_linker`.

    `ReFinEDLinker`, `BlinkLinker`, `GlossBertLinker`, `EwiserLinker`, and
    `EscLinker` all implement this -- their own `link()` calls it directly
    too, so a reranked source entity's top-k is always built from the same
    scores `link()` would otherwise resolve straight into `matching.match()`.
    """

    def score_candidates(
        self,
        dataset1: list[Entity],
        dataset2: list[Entity] | EntitySource,
        blocking: BlockingStrategy = DEFAULT_BLOCKING,
    ) -> dict[str, list[tuple[str, float]]]:
        """Blocked candidates per source entity id: `(target_id, score)`, unsorted."""


class LlmRerankerLinker(BaseLinker):
    """Re-ranks a `base_linker`'s top-`top_k` blocked candidates with an LLM.

    No `fit()` -- like [LlmBaseLinker][linkingtk.algorithms.llm.LlmBaseLinker],
    the LLM needs no training; only `base_linker`'s own model does, via
    whatever training path it normally uses (e.g.
    [Trainer][linkingtk.train.trainer.Trainer],
    [CrossEncoderTrainer][linkingtk.train.cross_encoder.CrossEncoderTrainer],
    or a loaded checkpoint), done by the caller before constructing this.

    Args:
        base_linker: An already-fitted/loaded
            [CandidateScorer][linkingtk.algorithms.llm_reranker.CandidateScorer]
            -- any of `ReFinEDLinker`/`BlinkLinker`/`GlossBertLinker`/
            `EwiserLinker`/`EscLinker`.
        client: An already-constructed [LlmClient][linkingtk.llm.client.LlmClient].
            This class never imports `openai`/`anthropic` itself.
        task: Which [PromptTemplate][linkingtk.algorithms.llm.PromptTemplate]
            instruction to use -- typically `"el"` or `"wsd"`, matching
            `base_linker`'s own task.
        top_k: How many of `base_linker`'s own highest-scored candidates
            per source entity to actually show the LLM. The LLM can only
            ever re-order within this window, never reach a candidate
            ranked below it -- the actual cost/quality trade-off this
            class exists to make, unlike `LlmBaseLinker`'s unbounded
            candidate count.
        threshold: Confidence shortcut -- if `base_linker`'s own top two
            scores for a source entity differ by more than this, skip the
            LLM call entirely and keep `base_linker`'s own ranking (a
            simplified analog of
            [ChatEALinker][linkingtk.algorithms.ea.chatea.ChatEALinker]'s
            same idea, without its iterative window-widening -- #23 isn't
            tied to a specific paper's algorithm). Defaults to `None`
            (disabled): unlike `ChatEALinker`'s own normalized [0, 1]
            multi-dimension score, `base_linker`'s score scale is
            arbitrary and method-specific -- a bi-encoder's cosine
            similarity lands roughly in [-1, 1] (a fixed threshold like
            `0.5` is meaningful there), but a cross-encoder's raw logit
            margin (e.g. `GlossBertLinker`/`EwiserLinker`/`EscLinker`) is
            unbounded and routinely exceeds any such threshold by 10x or
            more -- measured directly during #23's own WSD benchmark: a
            `threshold=0.5` default silently skipped the LLM for 28/30
            sampled entities, masking any real re-ranking effect entirely.
            Only pass a threshold when you know `base_linker`'s own score
            scale well enough to pick one.
        matching: Strategy used to resolve scored candidates into final
            links.
        max_tokens: Forwarded to `client.complete_structured` per call.

    Note:
        `graph` is accepted for interface compliance but not used, same as
        `LlmBaseLinker`. `blocking` is applied twice per `link()` call --
        once inside `base_linker.score_candidates()`, once again here to
        recover real `Entity` objects for prompt rendering (scores alone
        aren't enough to build a prompt). `BlockingStrategy`s are cheap
        index lookups next to a transformer forward pass, so this isn't
        worth threading Entity objects through `CandidateScorer`'s return
        type to avoid.
    """

    def __init__(
        self,
        base_linker: CandidateScorer,
        client: LlmClient,
        task: Task = "el",
        top_k: int = 10,
        threshold: float | None = None,
        matching: Matcher = DEFAULT_MATCHER,
        max_tokens: int = 2048,
    ) -> None:
        self.base_linker = base_linker
        self.client = client
        self.task = task
        self.top_k = top_k
        self.threshold = threshold
        self.matching = matching
        self.max_tokens = max_tokens

    def link(
        self,
        dataset1: list[Entity],
        dataset2: list[Entity] | EntitySource,
        graph: Graph = None,
        blocking: BlockingStrategy = DEFAULT_BLOCKING,
    ) -> list[AlignmentResult]:
        base_candidates = self.base_linker.score_candidates(dataset1, dataset2, blocking)
        pairs = blocking.candidate_pairs(dataset1, dataset2)
        entities_by_id = {entity2.id: entity2 for _, entity2 in pairs}
        sources_by_id = {entity.id: entity for entity in dataset1}
        instruction = _PROMPTS[self.task].instruction

        final_candidates: dict[str, list[tuple[str, float]]] = {}
        for source_id, scored in base_candidates.items():
            ranked = sorted(scored, key=lambda item: -item[1])[: self.top_k]
            if (
                self.threshold is not None
                and len(ranked) >= 2
                and ranked[0][1] - ranked[1][1] > self.threshold
            ):
                final_candidates[source_id] = ranked
                continue

            candidates = [entities_by_id[cid] for cid, _ in ranked if cid in entities_by_id]
            if not candidates:
                final_candidates[source_id] = ranked
                continue

            messages = _build_prompt(instruction, sources_by_id[source_id], candidates)
            try:
                response = self.client.complete_structured(
                    messages, schema=_RESULT_SCHEMA, max_tokens=self.max_tokens
                )
            except Exception:
                logger.warning(
                    "LlmRerankerLinker: LLM call failed for source entity %r, "
                    "falling back to base ranking",
                    source_id,
                    exc_info=True,
                )
                final_candidates[source_id] = ranked
                continue

            valid_ids = {candidate.id for candidate in candidates}
            scores: dict[str, float] = dict.fromkeys(valid_ids, 0.0)
            for ranking in response.get("rankings", []):
                raw_id, score = ranking.get("candidate_id"), ranking.get("score")
                candidate_id = _resolve_candidate_id(raw_id, valid_ids)
                if candidate_id is None:
                    logger.warning(
                        "LlmRerankerLinker: ignoring hallucinated candidate id %r for source %r",
                        raw_id,
                        source_id,
                    )
                    continue
                scores[candidate_id] = float(score)
            final_candidates[source_id] = list(scores.items())

        return self.matching.match(final_candidates)

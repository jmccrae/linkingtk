"""Pairwise "same entity?" linker on Kev decision models.

[Kev](https://github.com/jaredpalmer/kev) (`jaredpalmer/kev-{0.8b,4b,9b,27b}`
on the Hugging Face Hub) is a family of *decision models*: one text
*state* and a set of typed questions go in, and a calibrated probability
per question comes out of a single forward pass (no text generation).
[KevLinker][linkingtk.algorithms.kev.KevLinker] lexicalizes each blocked
(source, candidate) pair and asks a `noul` (yes/no) question -- "is this
the same entity/sense?" -- then ranks candidates by P(yes).

Kev is reached over HTTP through its own server (`python -m kev.serve`),
not imported: its pinned `torch<2.9` conflicts with this project's torch,
and its inference on hybrid (Gated DeltaNet) Qwen3.5 backbones relies on
Kev's own prefix-cache/branch-mask code, which is better run as-is than
re-ported. Start a server from a separate checkout with::

    git clone https://github.com/jaredpalmer/kev.git && cd kev
    uv sync --extra serve
    uv run --extra serve python -m kev.serve --run jaredpalmer/kev-0.8b --port 8008
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from collections import defaultdict
from typing import Any

from linkingtk.algorithms._llm_prompting import Task
from linkingtk.algorithms.base import DEFAULT_BLOCKING, BaseLinker
from linkingtk.blocking.base import BlockingStrategy
from linkingtk.core.entity import Entity, description_text, label_texts
from linkingtk.core.result import AlignmentResult
from linkingtk.core.source import EntitySource
from linkingtk.matchers import DEFAULT_MATCHER, Matcher
from linkingtk.utils.graph import Graph

logger = logging.getLogger("linkingtk")

__all__ = ["KevClient", "KevError", "KevLinker"]

_SERVER_HINT = (
    "start one with `uv run --extra serve python -m kev.serve --run jaredpalmer/kev-0.8b "
    "--port 8008` from a checkout of https://github.com/jaredpalmer/kev"
)


class KevError(RuntimeError):
    """A Kev server request failed (unreachable, HTTP error, or malformed response)."""


class KevClient:
    """Minimal stdlib client for a Kev server's TypeSafe-compatible `/v1/systemone` API.

    Args:
        base_url: Root URL of a running `kev.serve` instance.
        model: The `model` field sent with each request. `kev.serve` serves
            whichever checkpoint it was started with (`--run`) under
            `"kev-latest"`.
        api_key: Sent as a bearer token, only needed if the server was
            started with `KEV_API_KEY` set.
        timeout: Per-request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8008",
        model: str = "kev-latest",
        api_key: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {"content-type": "application/json"}
        if self.api_key is not None:
            headers["authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body: dict[str, Any] = json.loads(response.read().decode("utf-8"))
                return body
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:500]
            raise KevError(f"Kev server returned HTTP {error.code}: {detail}") from error
        except (urllib.error.URLError, OSError) as error:
            raise KevError(
                f"Could not reach a Kev server at {self.base_url} ({error}); {_SERVER_HINT}"
            ) from error

    def ask_noul(
        self,
        state: str,
        questions: dict[str, str],
        criteria: dict[str, str] | None = None,
    ) -> dict[str, float]:
        """Ask several yes/no questions about one `state` in a single request.

        Args:
            state: The document every question is asked about.
            questions: Question id -> instructions text.
            criteria: Optional `{"true": ..., "false": ...}` descriptions of
                what each answer means, shared by every question.

        Returns:
            Question id -> P(yes).
        """
        spec: dict[str, Any] = {}
        for question_id, instructions in questions.items():
            question: dict[str, Any] = {"type": "noul", "instructions": instructions}
            if criteria:
                question["criteria"] = criteria
            spec[question_id] = question
        body = self._post("/v1/systemone", {"state": state, "model": self.model, "questions": spec})
        try:
            answers = body["answers"]
            return {qid: float(answers[qid]["noul"]) for qid in questions}
        except (KeyError, TypeError, ValueError) as error:
            raise KevError(f"Malformed Kev response: {str(body)[:500]}") from error

    def health(self) -> None:
        """Send one trivial question, raising `KevError` if the server isn't usable."""
        self.ask_noul("ping", {"q": "Is this a test?"})


_TRUE_FALSE: dict[Task, dict[str, str]] = {
    "el": {
        "true": "the mention refers to this same real-world entity",
        "false": "the mention refers to a different entity",
    },
    "wsd": {
        "true": "the word is used in exactly this sense here",
        "false": "the word is used in a different sense here",
    },
    "ea": {
        "true": "both descriptions denote the same real-world entity",
        "false": "they denote different entities",
    },
    "wsa": {
        "true": "both definitions describe the same word sense",
        "false": "they describe different senses",
    },
}

_QUESTIONS: dict[Task, str] = {
    "el": 'Does the mention "{surface}" in the text refer to this entity? {candidate}',
    "wsd": 'Is the word "{surface}" in the text used in this sense? {candidate}',
    "ea": "Is the entity described above the same as this entity? {candidate}",
    "wsa": "Does the sense described above mean the same as this sense? {candidate}",
}


def _mention_surface(entity: Entity) -> str:
    context = entity.context
    if isinstance(context, tuple):
        text, start, end = context
        return text[start:end]
    return " ".join(label_texts(entity))


def _windowed_context(entity: Entity, window_words: int) -> str:
    """The mention's context with its span bracketed as `[[...]]`, trimmed to
    `window_words` whitespace-separated words on each side.

    Kev was trained on states of roughly 384 tokens, while e.g.
    [AidaConllDataset][linkingtk.datasets.aida_conll.AidaConllDataset]
    contexts are whole news articles -- the window keeps the state
    in-distribution.
    """
    context = entity.context
    if isinstance(context, tuple):
        text, start, end = context
        left = text[:start].split()
        right = text[end:].split()
        left_text = " ".join(left[-window_words:]) if window_words > 0 else ""
        right_text = " ".join(right[:window_words]) if window_words > 0 else ""
        ellipsis_left = "... " if len(left) > window_words else ""
        ellipsis_right = " ..." if len(right) > window_words else ""
        return (
            f"{ellipsis_left}{left_text} [[{text[start:end]}]] {right_text}{ellipsis_right}"
        ).strip()
    return context if isinstance(context, str) else ""


def _describe(entity: Entity) -> str:
    """`label1, label2: description` (description omitted when absent)."""
    labels = ", ".join(dict.fromkeys(label_texts(entity)))
    description = description_text(entity)
    return f"{labels}: {description}" if description else labels


def _source_state(entity: Entity, task: Task, window_words: int) -> str:
    """Lexicalize the source entity/mention as Kev's `state`."""
    if task in ("el", "wsd"):
        lines = [f'Mention: "{_mention_surface(entity)}"']
        if task == "wsd":
            lines.append(f"Lemma: {', '.join(label_texts(entity))}")
        context = _windowed_context(entity, window_words)
        if context:
            lines.append(f"Text: {context}")
        return "\n".join(lines)
    return f"Entity: {_describe(entity)}"


def _candidate_question(source: Entity, candidate: Entity, task: Task) -> str:
    return _QUESTIONS[task].format(surface=_mention_surface(source), candidate=_describe(candidate))


class KevLinker(BaseLinker):
    """Scores each blocked (source, candidate) pair with a Kev `noul` question.

    Each source entity is lexicalized once as Kev's `state` (for EL/WSD:
    the mention's surface form and a word window of its context with the
    span bracketed), and every blocked candidate becomes one yes/no
    question ("does the mention refer to <label>: <description>?"). Kev
    runs each question as its own block-causal branch that attends only
    to the state and itself, so batching all of a source's candidates into
    one request still yields *independent pairwise* judgments -- no
    candidate sees another -- while encoding the shared context once.
    Candidates are then ranked by P(yes).

    No `fit()`: Kev is used zero-shot.

    Args:
        client: A [KevClient][linkingtk.algorithms.kev.KevClient] pointing
            at a running `kev.serve`.
        task: `"el"`, `"wsd"`, `"ea"` or `"wsa"` -- selects the
            lexicalization and question wording.
        context_window: Words of mention context kept on each side of the
            span (EL/WSD only).
        questions_per_request: Maximum candidates sent per request; a
            source with more candidates is split across requests.
        matching: Strategy used to resolve scored candidates into final
            links.

    Note:
        `graph` is accepted for interface compliance but not used. A
        failed request is logged and its candidates scored 0.0 rather than
        aborting the whole run, the same policy as
        [LlmBaseLinker][linkingtk.algorithms.llm.LlmBaseLinker].
    """

    def __init__(
        self,
        client: KevClient,
        task: Task = "el",
        context_window: int = 50,
        questions_per_request: int = 32,
        matching: Matcher = DEFAULT_MATCHER,
    ) -> None:
        if questions_per_request < 1:
            raise ValueError("questions_per_request must be at least 1")
        self.client = client
        self.task = task
        self.context_window = context_window
        self.questions_per_request = questions_per_request
        self.matching = matching

    def score_candidates(
        self,
        dataset1: list[Entity],
        dataset2: list[Entity] | EntitySource,
        blocking: BlockingStrategy = DEFAULT_BLOCKING,
    ) -> dict[str, list[tuple[str, float]]]:
        """P(yes) per blocked candidate: `{source_id: [(target_id, p_yes), ...]}`."""
        candidates: dict[str, list[Entity]] = defaultdict(list)
        sources: dict[str, Entity] = {}
        seen: set[tuple[str, str]] = set()
        for entity1, entity2 in blocking.candidate_pairs(dataset1, dataset2):
            sources[entity1.id] = entity1
            if (entity1.id, entity2.id) not in seen:
                seen.add((entity1.id, entity2.id))
                candidates[entity1.id].append(entity2)

        criteria = _TRUE_FALSE[self.task]
        scores: dict[str, list[tuple[str, float]]] = {}
        for source_id, source_candidates in candidates.items():
            source = sources[source_id]
            state = _source_state(source, self.task, self.context_window)
            scored: list[tuple[str, float]] = []
            for offset in range(0, len(source_candidates), self.questions_per_request):
                chunk = source_candidates[offset : offset + self.questions_per_request]
                questions = {
                    f"c{index}": _candidate_question(source, candidate, self.task)
                    for index, candidate in enumerate(chunk)
                }
                try:
                    p_yes = self.client.ask_noul(state, questions, criteria)
                except KevError as error:
                    logger.warning("Kev request failed for %s: %s", source_id, error)
                    p_yes = {}
                scored.extend(
                    (candidate.id, p_yes.get(f"c{index}", 0.0))
                    for index, candidate in enumerate(chunk)
                )
            scores[source_id] = scored
        return scores

    def link(
        self,
        dataset1: list[Entity],
        dataset2: list[Entity] | EntitySource,
        graph: Graph = None,
        blocking: BlockingStrategy = DEFAULT_BLOCKING,
    ) -> list[AlignmentResult]:
        return self.matching.match(self.score_candidates(dataset1, dataset2, blocking))

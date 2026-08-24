"""Training loop for [EscEncoder][linkingtk.algorithms.wsd.esc.EscEncoder].

A dedicated trainer, not a reuse of
[CrossEncoderTrainer][linkingtk.train.cross_encoder.CrossEncoderTrainer] or
[EwiserTrainer][linkingtk.train.ewiser_trainer.EwiserTrainer]: ESC's real
objective is extractive-QA span-position cross-entropy (one joint sequence
per **mention**, containing every one of that mention's candidates' glosses
at once, with the gold candidate's token span as the target start/end
position) -- not a pairwise margin/BCE loss over independent
`(mention, candidate)` pairs (`CrossEncoderTrainer`), and not per-sentence
full-inventory classification (`EwiserTrainer`). The loss itself needs no
hand-rolled cross-entropy: `EscEncoder.forward(..., start_positions=...,
end_positions=...)` already returns HF's own `(start_loss + end_loss) / 2`
via `.loss` (see `EscEncoder`'s own module docstring).

Unlike `EwiserTrainer` (whose training never needs a candidate-generation
step -- gold indices are resolved directly against its fixed output
vocabulary), ESC's training *is* candidate generation: each mention's
training example requires the same real, blocking-restricted candidate
senses (with real gloss text) that `EscLinker.link()` would score at
inference time. So `EscTrainer` takes a `senses` source unconditionally
(not only when evaluating, like `EwiserTrainer.eval_dataset2`) -- used to
build both training examples and (if `eval_data` is given) evaluation
candidates.
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

import torch

from linkingtk.algorithms.wsd._esc_text import (
    build_joint_sequence,
    candidate_gloss,
    insert_classify_markers,
    pad_encoded_sequences,
)
from linkingtk.algorithms.wsd._ewiser_text import mention_sentence_and_span
from linkingtk.algorithms.wsd.esc import EscEncoder
from linkingtk.blocking.base import BlockingStrategy
from linkingtk.blocking.exact import ExactMatch
from linkingtk.core.entity import Entity
from linkingtk.core.source import EntitySource
from linkingtk.eval.evaluator import Evaluator
from linkingtk.eval.report import EvaluationReport
from linkingtk.train._optim import build_optimizer
from linkingtk.train.arguments import TrainingArguments
from linkingtk.utils.device import resolve_device

logger = logging.getLogger("linkingtk")

DEFAULT_BLOCKING = ExactMatch()


class _TrainingExample(NamedTuple):
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    start_position: int
    end_position: int


class EscTrainer:
    """Trains an [EscEncoder][linkingtk.algorithms.wsd.esc.EscEncoder] with
    extractive-QA span-position cross-entropy.

    Args:
        model: The `EscEncoder` to train.
        args: Hyperparameters and settings, see
            [TrainingArguments][linkingtk.train.arguments.TrainingArguments].
            ``negative_samples_ratio``/``loss``/``temperature``/``margin``
            are ignored -- there's no candidate sampling or embedding-pair
            objective here, the loss comes straight from the QA head (see
            the module docstring).
        train_data: ``(mentions, ground_truth)`` -- `mentions` a
            `list[Entity]`, `ground_truth` a list of ``(mention_id,
            sense_id)`` pairs, same shape
            [EwiserTrainer][linkingtk.train.ewiser_trainer.EwiserTrainer]
            takes.
        senses: The real target sense set (or
            [EntitySource][linkingtk.core.source.EntitySource], e.g. a
            [WnEntitySource][linkingtk.sources.wn.WnEntitySource]) `blocking`
            resolves each mention's candidates against -- required
            unconditionally (unlike `EwiserTrainer.eval_dataset2`, which is
            only needed for evaluation): ESC's training examples are
            themselves built from real candidate glosses, not a fixed
            output vocabulary index.
        eval_data: Optional held-out ``(mentions, ground_truth)``,
            evaluated the same way `EscLinker.link()` would score it (via
            `model.score` + `blocking` against `senses`), at the end of
            every epoch.
        blocking: Candidate-generation strategy for both training examples
            and `eval_data`. Defaults to
            [ExactMatch][linkingtk.blocking.exact.ExactMatch], matching
            WSD's "blocking is always exact" convention (see
            `CrossEncoderTrainer`'s own docstring) -- pass a
            POS-restricted wrapper (as `examples/esc_reproduction.py`/
            `esc_benchmark.py` do) to match the reference's own candidate
            generation.

    Attributes:
        eval_history: One [EvaluationReport][linkingtk.eval.report.EvaluationReport]
            per epoch, appended in `train()` when `eval_data` is given.
        loss_history: Mean training loss per epoch, always appended in
            `train()` -- the most direct signal that the QA span-position
            objective is actually decreasing, independent of `eval_data`
            (which, on a very small eval set, can already sit at a ceiling
            precision@1 with no room to visibly "improve").
    """

    def __init__(
        self,
        model: EscEncoder,
        args: TrainingArguments,
        train_data: tuple[list[Entity], list[tuple[str, str]]],
        senses: list[Entity] | EntitySource,
        eval_data: tuple[list[Entity], list[tuple[str, str]]] | None = None,
        blocking: BlockingStrategy = DEFAULT_BLOCKING,
    ) -> None:
        self.model = model
        self.args = args
        self.train_data = train_data
        self.senses = senses
        self.eval_data = eval_data
        self.blocking = blocking
        self.eval_history: list[EvaluationReport] = []
        self.loss_history: list[float] = []

    def train(self) -> None:
        """Run the training loop, updating ``self.model`` in place."""
        args = self.args
        device = resolve_device(args.device)
        self.model.to(device)

        mentions, ground_truth = self.train_data
        examples = self._build_examples(mentions, ground_truth)
        num_batches_per_epoch = -(-len(examples) // args.batch_size)
        optimizer, scheduler = build_optimizer(
            self.model, args, num_batches_per_epoch * args.num_epochs
        )

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "EscTrainer: %d training examples (%d mentions given), %d batches/epoch, %d epochs",
            len(examples),
            len(mentions),
            num_batches_per_epoch,
            args.num_epochs,
        )

        for epoch in range(args.num_epochs):
            shuffled = list(examples)
            random.shuffle(shuffled)
            loss_total = 0.0
            num_batches = 0
            for start in range(0, len(shuffled), args.batch_size):
                batch = shuffled[start : start + args.batch_size]
                loss = self._train_step(batch, device)
                optimizer.zero_grad()
                loss.backward()  # type: ignore[no-untyped-call]
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                loss_total += float(loss.item())
                num_batches += 1

            mean_loss = loss_total / max(num_batches, 1)
            self.loss_history.append(mean_loss)

            if self.eval_data is not None:
                self.eval_history.append(self._evaluate())
                logger.info(
                    "EscTrainer: epoch %d/%d done, mean loss %.4f, eval %s",
                    epoch + 1,
                    args.num_epochs,
                    mean_loss,
                    self.eval_history[-1].metrics,
                )
            else:
                logger.info(
                    "EscTrainer: epoch %d/%d done, mean loss %.4f",
                    epoch + 1,
                    args.num_epochs,
                    mean_loss,
                )

            # Saved after *every* epoch, not just at the end -- a long run
            # can span hours unattended; without this, an interruption
            # loses everything (same rationale as EwiserTrainer/CrossEncoderTrainer).
            torch.save(self.model.state_dict(), output_dir / "model.pt")
            logger.info("EscTrainer: checkpoint saved to %s", output_dir / "model.pt")

    def _train_step(self, batch: list[_TrainingExample], device: torch.device) -> torch.Tensor:
        sequences = [(example.input_ids, example.attention_mask) for example in batch]
        pad_token_id = self.model.tokenizer.pad_token_id
        input_ids, attention_mask = pad_encoded_sequences(sequences, pad_token_id)
        start_positions = torch.tensor([example.start_position for example in batch])
        end_positions = torch.tensor([example.end_position for example in batch])
        outputs = self.model(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            start_positions=start_positions.to(device),
            end_positions=end_positions.to(device),
        )
        loss: torch.Tensor = outputs.loss
        return loss

    def _build_examples(
        self, mentions: list[Entity], ground_truth: list[tuple[str, str]]
    ) -> list[_TrainingExample]:
        gold_by_mention = dict(ground_truth)
        candidates_by_mention = _resolve_candidates(mentions, self.senses, self.blocking)

        examples: list[_TrainingExample] = []
        unresolved = 0
        for mention in mentions:
            gold_id = gold_by_mention.get(mention.id)
            if gold_id is None:
                continue
            candidates = list(candidates_by_mention.get(mention.id, []))
            candidate_ids = [sense.id for sense in candidates]
            if gold_id not in candidate_ids:
                unresolved += 1
                continue
            random.shuffle(candidates)
            gold_index = [sense.id for sense in candidates].index(gold_id)

            text, start, end = mention_sentence_and_span(mention)
            marked = insert_classify_markers(text, start, end)
            glosses = [candidate_gloss(sense) for sense in candidates]
            input_ids, attention_mask, spans = build_joint_sequence(
                self.model.tokenizer, marked, glosses, self.model.max_length
            )
            gold_span = spans[gold_index]
            if gold_span is None:
                unresolved += 1
                continue
            start_position, end_position = gold_span
            example = _TrainingExample(input_ids, attention_mask, start_position, end_position)
            examples.append(example)

        if unresolved:
            logger.warning(
                "EscTrainer: %d mention(s) dropped (gold sense not among resolved "
                "candidates, or its gloss span didn't survive truncation)",
                unresolved,
            )
        return examples

    def _evaluate(self) -> EvaluationReport:
        assert self.eval_data is not None  # only called when eval_data is set
        mentions, ground_truth = self.eval_data
        self.model.eval()
        with torch.no_grad():
            pairs = list(self.blocking.candidate_pairs(mentions, self.senses))
            scores = self.model.score(pairs) if pairs else torch.empty(0)
        self.model.train()

        scores_by_source: dict[str, list[tuple[str, float]]] = {}
        for (entity1, entity2), score in zip(pairs, scores.tolist(), strict=True):
            scores_by_source.setdefault(entity1.id, []).append((entity2.id, score))

        ranked_predictions = [
            (
                source_id,
                [
                    target_id
                    for target_id, _score in sorted(
                        candidates, key=lambda item: item[1], reverse=True
                    )
                ],
            )
            for source_id, candidates in scores_by_source.items()
        ]
        return Evaluator.evaluate_ranked(ranked_predictions, ground_truth, top_k=[1])


def _resolve_candidates(
    mentions: list[Entity], senses: list[Entity] | EntitySource, blocking: BlockingStrategy
) -> dict[str, list[Entity]]:
    """Every distinct candidate `blocking` resolves for each of `mentions`,
    deduplicated by sense id, preserving first-seen order."""
    candidates_by_mention: dict[str, list[Entity]] = defaultdict(list)
    seen: dict[str, set[str]] = defaultdict(set)
    for mention, sense in blocking.candidate_pairs(mentions, senses):
        if sense.id in seen[mention.id]:
            continue
        seen[mention.id].add(sense.id)
        candidates_by_mention[mention.id].append(sense)
    return candidates_by_mention

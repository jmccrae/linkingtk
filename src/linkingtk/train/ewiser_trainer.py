"""Training loop for [EwiserEncoder][linkingtk.algorithms.wsd.ewiser.EwiserEncoder].

A dedicated trainer, not a reuse of
[CrossEncoderTrainer][linkingtk.train.cross_encoder.CrossEncoderTrainer]:
EWISER's real objective is full-inventory classification (one softmax
cross-entropy per annotated word, over the entire
[SenseVocabulary][linkingtk.algorithms.wsd._ewiser_vocab.SenseVocabulary]),
not a pairwise margin/BCE loss over `(mention, candidate)` pairs -- there's
no independently-embedded pair of vectors to contrast, and batching is
per-sentence (every annotated word in a sentence scored by one shared
forward pass), not per-pair.

Confirmed directly against the reference's own training code
(`ewiser/fairseq_ext/criterions/weighted_cross_entropy.py`): despite the
name, "weighted cross-entropy" is **not** frequency-based class balancing
-- the weight is `1.0` on every real sense and `0.0` only on the handful
of reserved/special vocabulary slots (a padding mask, not class
weighting, and never actually reachable as a *gold* class anyway). This
trainer's loss is plain ``torch.nn.functional.cross_entropy`` over only
the resolved (word position, gold vocabulary index) pairs -- unresolved
mentions are filtered out before the loss is built (see
`_group_into_sentence_batches`), which does the same masking job as the
reference's own `ignore_index` without needing one.
"""

from __future__ import annotations

import dataclasses
import logging
import random
from pathlib import Path

import torch
import torch.nn.functional as functional

from linkingtk.algorithms.wsd._ewiser_text import (
    mention_sentence_and_span,
    whitespace_tokenize_with_offsets,
    word_index_for_span,
)
from linkingtk.algorithms.wsd._ewiser_vocab import SenseVocabulary
from linkingtk.algorithms.wsd.ewiser import EwiserEncoder
from linkingtk.blocking.base import BlockingStrategy
from linkingtk.blocking.exact import ExactMatch
from linkingtk.core.entity import Entity
from linkingtk.core.source import EntitySource
from linkingtk.eval.evaluator import Evaluator
from linkingtk.eval.report import EvaluationReport
from linkingtk.exceptions import LinkingTKError
from linkingtk.train._optim import build_optimizer
from linkingtk.train.arguments import TrainingArguments
from linkingtk.utils.device import resolve_device

logger = logging.getLogger("linkingtk")

DEFAULT_BLOCKING = ExactMatch()


class EwiserTrainer:
    """Trains an [EwiserEncoder][linkingtk.algorithms.wsd.ewiser.EwiserEncoder]
    with full-inventory cross-entropy.

    Args:
        model: The `EwiserEncoder` to train.
        args: Hyperparameters and settings, see
            [TrainingArguments][linkingtk.train.arguments.TrainingArguments].
            ``negative_samples_ratio``/``loss``/``temperature``/``margin``/
            ``use_peft`` are ignored -- there's no candidate sampling or
            embedding-pair objective here, and PEFT/LoRA has nothing
            meaningful to adapt on a model whose only trainable parameters
            are already the small decoder (the encoder stays frozen, see
            [EwiserEncoder][linkingtk.algorithms.wsd.ewiser.EwiserEncoder]'s
            own docstring). ``args.learning_rate`` is the single-stage (or
            post-freeze) learning rate; see `output_freeze_lr` for the
            freeze-stage rate.
        train_data: ``(mentions, ground_truth)`` -- `mentions` a
            `list[Entity]` (the shape
            [SemCorDataset][linkingtk.datasets.semcor.SemCorDataset]/
            [UfsacDataset][linkingtk.datasets.ufsac.UfsacDataset] produce
            as `dataset1`), `ground_truth` a list of
            ``(mention_id, synset_id)`` pairs. A mention whose gold
            synset isn't in `model.vocabulary` is dropped from training
            (logged, not an error) -- the same "unresolved -> excluded"
            convention `UfsacDataset` itself already uses.
        eval_data: Optional held-out ``(mentions, ground_truth)``,
            evaluated the same way `link()` would score it (via
            `model.score` + `blocking`), at the end of every epoch.
        eval_dataset2: The real target set (or
            [EntitySource][linkingtk.core.source.EntitySource], e.g. a
            [WnEntitySource][linkingtk.sources.wn.WnEntitySource]) to
            block `eval_data`'s mentions against. **Required** whenever
            `eval_data` is given -- unlike
            [CrossEncoderTrainer][linkingtk.train.cross_encoder.CrossEncoderTrainer],
            which can (if less accurately) fall back to a `dataset2`
            derived from its own `train_data`'s real candidate `Entity`
            objects, `EwiserTrainer.train_data`/`eval_data` carry only
            gold synset *ids* (see `train_data`), not full `Entity`
            objects with real labels -- a same-shaped fallback here would
            synthesize label-less entities that can never match anything
            under `ExactMatch`, silently produce zero candidates, and
            zero out every metric. Failing loudly beats that.
        blocking: Candidate-generation strategy for `eval_data`. Defaults
            to [ExactMatch][linkingtk.blocking.exact.ExactMatch], matching
            WSD's "blocking is always exact" convention.
        freeze_output_epochs: Number of leading epochs trained with
            `model.decoder.logits.weight` (the sense output embedding
            layer) frozen -- matches the reference's own "freeze-then-thaw"
            schedule (Section 3.4.1 of the paper), which protects an
            externally-pretrained (LMMS/SensEmBERT) initialization from
            noisy early gradients. Defaults to ``0`` (single-stage):
            freezing a *randomly initialized* output layer for a few
            epochs has no comparable benefit -- pass a nonzero value only
            when `model` was built with a real pretrained
            `output_embedding_init` (see
            [load_synset_centroid_vectors][linkingtk.algorithms.wsd._ewiser_sense_embeddings.load_synset_centroid_vectors]/
            [build_synset_centroid_vectors_from_lmms][linkingtk.algorithms.wsd._ewiser_sense_embeddings.build_synset_centroid_vectors_from_lmms]),
            e.g. when continuing training from one of the published
            checkpoints or from-scratch with LMMS/SensEmBERT vectors --
            this package doesn't bundle those vector files themselves
            (same as it doesn't bundle the published checkpoints), only
            the loaders that consume them.
        output_freeze_lr: Learning rate for the leading
            `freeze_output_epochs` epochs. Only relevant when
            `freeze_output_epochs > 0`.
        output_unfreeze_lr: Learning rate from `freeze_output_epochs`
            onward (or for the entire run, when `freeze_output_epochs=0`
            -- `args.learning_rate` is used instead in that single-stage
            case; see `args`).

    Attributes:
        eval_history: One [EvaluationReport][linkingtk.eval.report.EvaluationReport]
            per epoch, appended in `train()` when `eval_data` is given.
    """

    def __init__(
        self,
        model: EwiserEncoder,
        args: TrainingArguments,
        train_data: tuple[list[Entity], list[tuple[str, str]]],
        eval_data: tuple[list[Entity], list[tuple[str, str]]] | None = None,
        eval_dataset2: list[Entity] | EntitySource | None = None,
        blocking: BlockingStrategy = DEFAULT_BLOCKING,
        freeze_output_epochs: int = 0,
        output_freeze_lr: float = 1e-4,
        output_unfreeze_lr: float = 1e-5,
    ) -> None:
        if eval_data is not None and eval_dataset2 is None:
            raise LinkingTKError(
                "EwiserTrainer requires eval_dataset2 whenever eval_data is given -- "
                "see eval_dataset2's own docstring for why no fallback is offered."
            )
        self.model = model
        self.args = args
        self.train_data = train_data
        self.eval_data = eval_data
        self.eval_dataset2 = eval_dataset2
        self.blocking = blocking
        self.freeze_output_epochs = freeze_output_epochs
        self.output_freeze_lr = output_freeze_lr
        self.output_unfreeze_lr = output_unfreeze_lr
        self.eval_history: list[EvaluationReport] = []

    def train(self) -> None:
        """Run the training loop, updating ``self.model`` in place."""
        device = resolve_device(self.args.device)
        self.model.to(device)

        batches = _group_into_sentence_batches(
            *self.train_data, self.model.vocabulary, self.args.batch_size
        )
        logger.info(
            "EwiserTrainer: %d sentence-batches/epoch, %d epochs",
            len(batches),
            self.args.num_epochs,
        )

        stage_lr = (
            self.output_freeze_lr if self.freeze_output_epochs > 0 else self.args.learning_rate
        )
        if self.freeze_output_epochs > 0:
            self.model.decoder.logits.weight.requires_grad = False
        optimizer, scheduler = build_optimizer(
            self.model,
            dataclasses.replace(self.args, learning_rate=stage_lr),
            len(batches) * self.args.num_epochs,
        )

        output_dir = Path(self.args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(self.args.num_epochs):
            if epoch == self.freeze_output_epochs and self.freeze_output_epochs > 0:
                self.model.decoder.logits.weight.requires_grad = True
                optimizer, scheduler = build_optimizer(
                    self.model,
                    dataclasses.replace(self.args, learning_rate=self.output_unfreeze_lr),
                    len(batches) * (self.args.num_epochs - epoch),
                )
                logger.info(
                    "EwiserTrainer: epoch %d unfreezing decoder.logits.weight, lr -> %s",
                    epoch + 1,
                    self.output_unfreeze_lr,
                )

            shuffled = list(batches)
            random.shuffle(shuffled)
            loss_total = 0.0
            for sentences, targets in shuffled:
                logits_per_sentence = self.model._encode_chunk(sentences)
                losses = []
                for logits, (word_indices, gold_indices) in zip(
                    logits_per_sentence, targets, strict=True
                ):
                    if not word_indices:
                        continue
                    word_index_tensor = torch.tensor(word_indices, device=device)
                    gold_index_tensor = torch.tensor(gold_indices, device=device)
                    losses.append(
                        functional.cross_entropy(
                            logits[word_index_tensor], gold_index_tensor, reduction="sum"
                        )
                    )
                if not losses:
                    continue
                loss = torch.stack(losses).sum() / sum(len(w) for w, _g in targets)
                optimizer.zero_grad()
                loss.backward()  # type: ignore[no-untyped-call]
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                loss_total += float(loss.item())

            mean_loss = loss_total / max(len(batches), 1)

            if self.eval_data is not None:
                self.eval_history.append(self._evaluate())
                logger.info(
                    "EwiserTrainer: epoch %d/%d done, mean loss %.4f, eval %s",
                    epoch + 1,
                    self.args.num_epochs,
                    mean_loss,
                    self.eval_history[-1].metrics,
                )
            else:
                logger.info(
                    "EwiserTrainer: epoch %d/%d done, mean loss %.4f",
                    epoch + 1,
                    self.args.num_epochs,
                    mean_loss,
                )

            # Saved after *every* epoch, not just at the end -- a long run
            # can span hours unattended; without this, an interruption
            # loses everything (same rationale as CrossEncoderTrainer).
            torch.save(self.model.state_dict(), output_dir / "model.pt")
            logger.info("EwiserTrainer: checkpoint saved to %s", output_dir / "model.pt")

    def _evaluate(self) -> EvaluationReport:
        assert self.eval_data is not None  # only called when eval_data is set
        assert self.eval_dataset2 is not None  # enforced in __init__
        mentions, ground_truth = self.eval_data
        dataset2 = self.eval_dataset2
        self.model.eval()
        with torch.no_grad():
            pairs = list(self.blocking.candidate_pairs(mentions, dataset2))
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


def _group_into_sentence_batches(
    mentions: list[Entity],
    ground_truth: list[tuple[str, str]],
    vocabulary: SenseVocabulary,
    batch_size: int,
) -> list[tuple[list[str], list[tuple[list[int], list[int]]]]]:
    """Group `mentions` into per-sentence batches of `batch_size` sentences,
    with each sentence's annotated word indices and gold vocabulary indices
    resolved up front.

    Returns a list of ``(sentences, targets)`` where ``targets[i]`` is
    ``(word_indices, gold_indices)`` for ``sentences[i]`` -- parallel lists
    of every annotated word's position and gold class index in that
    sentence, ready for `torch.nn.functional.cross_entropy`.
    """
    gold_by_mention: dict[str, str] = dict(ground_truth)
    mentions_by_sentence: dict[str, list[Entity]] = {}
    for mention in mentions:
        if mention.id not in gold_by_mention:
            continue
        text, _start, _end = mention_sentence_and_span(mention)
        mentions_by_sentence.setdefault(text, []).append(mention)

    unresolved = 0
    sentences: list[str] = []
    targets: list[tuple[list[int], list[int]]] = []
    for text, sentence_mentions in mentions_by_sentence.items():
        tokens = whitespace_tokenize_with_offsets(text)
        word_indices: list[int] = []
        gold_indices: list[int] = []
        for mention in sentence_mentions:
            _text, start, end = mention_sentence_and_span(mention)
            word_index = word_index_for_span(tokens, start, end)
            gold_index = vocabulary.index_for(gold_by_mention[mention.id])
            if word_index is None or gold_index is None:
                unresolved += 1
                continue
            word_indices.append(word_index)
            gold_indices.append(gold_index)
        sentences.append(text)
        targets.append((word_indices, gold_indices))

    if unresolved:
        logger.warning(
            "EwiserTrainer: %d mention(s) dropped (gold synset not in vocabulary or "
            "span didn't resolve to a word position)",
            unresolved,
        )

    batches: list[tuple[list[str], list[tuple[list[int], list[int]]]]] = []
    for start in range(0, len(sentences), batch_size):
        batches.append((sentences[start : start + batch_size], targets[start : start + batch_size]))
    return batches

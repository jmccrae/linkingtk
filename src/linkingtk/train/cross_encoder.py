"""Training loop for cross-encoder linking modules.

Complements [Trainer][linkingtk.train.trainer.Trainer] (bi-encoder/GNN
models: independently `encode()` each side, then contrast the resulting
embeddings) with the shape a true cross-encoder needs: one joint forward
pass per `(entity1, entity2)` pair, scored directly, with no independently
comparable embedding space to take a dot product over.
[GlossBertLinker][linkingtk.algorithms.wsd.glossbert.GlossBertLinker] is
the first (and, as of this writing, only) consumer.

``model`` must satisfy [CrossEncoderModel][linkingtk.train.cross_encoder.CrossEncoderModel]:
a ``torch.nn.Module`` exposing a differentiable ``score(pairs) -> torch.Tensor``
method, one logit-margin per pair (higher = more likely a true match).
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import torch
import torch.nn.functional as functional

from linkingtk.blocking.base import BlockingStrategy
from linkingtk.blocking.exact import ExactMatch
from linkingtk.blocking.negative_sampling import sample_hard_negatives
from linkingtk.core.entity import Entity
from linkingtk.core.source import EntitySource
from linkingtk.eval.evaluator import Evaluator
from linkingtk.train._optim import build_optimizer
from linkingtk.train._peft import apply_peft
from linkingtk.train.arguments import TrainingArguments
from linkingtk.utils.device import resolve_device

if TYPE_CHECKING:
    from linkingtk.eval.report import EvaluationReport

DEFAULT_BLOCKING = ExactMatch()


class CrossEncoderModel(Protocol):
    """Structural type for the model
    [CrossEncoderTrainer][linkingtk.train.cross_encoder.CrossEncoderTrainer] trains.

    Any ``torch.nn.Module`` exposing a differentiable ``score`` method
    satisfies this -- gradients must flow from its output back to
    ``self.parameters()``, and it must return its output on the same
    device its parameters currently live on (``CrossEncoderTrainer`` moves
    the model via ``.to(device)`` but never touches tensors itself).
    """

    def score(self, pairs: list[tuple[Entity, Entity]]) -> torch.Tensor:
        """Return a ``(len(pairs),)`` tensor: one logit-margin per pair."""
        ...

    def parameters(self) -> Any: ...
    def to(self, device: Any) -> Any: ...
    def train(self, mode: bool = True) -> Any: ...
    def eval(self) -> Any: ...
    def state_dict(self) -> Any: ...


class CrossEncoderTrainer:
    """Trains a [CrossEncoderModel][linkingtk.train.cross_encoder.CrossEncoderModel]
    with binary cross-entropy.

    Args:
        model: The model to train.
        args: Hyperparameters and settings, see
            [TrainingArguments][linkingtk.train.arguments.TrainingArguments].
            ``loss``/``temperature``/``margin`` are ignored -- binary
            cross-entropy on `model`'s own logit-margin is the only
            objective a cross-encoder needs (there's no separately
            embedded pair of vectors to contrast).
        train_data: ``(source_entity, target_entity)`` positive pairs.
        eval_data: Optional held-out positive pairs. If given, every
            epoch's end scores `blocking.candidate_pairs(eval_dataset1,
            eval_dataset2)` and appends a Hits@1 report to
            ``eval_history``. Deliberately **not** exhaustive against a
            full target set the way ``Trainer._evaluate`` is: unlike EA/EL
            (where blocking is a scalability shortcut over a semantically
            meaningful full target set, so exhaustive ranking is the
            honest measure -- see issue #37/#38), DESIGN.md defines WSD
            blocking as *always* exact, because ranking a mention against
            an unrelated lemma's senses isn't a meaningful question in the
            first place. A mention's own blocking-restricted candidates --
            typically just its lemma's own senses -- *are* the right
            ranking scope here, matching the standard WSD "all-words"
            evaluation protocol (and every published WSD system, including
            GlossBERT itself).
        eval_dataset2: The real target set (or
            [EntitySource][linkingtk.core.source.EntitySource], e.g. a
            [WnEntitySource][linkingtk.sources.wn.WnEntitySource]) to block
            `eval_data`'s sources against. Defaults to the deduplicated
            targets *within* `eval_data` itself, matching
            ``Trainer._evaluate``'s convention -- correct for EA/EL, where
            the eval split's own KB subset already stands in for the full
            target set, but silently too easy for WSD: the eval split's
            own gold senses are a far narrower (and far less confusable)
            candidate pool than a mention's real lemma-wide sense
            inventory. Concretely, without this the reported Hits@1 can
            come out roughly 2x optimistic (measured directly:
            60% self-reported vs. 33% measured via
            [GlossBertLinker][linkingtk.algorithms.wsd.glossbert.GlossBertLinker]'s
            own ``link()``, same held-out mentions, same trained model --
            always pass the real `dataset2` for WSD.
        blocking: Strategy used to mine hard negatives from ``train_data``
            (via
            [sample_hard_negatives][linkingtk.blocking.negative_sampling.sample_hard_negatives])
            and to restrict eval candidates. Defaults to
            [ExactMatch][linkingtk.blocking.exact.ExactMatch], matching
            WSD's own "blocking is always exact" convention.

    Attributes:
        eval_history: One [EvaluationReport][linkingtk.eval.report.EvaluationReport]
            per epoch, appended in ``train()`` when ``eval_data`` is given.
    """

    def __init__(
        self,
        model: CrossEncoderModel,
        args: TrainingArguments,
        train_data: list[tuple[Entity, Entity]],
        eval_data: list[tuple[Entity, Entity]] | None = None,
        eval_dataset2: list[Entity] | EntitySource | None = None,
        blocking: BlockingStrategy = DEFAULT_BLOCKING,
    ) -> None:
        self.model = model
        self.args = args
        self.train_data = train_data
        self.eval_data = eval_data
        self.eval_dataset2 = eval_dataset2
        self.blocking = blocking
        self.eval_history: list[EvaluationReport] = []

    def train(self) -> None:
        """Run the training loop, updating ``self.model`` in place."""
        args = self.args
        device = resolve_device(args.device)
        self.model.to(device)
        self.model = apply_peft(self.model, args)

        dataset1, dataset2, ground_truth = _entities_and_ground_truth(self.train_data)
        negatives = sample_hard_negatives(
            dataset1, dataset2, ground_truth, self.blocking, top_k=args.negative_samples_ratio
        )
        examples: list[tuple[Entity, Entity, float]] = [
            (entity1, entity2, 1.0) for entity1, entity2 in self.train_data
        ] + [(entity1, entity2, 0.0) for entity1, entity2 in negatives]

        num_batches_per_epoch = -(-len(examples) // args.batch_size)
        optimizer, scheduler = build_optimizer(
            self.model, args, num_batches_per_epoch * args.num_epochs
        )

        for _ in range(args.num_epochs):
            shuffled = list(examples)
            random.shuffle(shuffled)
            for start in range(0, len(shuffled), args.batch_size):
                batch = shuffled[start : start + args.batch_size]
                pairs = [(entity1, entity2) for entity1, entity2, _label in batch]
                labels = torch.tensor([label for _entity1, _entity2, label in batch], device=device)
                logits = self.model.score(pairs)
                loss = functional.binary_cross_entropy_with_logits(logits, labels)
                optimizer.zero_grad()
                loss.backward()  # type: ignore[no-untyped-call]
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            if self.eval_data is not None:
                self.eval_history.append(self._evaluate())

        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), Path(args.output_dir) / "model.pt")

    def _evaluate(self) -> EvaluationReport:
        assert self.eval_data is not None  # only called when eval_data is set
        dataset1, derived_dataset2, ground_truth = _entities_and_ground_truth(self.eval_data)
        dataset2 = self.eval_dataset2 if self.eval_dataset2 is not None else derived_dataset2
        self.model.eval()
        with torch.no_grad():
            pairs = list(self.blocking.candidate_pairs(dataset1, dataset2))
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


def _entities_and_ground_truth(
    pairs: list[tuple[Entity, Entity]],
) -> tuple[list[Entity], list[Entity], list[tuple[str, str]]]:
    dataset1 = list({entity1.id: entity1 for entity1, _ in pairs}.values())
    dataset2 = list({entity2.id: entity2 for _, entity2 in pairs}.values())
    ground_truth = [(entity1.id, entity2.id) for entity1, entity2 in pairs]
    return dataset1, dataset2, ground_truth

"""Unified training loop for trainable linking modules.

Supports bi-encoder/GNN contrastive training (InfoNCE or Margin Ranking
Loss) with hard negative sampling drawn from a
[BlockingStrategy][linkingtk.blocking.base.BlockingStrategy] via
[sample_hard_negatives][linkingtk.blocking.negative_sampling.sample_hard_negatives],
and optional LoRA adapters via PEFT (see
[TrainingArguments.use_peft][linkingtk.train.arguments.TrainingArguments]).

``model`` must satisfy [TrainableModel][linkingtk.train.trainer.TrainableModel]:
a ``torch.nn.Module`` exposing a differentiable ``encode(entities) ->
torch.Tensor`` method, one embedding row per entity, on the model's own
device. No linker in this repo implements that yet -- this class is
validated against a small in-repo toy encoder (see
``tests/train/test_trainer.py``); future bi-encoder/GNN linkers implement
``encode()`` to plug into it.
"""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import torch
import torch.nn.functional as functional

from linkingtk.blocking.base import BlockingStrategy
from linkingtk.blocking.exact import ExactMatch
from linkingtk.blocking.negative_sampling import sample_hard_negatives
from linkingtk.core.entity import Entity
from linkingtk.eval.evaluator import Evaluator
from linkingtk.train._optim import build_optimizer
from linkingtk.train._peft import apply_peft
from linkingtk.train.arguments import TrainingArguments
from linkingtk.utils.device import resolve_device

if TYPE_CHECKING:
    from linkingtk.eval.report import EvaluationReport

DEFAULT_BLOCKING = ExactMatch()


class TrainableModel(Protocol):
    """Structural type for the model [Trainer][linkingtk.train.trainer.Trainer] trains.

    Any ``torch.nn.Module`` exposing a differentiable ``encode`` method
    satisfies this -- gradients must flow from its output back to
    ``self.parameters()``, and it must return its output on the same
    device its parameters currently live on (``Trainer`` moves the model
    via ``.to(device)`` but never touches tensors itself).
    """

    def encode(self, entities: list[Entity]) -> torch.Tensor:
        """Return a ``(len(entities), dim)`` tensor of embeddings, one row per entity."""
        ...

    def parameters(self) -> Any: ...
    def to(self, device: Any) -> Any: ...
    def train(self, mode: bool = True) -> Any: ...
    def eval(self) -> Any: ...
    def state_dict(self) -> Any: ...


class Trainer:
    """Trains a [TrainableModel][linkingtk.train.trainer.TrainableModel] via contrastive learning.

    Args:
        model: The model to train.
        args: Hyperparameters and settings, see
            [TrainingArguments][linkingtk.train.arguments.TrainingArguments].
        train_data: ``(source_entity, target_entity)`` positive pairs.
        eval_data: Optional held-out positive pairs. If given, every
            epoch's end evaluates Hits@1/Hits@10/MRR over an exhaustive
            (not blocking-restricted) ranking of every eval entity, and
            appends the result to ``eval_history``.
        blocking: Strategy used to mine hard negatives from ``train_data``,
            via
            [sample_hard_negatives][linkingtk.blocking.negative_sampling.sample_hard_negatives].
            Defaults to [ExactMatch][linkingtk.blocking.exact.ExactMatch],
            matching every other linker in this repo -- pass something
            tolerant of partial overlap (e.g.
            [LabelOverlap][linkingtk.blocking.label_overlap.LabelOverlap])
            for it to find anything beyond ``train_data``'s own true pairs.

    Attributes:
        eval_history: One [EvaluationReport][linkingtk.eval.report.EvaluationReport]
            per epoch, appended in ``train()`` when ``eval_data`` is given.
    """

    def __init__(
        self,
        model: TrainableModel,
        args: TrainingArguments,
        train_data: list[tuple[Entity, Entity]],
        eval_data: list[tuple[Entity, Entity]] | None = None,
        blocking: BlockingStrategy = DEFAULT_BLOCKING,
    ) -> None:
        self.model = model
        self.args = args
        self.train_data = train_data
        self.eval_data = eval_data
        self.blocking = blocking
        self.eval_history: list[EvaluationReport] = []

    def train(self) -> None:
        """Run the training loop, updating ``self.model`` in place."""
        args = self.args
        device = resolve_device(args.device)
        self.model.to(device)

        self.model = apply_peft(self.model, args)

        dataset1, dataset2, ground_truth = _entities_and_ground_truth(self.train_data)
        negatives_by_source = _group_by_source(
            sample_hard_negatives(
                dataset1, dataset2, ground_truth, self.blocking, top_k=args.negative_samples_ratio
            )
        )

        num_batches_per_epoch = -(-len(self.train_data) // args.batch_size)
        optimizer, scheduler = build_optimizer(
            self.model, args, num_batches_per_epoch * args.num_epochs
        )

        for _ in range(args.num_epochs):
            shuffled = list(self.train_data)
            random.shuffle(shuffled)
            for start in range(0, len(shuffled), args.batch_size):
                batch = shuffled[start : start + args.batch_size]
                loss = self._batch_loss(batch, negatives_by_source)
                if loss is None:
                    continue
                optimizer.zero_grad()
                loss.backward()  # type: ignore[no-untyped-call]
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            if self.eval_data is not None:
                self.eval_history.append(self._evaluate())

        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), Path(args.output_dir) / "model.pt")

    def _batch_loss(
        self,
        batch: list[tuple[Entity, Entity]],
        negatives_by_source: dict[str, list[Entity]],
    ) -> torch.Tensor | None:
        anchors = [entity1 for entity1, _ in batch]
        positives = [entity2 for _, entity2 in batch]
        anchor_emb = self.model.encode(anchors)
        positive_emb = self.model.encode(positives)
        negative_embs = self._encode_negatives(anchors, negatives_by_source, anchor_emb.shape[-1])

        if self.args.loss == "margin":
            return _margin_loss(anchor_emb, positive_emb, negative_embs, self.args.margin)
        return _info_nce_loss(anchor_emb, positive_emb, negative_embs, self.args.temperature)

    def _encode_negatives(
        self,
        anchors: list[Entity],
        negatives_by_source: dict[str, list[Entity]],
        embedding_dim: int,
    ) -> list[torch.Tensor]:
        per_anchor_negatives = [negatives_by_source.get(anchor.id, []) for anchor in anchors]
        flat = [entity for negatives in per_anchor_negatives for entity in negatives]
        if not flat:
            device = next(self.model.parameters()).device
            return [torch.empty(0, embedding_dim, device=device) for _ in anchors]
        flat_emb = self.model.encode(flat)
        counts = [len(negatives) for negatives in per_anchor_negatives]
        return list(torch.split(flat_emb, counts))

    def _evaluate(self) -> EvaluationReport:
        assert self.eval_data is not None  # only called when eval_data is set
        dataset1, dataset2, ground_truth = _entities_and_ground_truth(self.eval_data)
        self.model.eval()
        with torch.no_grad():
            emb1 = self.model.encode(dataset1)
            emb2 = self.model.encode(dataset2)
        self.model.train()

        similarities = emb1 @ emb2.T
        ranked_predictions = [
            (entity1.id, [dataset2[j].id for j in order.tolist()])
            for entity1, order in zip(
                dataset1, torch.argsort(similarities, dim=1, descending=True), strict=True
            )
        ]
        return Evaluator.evaluate_ranked(ranked_predictions, ground_truth, top_k=[1, 10])


def _entities_and_ground_truth(
    pairs: list[tuple[Entity, Entity]],
) -> tuple[list[Entity], list[Entity], list[tuple[str, str]]]:
    dataset1 = list({entity1.id: entity1 for entity1, _ in pairs}.values())
    dataset2 = list({entity2.id: entity2 for _, entity2 in pairs}.values())
    ground_truth = [(entity1.id, entity2.id) for entity1, entity2 in pairs]
    return dataset1, dataset2, ground_truth


def _group_by_source(pairs: list[tuple[Entity, Entity]]) -> dict[str, list[Entity]]:
    grouped: dict[str, list[Entity]] = defaultdict(list)
    for entity1, entity2 in pairs:
        grouped[entity1.id].append(entity2)
    return grouped


def _info_nce_loss(
    anchor_emb: torch.Tensor,
    positive_emb: torch.Tensor,
    negative_embs: list[torch.Tensor],
    temperature: float,
) -> torch.Tensor:
    """In-batch positives-of-other-anchors, plus each anchor's own mined negatives."""
    # (B, B); row i, column j = sim(anchor_i, positive_j) -- column i is anchor_i's own positive.
    in_batch_sims = anchor_emb @ positive_emb.T / temperature
    losses = []
    for i in range(anchor_emb.shape[0]):
        logits = in_batch_sims[i]
        if negative_embs[i].shape[0] > 0:
            hard_sims = (anchor_emb[i] @ negative_embs[i].T) / temperature
            logits = torch.cat([logits, hard_sims])
        target = torch.tensor(i, device=logits.device)
        losses.append(functional.cross_entropy(logits, target))
    return torch.stack(losses).mean()


def _margin_loss(
    anchor_emb: torch.Tensor,
    positive_emb: torch.Tensor,
    negative_embs: list[torch.Tensor],
    margin: float,
) -> torch.Tensor | None:
    """``max(0, margin - (sim(anchor, positive) - sim(anchor, negative)))`` per mined negative."""
    losses = []
    for i in range(anchor_emb.shape[0]):
        if negative_embs[i].shape[0] == 0:
            continue
        positive_sim = (anchor_emb[i] @ positive_emb[i]).expand(negative_embs[i].shape[0])
        negative_sim = anchor_emb[i] @ negative_embs[i].T
        target = torch.ones_like(positive_sim)
        losses.append(
            functional.margin_ranking_loss(positive_sim, negative_sim, target, margin=margin)
        )
    if not losses:
        return None
    return torch.stack(losses).mean()

"""Unified training loop for trainable linking modules.

Not yet implemented (see DESIGN.md Milestones, Phase 4). Intended to support
bi-encoder/contrastive training (InfoNCE or Margin Ranking Loss) with hard
negative sampling drawn from a :class:`~linkingtk.blocking.base.BlockingStrategy`.
"""

from __future__ import annotations

from typing import Any

from linkingtk.blocking.base import BlockingStrategy
from linkingtk.blocking.exact import ExactMatch
from linkingtk.core.entity import Entity
from linkingtk.train.arguments import TrainingArguments

DEFAULT_BLOCKING = ExactMatch()


class Trainer:
    """Trains a trainable linker model (e.g. a bi-encoder or GNN)."""

    def __init__(
        self,
        model: Any,
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

    def train(self) -> None:
        """Run the training loop."""
        raise NotImplementedError("Trainer.train is not yet implemented.")

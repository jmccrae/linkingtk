"""Configuration for [linkingtk.train.trainer.Trainer][] runs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainingArguments:
    """Hyperparameters and settings for a training run.

    Attributes:
        output_dir: Directory where checkpoints and the final model are
            written.
        learning_rate: Optimizer learning rate.
        negative_samples_ratio: Number of hard negatives sampled per
            positive pair during batch generation.
        use_peft: Whether to wrap the model with LoRA adapters via PEFT.
        num_epochs: Number of passes over the training data.
        batch_size: Number of entity pairs per batch.
    """

    output_dir: str
    learning_rate: float = 3e-5
    negative_samples_ratio: int = 5
    use_peft: bool = False
    num_epochs: int = 3
    batch_size: int = 32

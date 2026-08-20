"""Configuration for [Trainer][linkingtk.train.trainer.Trainer] and
[CrossEncoderTrainer][linkingtk.train.cross_encoder.CrossEncoderTrainer] runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


@dataclass
class TrainingArguments:
    """Hyperparameters and settings for a training run.

    Attributes:
        output_dir: Directory where the trained model's ``state_dict`` is
            written.
        learning_rate: Optimizer learning rate.
        negative_samples_ratio: Number of hard negatives sampled per
            positive pair during batch generation, via
            [sample_hard_negatives][linkingtk.blocking.negative_sampling.sample_hard_negatives].
        use_peft: Whether to wrap the model with LoRA adapters via PEFT.
            Requires ``peft_config`` to be set.
        peft_config: A ``peft.LoraConfig`` describing which of the model's
            layers to adapt (e.g. ``target_modules``). Ignored unless
            ``use_peft`` is `True`, but required in that case -- there's
            no safe universal default, since ``target_modules`` names
            differ per model architecture (e.g. ``["query", "value"]`` vs
            ``["q_proj", "v_proj"]``); a wrong guess would silently adapt
            the wrong (or no) layers. Typed loosely (``Any``) to avoid a
            hard import of ``peft`` at the dataclass-definition level.
        num_epochs: Number of passes over the training data.
        batch_size: Number of entity pairs per batch.
        loss: Contrastive objective for ``Trainer`` (ignored by
            ``CrossEncoderTrainer``, which always trains with binary
            cross-entropy -- there's no separately-embedded pair of
            vectors to contrast). ``"infonce"`` (default) treats every
            other pair's target in the batch, plus each anchor's own
            mined hard negatives, as negatives and applies a
            softmax-temperature cross-entropy loss. ``"margin"`` applies
            ``torch.nn.MarginRankingLoss`` between each anchor's positive
            and each of its mined hard negatives.
        temperature: Softmax temperature for the ``"infonce"`` loss.
            Ignored by ``CrossEncoderTrainer``.
        margin: Margin for the ``"margin"`` loss. Ignored by
            ``CrossEncoderTrainer``.
        device: Torch device to train on, e.g. ``"cpu"`` (default) or
            ``"cuda"``. Resolved via
            [resolve_device][linkingtk.utils.device.resolve_device].
        weight_decay: AdamW weight decay, applied to every parameter
            except bias and LayerNorm weights (standard BERT fine-tuning
            convention). Defaults to ``0.01``, ``torch.optim.AdamW``'s own
            default -- preserves ``Trainer``'s original behavior, which
            relied on that implicit default rather than setting it
            explicitly.
        warmup_ratio: Fraction of total optimizer steps spent linearly
            warming up the learning rate before linearly decaying it back
            to zero, via ``transformers.get_linear_schedule_with_warmup``.
            ``0.0`` (the default) skips the scheduler entirely -- a flat
            learning rate for the whole run, every existing caller's
            behavior before this field existed.
    """

    output_dir: str
    learning_rate: float = 3e-5
    negative_samples_ratio: int = 5
    use_peft: bool = False
    peft_config: Any | None = None
    num_epochs: int = 3
    batch_size: int = 32
    loss: Literal["infonce", "margin"] = "infonce"
    temperature: float = 0.05
    margin: float = 1.0
    device: str = "cpu"
    weight_decay: float = 0.01
    warmup_ratio: float = 0.0

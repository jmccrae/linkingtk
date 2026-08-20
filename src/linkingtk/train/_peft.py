"""Shared PEFT/LoRA wrapping for [Trainer][linkingtk.train.trainer.Trainer] and
[CrossEncoderTrainer][linkingtk.train.cross_encoder.CrossEncoderTrainer].
"""

from __future__ import annotations

from typing import Any

from linkingtk.exceptions import LinkingTKError
from linkingtk.train.arguments import TrainingArguments


def apply_peft(model: Any, args: TrainingArguments) -> Any:
    """Wrap `model` with LoRA adapters via PEFT, if `args.use_peft`.

    A no-op passthrough when `args.use_peft` is `False`.

    Raises:
        LinkingTKError: If `args.use_peft` is `True` but `args.peft_config`
            is unset -- there's no safe universal default for LoRA's
            `target_modules`, since it differs per model architecture.
    """
    if not args.use_peft:
        return model
    if args.peft_config is None:
        raise LinkingTKError(
            "TrainingArguments.use_peft=True requires peft_config to be set "
            "-- there's no safe universal default for LoRA's target_modules, "
            "since it differs per model architecture."
        )
    from peft import get_peft_model

    return get_peft_model(model, args.peft_config)

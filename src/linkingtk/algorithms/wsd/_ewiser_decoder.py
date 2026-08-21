"""The trainable EWISER decoder head, and released-checkpoint loading.

Split out of `ewiser.py` to keep that module focused on the public
`EwiserEncoder`/`EwiserLinker` API (see this package's ~300-line-per-file
convention).
"""

from __future__ import annotations

import pickle
import types
from pathlib import Path
from typing import Any

import torch
from torch import nn

from linkingtk.algorithms.wsd._ewiser_structured_logits import StructuredLogits


class EwiserDecoder(nn.Module):
    """The small trainable head: ``BatchNorm -> Linear -> swish -> Linear`` + graph conv.

    Submodule names (`norm`, `linears`, `logits`, `structured_logits`)
    match the released checkpoints' own state dict keys exactly (confirmed
    directly against a real checkpoint file), so a checkpoint's decoder
    weights load via plain ``load_state_dict(strict=True)``.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        vocab_size: int,
        adjacency: torch.Tensor | None = None,
        structured_logits_trainable: bool = True,
        structured_logits_renormalize: bool = False,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm = nn.BatchNorm1d(input_dim)
        self.linears = nn.ModuleList([nn.Linear(input_dim, hidden_dim)])
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.SiLU()  # swish(x) = x * sigmoid(x), confirmed against the reference
        self.logits = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.structured_logits = (
            StructuredLogits(
                adjacency,
                trainable=structured_logits_trainable,
                renormalize=structured_logits_renormalize,
            )
            if adjacency is not None
            else None
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        shape = hidden.shape
        out = self.norm(hidden.reshape(-1, shape[-1])).reshape(shape)
        out = self.linears[0](out)
        out = self.dropout(out)
        out = self.activation(out)
        raw_logits: torch.Tensor = self.logits(out)
        if self.structured_logits is not None:
            propagated: torch.Tensor = self.structured_logits(raw_logits)
            return propagated
        return raw_logits


def load_fairseq_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load a released EWISER ``.pt`` checkpoint file.

    Its pickle stream references training-only bookkeeping classes from
    its original (pre-release) package name, ``qbert``, and from
    ``fairseq`` (e.g. ``qbert.fairseq_ext.meters.SumMeter``,
    ``fairseq.meters.AverageMeter``) -- none of which this package depends
    on or needs (only `data["model"]`'s plain tensors and `data["args"]`'s
    plain `argparse.Namespace` are used), but standard unpickling fails the
    *entire* load if even one referenced class can't be resolved. Works
    around this with a permissive `Unpickler.find_class` that substitutes
    an inert placeholder class for anything unresolvable instead of
    failing -- confirmed safe by inspecting exactly which classes get
    requested against a real checkpoint file (only optimizer/meter
    training-history objects, never anything under `data["model"]`).
    """

    class _PermissiveUnpickler(pickle.Unpickler):
        def find_class(self, module: str, name: str) -> Any:
            try:
                return super().find_class(module, name)
            except Exception:
                return type(name, (), {})

    pickle_module = types.ModuleType("_ewiser_permissive_pickle")
    pickle_module.Unpickler = _PermissiveUnpickler  # type: ignore[attr-defined]
    pickle_module.load = pickle.load  # type: ignore[attr-defined]

    data: dict[str, Any] = torch.load(
        str(path), map_location="cpu", weights_only=False, pickle_module=pickle_module
    )
    return data

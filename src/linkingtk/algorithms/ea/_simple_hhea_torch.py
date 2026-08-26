"""Time2Vec + fusion model for
[SimpleHHEALinker][linkingtk.algorithms.ea.simple_hhea.SimpleHHEALinker].

Ports `model.py` from https://github.com/DataArcTech/Simple-HHEA: a small
MLP that concatenates (name, time, structure) feature groups -- each
already computed elsewhere (`_simple_hhea_name.py`,
`_simple_hhea_structure.py`, `build_time_histogram` below) -- into one
final per-entity embedding. Name/time/structure embeddings are held as
fixed buffers (not trained); only the small linear/`Time2Vec` layers on
top are trainable, matching the reference's plain (non-`nn.Parameter`)
tensor attributes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import nn

if TYPE_CHECKING:
    import numpy.typing as npt

TIME_SPAN = 1 + 27 * 13
"""Number of month-bins: 1 catch-all pre-1995 bin, plus 27 years * 13
(the reference's own off-by-one month indexing, see `_time_bin`) monthly
bins -- matches the reference's own `1+27*13`, wide enough for ICEWS data
(1995-2021ish)."""


def _time_bin(year: int, month: int) -> int:
    """Month bin index for a resolved ``(year, month)``, per `rel_time_cal`."""
    return (year - 1995) * 13 + month + 1


def _parse_time_label(label: str | None) -> tuple[int, int] | None:
    if label is None:
        return None
    parts = label.split("-")
    if len(parts) != 2:
        return None
    try:
        year, month = int(parts[0]), int(parts[1])
    except ValueError:
        return None
    return year, month


def build_time_histogram(
    entity_ids: list[str], temporal_triples: list[tuple[str, str, str, str | None, str | None]]
) -> npt.NDArray[np.floating[Any]]:
    """Per-entity month-activity histogram, ``(len(entity_ids), TIME_SPAN)``.

    For each ``(subject, relation, object, start_label, end_label)``
    temporal triple, increments the *subject*'s histogram at every month
    bin in ``[start, end]`` (inclusive) if both resolve; a single bin if
    only one resolves; skipped if neither does. A resolved year before
    1995 (ICEWS's real lower bound) falls into bin ``0``, the catch-all
    bin. One symmetric rule for every triple -- unlike the reference's own
    `load_ent_time_matrix`, whose two KG-side branches diverge (one
    increments two independent point-bins, the other fills a range) and
    whose range branch reads a stale `time_y` left over from an unrelated
    earlier loop (a genuine bug, not a deliberate per-side design choice
    -- confirmed by reading `utils.py` directly). Not ported here.
    """
    index = {entity_id: row for row, entity_id in enumerate(entity_ids)}
    histogram = np.zeros((len(entity_ids), TIME_SPAN), dtype=np.float64)
    for subject, _relation, _obj, start_label, end_label in temporal_triples:
        row = index.get(subject)
        if row is None:
            continue
        start = _parse_time_label(start_label)
        end = _parse_time_label(end_label)
        if start is None and end is None:
            continue
        start = start or end
        end = end or start
        assert start is not None and end is not None
        start_bin = 0 if start[0] < 1995 else _time_bin(*start)
        end_bin = 0 if end[0] < 1995 else _time_bin(*end)
        if start_bin > end_bin:
            start_bin, end_bin = end_bin, start_bin
        histogram[row, start_bin : end_bin + 1] += 1
    return histogram


class CosineActivation(nn.Module):
    """Time2Vec's periodic + linear feature bank (Kazemi et al. 2019)."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.w0 = nn.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.Parameter(torch.randn(1))
        self.w = nn.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.Parameter(torch.randn(out_features - 1))

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        periodic = torch.cos(torch.matmul(tau, self.w) + self.b)
        linear = torch.matmul(tau, self.w0) + self.b0
        return torch.cat([periodic, linear], dim=1)


class Time2Vec(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.activation = CosineActivation(1, hidden_dim)
        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        result: torch.Tensor = self.fc(self.activation(tau))
        return result


class SimpleHHEAModel(nn.Module):
    """Fuses name/time/structure feature groups into one final embedding per entity.

    Args:
        name_emb: ``(num_entities, name_dim)`` raw (whitened) name
            embeddings, fixed.
        time_emb: ``(num_entities, TIME_SPAN)`` month-activity histogram
            (see
            [build_time_histogram][linkingtk.algorithms.ea._simple_hhea_torch.build_time_histogram]),
            fixed. ``None`` disables the time branch.
        structure_emb: ``(num_entities, structure_dim)`` node2vec
            embeddings, fixed. ``None`` disables the structure branch.
        emb_size: Final output embedding dimension.
        structure_size: Post-projection size of the structure branch
            before concatenation.
        time_size: Post-projection size of the time branch before
            concatenation.
        device: Where to place the fixed feature tensors -- plain
            attributes (matching the reference), not registered buffers,
            so they don't move with a later `.to(device)` call on the
            model; set correctly up front instead.
    """

    def __init__(
        self,
        name_emb: npt.NDArray[np.floating[Any]],
        time_emb: npt.NDArray[np.floating[Any]] | None,
        structure_emb: npt.NDArray[np.floating[Any]] | None,
        emb_size: int = 64,
        structure_size: int = 8,
        time_size: int = 8,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        self.use_time = time_emb is not None
        self.use_structure = structure_emb is not None

        linear_size = emb_size
        if self.use_time:
            linear_size += time_size
            assert time_emb is not None
            self.time_emb: torch.Tensor = torch.tensor(time_emb).float().to(device)
            self.fc_time_0 = nn.Linear(32, 32)
            self.fc_time = nn.Linear(32, time_size)
            self.time2vec = Time2Vec(hidden_dim=32)
            self.time_span_index: torch.Tensor = (
                torch.arange(TIME_SPAN).unsqueeze(1).float().to(device)
            )

        if self.use_structure:
            linear_size += structure_size
            assert structure_emb is not None
            self.structure_emb: torch.Tensor = torch.tensor(structure_emb).float().to(device)
            self.fc_structure_0 = nn.Linear(structure_emb.shape[-1], emb_size)
            self.fc_structure = nn.Linear(emb_size, structure_size)

        self.name_emb: torch.Tensor = torch.tensor(name_emb).float().to(device)
        self.fc_name_0 = nn.Linear(name_emb.shape[-1], emb_size)
        self.fc_name = nn.Linear(emb_size, emb_size)

        self.fc_final = nn.Linear(linear_size, emb_size)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self) -> torch.Tensor:
        features = [self.fc_name(self.fc_name_0(self.dropout(self.name_emb)))]

        if self.use_time:
            time_span_feature = self.time2vec(self.time_span_index)
            time_feature = torch.mm(self.time_emb, time_span_feature) / TIME_SPAN
            features.append(self.fc_time(self.fc_time_0(self.dropout(time_feature))))

        if self.use_structure:
            structure_feature = self.fc_structure_0(self.dropout(self.structure_emb))
            features.append(self.fc_structure(structure_feature))

        result: torch.Tensor = self.fc_final(torch.cat(features, dim=1))
        return result

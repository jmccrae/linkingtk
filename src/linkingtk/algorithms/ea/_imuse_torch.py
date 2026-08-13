"""Torch-touching training-step helpers for
[IMUSELinker][linkingtk.algorithms.ea.imuse.IMUSELinker].

Separated from ``_imuse_text.py`` (which stays plain-python/numpy and
independently testable without ``torch`` installed) because this function
builds/consumes PyTorch tensors directly. Callers must already have
confirmed ``torch`` is importable (``imuse.py``'s ``fit()`` does this via
``OptionalDependencyError`` before this runs) -- that's why this imports
``torch`` unconditionally rather than guarding again.

IMUSE's structural (SE) half is identical in shape to every other linker
in this family's own SE half, so it reuses
[KGContext][linkingtk.algorithms.ea._kdcoe_torch.KGContext],
[build_kg_context][linkingtk.algorithms.ea._kdcoe_torch.build_kg_context]
and
[train_structural_epoch][linkingtk.algorithms.ea._kdcoe_torch.train_structural_epoch]
directly rather than reimplementing a sixth copy, and
[validation_hits1][linkingtk.algorithms.ea._iptranse_torch.validation_hits1]
for early stopping (no mapping-matrix projection needed -- IMUSE has no
mapping matrix, see ``train_align_epoch`` below). Only the alignment loss
itself is new.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt
    import torch


def train_align_epoch(
    entity_embeds: torch.nn.Parameter,
    optimizer: torch.optim.Optimizer,
    source_ids: npt.NDArray[np.int64],
    target_ids: npt.NDArray[np.int64],
    steps: int,
) -> None:
    """One epoch of squared-distance alignment-loss training.

    Mirrors OpenEA's ``launch_align_training_1epo``: unlike every other
    linker here, IMUSE has no mapping matrix and no shared-id merge --
    alignment is a direct ``sum(||ent_embeds[e1] - ent_embeds[e2]||^2)``
    pull between the two (still separate-row) entity embeddings of each
    bootstrapped pair. OpenEA re-runs this over the **full** bootstrapped-
    pair list ``steps = ceil(len(pairs)/batch_size)`` times per epoch --
    the pairs themselves are never actually sliced/batched -- the same
    full-list-re-run quirk
    [AttrELinker][linkingtk.algorithms.ea.attre.AttrELinker]'s
    ``train_joint_epoch`` and
    [KDCoELinker][linkingtk.algorithms.ea.kdcoe.KDCoELinker]'s
    ``train_mapping_epoch`` already document and port literally; same
    treatment here. Both lookups are L2-normalized on every forward pass
    (``ent_l2_norm: true`` in OpenEA's config), matching this family's
    established convention. No-ops if ``source_ids`` is empty.
    """
    import torch
    import torch.nn.functional as functional

    if len(source_ids) == 0:
        return
    source_t = torch.from_numpy(source_ids).long()
    target_t = torch.from_numpy(target_ids).long()
    for _ in range(steps):
        s = functional.normalize(entity_embeds[source_t], dim=1)
        t = functional.normalize(entity_embeds[target_t], dim=1)
        loss = ((s - t) ** 2).sum()
        optimizer.zero_grad()
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()

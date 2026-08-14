"""Shared device-resolution helper for every hand-rolled-PyTorch EA linker.

[KGELinker][linkingtk.algorithms.ea.kge.KGELinker] is excluded -- it's
pykeen-backed, and pykeen's own trainer already handles device placement.
Every other linker in ``linkingtk.algorithms.ea`` (``MTransELinker``,
``IPTransELinker``, ``JAPELinker``, ``KDCoELinker``, ``AttrELinker``,
``IMUSELinker``, ``MultiKELinker``) hand-rolls its own PyTorch training loop
and calls [resolve_device][linkingtk.algorithms.ea._device.resolve_device]
once at the top of ``fit()``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from linkingtk.exceptions import LinkingTKError

if TYPE_CHECKING:
    import torch


def resolve_device(device: str) -> torch.device:
    """Resolve a device string (e.g. ``"cpu"``, ``"cuda"``, ``"cuda:0"``) to a `torch.device`.

    Args:
        device: A device string acceptable to `torch.device`.

    Returns:
        The resolved `torch.device`.

    Raises:
        LinkingTKError: If ``device`` isn't a valid device string, or
            requests a CUDA device while `torch.cuda.is_available()` is
            `False`. Deliberately doesn't fall back to CPU silently --
            that would hide a real misconfiguration.
    """
    import torch

    try:
        resolved = torch.device(device)
    except RuntimeError as exc:
        raise LinkingTKError(f"Invalid device {device!r}: {exc}") from exc

    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise LinkingTKError(
            f"device={device!r} requested but torch.cuda.is_available() is False "
            "-- no CUDA device is visible to torch in this environment."
        )

    return resolved

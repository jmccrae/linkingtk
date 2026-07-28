"""Base exception types for LinkingTK."""

from __future__ import annotations


class LinkingTKError(Exception):
    """Base class for all exceptions raised by LinkingTK."""


class OptionalDependencyError(LinkingTKError):
    """Raised when an optional feature is used without its dependency installed."""

    def __init__(self, feature: str, extra: str) -> None:
        super().__init__(
            f"'{feature}' requires an optional dependency. "
            f"Install it via 'pip install linkingtk[{extra}]'."
        )

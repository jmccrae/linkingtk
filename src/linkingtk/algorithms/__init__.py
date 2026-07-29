"""Task-specific linking algorithms built on top of :class:`BaseLinker`."""

from linkingtk.algorithms.base import BaseLinker
from linkingtk.algorithms.string_similarity import StringSimilarityLinker

__all__ = ["BaseLinker", "StringSimilarityLinker"]

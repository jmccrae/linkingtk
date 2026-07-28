"""Blocking strategies used to generate candidate entity pairs."""

from linkingtk.blocking.base import BlockingStrategy
from linkingtk.blocking.embedding import EmbeddingSimilarityBlocker
from linkingtk.blocking.exact import ExactMatch
from linkingtk.blocking.label_overlap import LabelOverlap

__all__ = [
    "BlockingStrategy",
    "ExactMatch",
    "LabelOverlap",
    "EmbeddingSimilarityBlocker",
]

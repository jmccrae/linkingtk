"""Blocking strategies used to generate candidate entity pairs."""

from linkingtk.blocking.base import BlockingStrategy
from linkingtk.blocking.embedding import EmbeddingSimilarityBlocker
from linkingtk.blocking.exact import ExactMatch
from linkingtk.blocking.label_overlap import LabelOverlap
from linkingtk.blocking.negative_sampling import sample_hard_negatives

__all__ = [
    "BlockingStrategy",
    "ExactMatch",
    "LabelOverlap",
    "EmbeddingSimilarityBlocker",
    "sample_hard_negatives",
]

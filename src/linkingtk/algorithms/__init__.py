"""Task-specific linking algorithms built on top of
[BaseLinker][linkingtk.algorithms.base.BaseLinker]."""

from linkingtk.algorithms.base import BaseLinker
from linkingtk.algorithms.feature_classifier import FeatureClassifierLinker
from linkingtk.algorithms.matching import GreedyMatcher, Matcher, OptimalMatcher
from linkingtk.algorithms.string_similarity import StringSimilarityLinker

__all__ = [
    "BaseLinker",
    "FeatureClassifierLinker",
    "GreedyMatcher",
    "Matcher",
    "OptimalMatcher",
    "StringSimilarityLinker",
]

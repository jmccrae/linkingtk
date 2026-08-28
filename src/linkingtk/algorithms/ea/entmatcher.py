"""EntMatcher-style Entity Alignment linker.

References:
    EntMatcher: An Open-Source Library for Entity Alignment via Feature
    Matching in Knowledge Graphs. https://github.com/DexterZeng/EntMatcher
    (research repo, not a dependency of this project — see
    [linkingtk.algorithms.feature_classifier][] for what's reused).
"""

from __future__ import annotations

from linkingtk.algorithms.feature_classifier import FeatureClassifierLinker
from linkingtk.matchers import OptimalMatcher


class EntMatcherLinker(FeatureClassifierLinker):
    """EA linker combining hand-crafted similarity features with optimal one-to-one matching.

    A preconfigured
    [FeatureClassifierLinker][linkingtk.algorithms.feature_classifier.FeatureClassifierLinker]
    (``matching=OptimalMatcher()``) — see that class for the feature set,
    training, and matching behavior. Kept as its own named class since it
    maps to a recognizable reference (EntMatcher's key insight: a globally
    optimal assignment can outperform independent per-source matching).
    """

    def __init__(self) -> None:
        super().__init__(matching=OptimalMatcher())

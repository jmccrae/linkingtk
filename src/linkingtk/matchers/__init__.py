"""Matching strategies that turn scored candidate pairs into final links.

Complements [BlockingStrategy][linkingtk.blocking.base.BlockingStrategy] ("how do we
generate candidates") with the question of how to resolve them into final
links once scored — independent per-source argmax, a globally optimal
assignment, or (in principle, not yet implemented) something that isn't
one-to-one at all, like hierarchical broader/narrower relations.
"""

from linkingtk.matchers.base import Matcher
from linkingtk.matchers.greedy import GreedyMatcher
from linkingtk.matchers.optimal import OptimalMatcher

DEFAULT_MATCHER = GreedyMatcher()

__all__ = [
    "Matcher",
    "GreedyMatcher",
    "OptimalMatcher",
    "DEFAULT_MATCHER",
]

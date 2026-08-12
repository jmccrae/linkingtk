"""Entity Alignment (EA) algorithms.

EA finds matches between two knowledge graphs, often across languages.
Entities typically carry labels and descriptions but rarely a context, and
a supporting graph is normally available. See DESIGN.md for references
(OpenEA, Simple-HHEA, ChatEA, ProLEA, EntMatcher).
"""

from linkingtk.algorithms.ea.entmatcher import EntMatcherLinker
from linkingtk.algorithms.ea.iptranse import IPTransELinker
from linkingtk.algorithms.ea.kge import KGELinker
from linkingtk.algorithms.ea.mtranse import MTransELinker

__all__ = ["EntMatcherLinker", "IPTransELinker", "KGELinker", "MTransELinker"]

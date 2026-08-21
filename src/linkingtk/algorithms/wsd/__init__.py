"""Word Sense Disambiguation (WSD) algorithms.

WSD identifies the relevant dictionary sense for a mention in context.
Similar to EL, but the target dataset is a dictionary rather than a KB.
See DESIGN.md for references (Lesk, EWISER, GlossBERT, ESC).
"""

from linkingtk.algorithms.wsd.ewiser import EwiserEncoder, EwiserLinker
from linkingtk.algorithms.wsd.glossbert import GlossBertEncoder, GlossBertLinker
from linkingtk.algorithms.wsd.lesk import LeskLinker

__all__ = [
    "LeskLinker",
    "GlossBertEncoder",
    "GlossBertLinker",
    "EwiserEncoder",
    "EwiserLinker",
]

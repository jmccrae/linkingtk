"""Entity Linking (EL) algorithms.

EL links named-entity mentions in text (with context, no description) to
entries in a knowledge base (with description, no context). Related
systems: spaCy's EntityLinker, ReFinED, BLINK.
"""

from linkingtk.algorithms.el.blink import BlinkEncoder, BlinkLinker
from linkingtk.algorithms.el.refined import ReFinEDEncoder, ReFinEDLinker

__all__ = [
    "BlinkEncoder",
    "BlinkLinker",
    "ReFinEDEncoder",
    "ReFinEDLinker",
]

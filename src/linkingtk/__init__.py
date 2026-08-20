"""LinkingTK: a unified toolkit for Entity Alignment, Entity Linking,
Word Sense Disambiguation and Word Sense Alignment.
"""

from linkingtk.core.entity import Entity
from linkingtk.core.result import AlignmentResult
from linkingtk.core.source import CachingEntitySource, EntitySource
from linkingtk.exceptions import LinkingTKError

__version__ = "0.1.0"

__all__ = [
    "Entity",
    "AlignmentResult",
    "EntitySource",
    "CachingEntitySource",
    "LinkingTKError",
    "__version__",
]

"""Lesk-style Word Sense Disambiguation linker.

References:
    Lesk, M. (1986). Automatic sense disambiguation using machine readable
    dictionaries: how to tell a pine cone from an ice cream cone. In
    Proceedings of the 5th annual international conference on Systems
    documentation (SIGDOC '86).
"""

from __future__ import annotations

from linkingtk.algorithms.string_similarity import StringSimilarityLinker


class LeskLinker(StringSimilarityLinker):
    """WSD linker that scores candidate senses by context/gloss word overlap.

    A preconfigured :class:`~linkingtk.algorithms.string_similarity.StringSimilarityLinker`
    (``source_field="context"``, ``target_field="description"``,
    ``metric="word_overlap"``) — see that class for the ranking and
    ``alternatives`` behavior. Kept as its own named class since Lesk is a
    well-known algorithm in its own right.
    """

    def __init__(self) -> None:
        super().__init__(source_field="context", target_field="description", metric="word_overlap")

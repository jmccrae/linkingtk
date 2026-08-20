"""EntitySource wrapper for WordNet via the `wn` library.

Wraps [wn](https://github.com/goodmami/wn) -- a query-driven interface to
locally-installed wordnets (e.g. Open English WordNet) -- as an
[EntitySource][linkingtk.core.source.EntitySource], so WSD/WSA can target a
full dictionary without materializing every synset as an ``Entity`` up
front. ``wn`` itself still requires a lexicon to have been downloaded once
(``python -m wn download oewn:2021``, or ``wn.download("oewn:2021")``) --
this module only avoids loading its senses into memory ahead of time, not
the download step.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from linkingtk.core.entity import Entity, LabelWithLang
from linkingtk.core.source import EntitySource
from linkingtk.exceptions import OptionalDependencyError

if TYPE_CHECKING:
    import wn as wn_module


class WnEntitySource(EntitySource):
    """Queries a locally-installed `wn` lexicon as an
    [EntitySource][linkingtk.core.source.EntitySource].

    See the module docstring for the one-time lexicon download `wn`
    itself requires.

    Args:
        lexicon: A `wn` lexicon specifier (e.g. the default
            ``"oewn:2021"``, Open English WordNet 2021). Passed straight
            through to ``wn.synsets``/``wn.synset``.
        lang: Restrict to a single BCP-47 language code (e.g. ``"en"``).
            Left as ``None`` (the default), an entity's labels carry a
            per-word language tag instead of a plain string, since a
            multilingual/multi-lexicon query can return words from more
            than one language on a single synset.

    Raises:
        OptionalDependencyError: If `wn` isn't installed.
    """

    def __init__(self, lexicon: str = "oewn:2021", lang: str | None = None) -> None:
        try:
            import wn
        except ImportError as exc:
            raise OptionalDependencyError("WnEntitySource", "wn") from exc
        self._wn = wn
        self.lexicon = lexicon
        self.lang = lang

    def search(self, query: str, top_k: int = 10) -> list[Entity]:
        """Return up to ``top_k`` synsets containing ``query`` as a lemma.

        Ordering follows ``wn.synsets``' own (the lexicon's stored sense
        order -- for Open English WordNet, most-frequent-sense first).
        """
        synsets = self._wn.synsets(query, lexicon=self.lexicon, lang=self.lang)
        return [self._to_entity(synset) for synset in synsets[:top_k]]

    def get(self, entity_id: str) -> Entity | None:
        """Look up a synset by its `wn` id, or ``None`` if it doesn't exist."""
        try:
            synset = self._wn.synset(entity_id, lexicon=self.lexicon, lang=self.lang)
        except self._wn.Error:
            return None
        return self._to_entity(synset)

    def _to_entity(self, synset: wn_module.Synset) -> Entity:
        return Entity(id=synset.id, labels=self._labels(synset), description=synset.definition())

    def _labels(self, synset: wn_module.Synset) -> list[str | LabelWithLang]:
        if self.lang is not None:
            return list(synset.lemmas())
        return [(word.lemma(), word.lexicon().language) for word in synset.words()]

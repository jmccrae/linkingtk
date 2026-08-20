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

# Multi-word lemmas (e.g. phrasal verbs) are joined with underscores in the
# classic WordNet distribution format (and so in every corpus in this
# package tagged with real sense keys -- e.g. "point_out%2:32:01::"), but
# OMW's WordNet-3.0-based lexicons (omw-en:1.4, omw-en:2.0) index the same
# lemma space-separated ("point out") -- confirmed directly:
# wn.synsets("point_out", ...) returns nothing there, wn.synsets("point
# out", ...) returns 3. Every function below that queries by lemma retries
# with underscores replaced by spaces if the first query comes up empty --
# without it, every phrasal-verb instance in a real corpus (~4-5% of
# SemEval-2007's) silently fails to resolve.


def _adjective_satellite_variant(sense_key: str) -> str | None:
    """Swap a sense key's part-of-speech digit between ``3`` (adjective) and
    ``5`` (adjective satellite), or ``None`` if `sense_key` is neither.

    A sense key's part of speech is a single digit right after ``%`` (1
    noun, 2 verb, 3 adjective, 4 adverb, 5 adjective satellite). Plain and
    satellite adjectives are conflated inconsistently across WordNet
    tools/distributions -- confirmed directly: UFSAC's own ``wn30_key``
    layer tags "peculiar" (meaning "specific") as
    ``"peculiar%3:00:00:specific:00"``, but that string matches no sense
    in ``omw-en:1.4``, which stores the identical sense as
    ``"peculiar%5:00:00:specific:00"`` instead. Without this fallback,
    every satellite-adjective instance in a real corpus (~7.5% of
    SensEval-2's) silently fails to resolve.
    """
    percent = sense_key.find("%")
    if percent == -1 or percent + 1 >= len(sense_key):
        return None
    pos_digit = sense_key[percent + 1]
    swapped = {"3": "5", "5": "3"}.get(pos_digit)
    if swapped is None:
        return None
    return sense_key[: percent + 1] + swapped + sense_key[percent + 2 :]


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
        if not synsets and "_" in query:
            synsets = self._wn.synsets(
                query.replace("_", " "), lexicon=self.lexicon, lang=self.lang
            )
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


def sensekey_to_synset_id(sense_key: str, lexicon: str = "omw-en:1.4") -> str | None:
    """Resolve a Princeton WordNet sense key (e.g. ``"group%1:03:00::"``) to its synset id.

    A sense key isn't a `wn` id by itself -- `wn` has no lookup-by-sense-key
    API, so this instead lists every sense of the key's lemma in `lexicon`
    and matches by each sense's own recorded ``identifier`` metadata (the
    sense key `wn` was built from). Also tries `sense_key`'s
    adjective/adjective-satellite variant (its part-of-speech digit
    swapped between ``3`` and ``5``) if the exact one doesn't match --
    without it, every satellite-adjective instance in a real corpus
    silently fails to resolve. Used by
    [UfsacDataset][linkingtk.datasets.ufsac.UfsacDataset], whose corpora
    are tagged with WordNet 3.0 sense keys rather than synset ids.

    Args:
        sense_key: A WordNet sense key, e.g. ``"group%1:03:00::"``.
        lexicon: A `wn` lexicon specifier whose senses carry the original
            sense key as their ``identifier`` metadata -- true of
            ``"omw-en:1.4"`` (the default, OMW's English WordNet based on
            WordNet 3.0) but not of every lexicon.

    Returns:
        The matching synset's `wn` id, or ``None`` if no sense in `lexicon`
        carries that sense key.

    Raises:
        OptionalDependencyError: If `wn` isn't installed.
    """
    try:
        import wn
    except ImportError as exc:
        raise OptionalDependencyError("sensekey_to_synset_id", "wn") from exc

    lemma = sense_key.split("%", 1)[0]
    senses = wn.senses(lemma, lexicon=lexicon)
    if not senses and "_" in lemma:
        senses = wn.senses(lemma.replace("_", " "), lexicon=lexicon)
    candidates = {sense_key}
    variant = _adjective_satellite_variant(sense_key)
    if variant is not None:
        candidates.add(variant)
    for sense in senses:
        if sense.metadata().get("identifier") in candidates:
            return sense.synset().id
    return None


def synset_id_to_sensekey(synset_id: str, lemma: str, lexicon: str = "omw-en:1.4") -> str | None:
    """The inverse of [sensekey_to_synset_id][linkingtk.sources.wn.sensekey_to_synset_id]:
    given a synset id and the lemma it was reached through, return that
    lemma's sense key on that synset.

    A synset can lexicalize several lemmas, each with its own sense key on
    that synset -- `lemma` picks which one, the same way UFSAC/SemEval's
    own gold-standard sense keys are always relative to a specific
    instance's lemma, not just its synset. Useful for converting this
    package's predictions back to sense-key form to cross-check against an
    external, sense-key-based scorer (e.g. the Raganato et al. WSD
    evaluation framework's own `Scorer.java`).

    Returns `lexicon`'s own stored form as-is, unlike
    `sensekey_to_synset_id`'s adjective/adjective-satellite fallback --
    there's no target key here to fall back *towards*. In practice this
    is `omw-en:1.4`'s own convention (always digit `5`, adjective
    satellite), which for adjectives can differ from what a specific
    external gold file uses for that word (the official Raganato et al.
    gold files use a genuine, per-word mix of both `3` and `5`, not one
    uniformly) -- a real, known source of mismatched-but-not-wrong
    predictions when cross-checking adjective instances against an
    external scorer.

    Args:
        synset_id: A `wn` synset id, e.g. ``"omw-en-00031264-n"``.
        lemma: The lemma to resolve the sense key for.
        lexicon: A `wn` lexicon specifier whose senses carry the original
            sense key as their ``identifier`` metadata -- see
            `sensekey_to_synset_id`.

    Returns:
        The matching sense key, or ``None`` if `lemma` has no sense on
        `synset_id` in `lexicon`.

    Raises:
        OptionalDependencyError: If `wn` isn't installed.
    """
    try:
        import wn
    except ImportError as exc:
        raise OptionalDependencyError("synset_id_to_sensekey", "wn") from exc

    senses = wn.senses(lemma, lexicon=lexicon)
    if not senses and "_" in lemma:
        senses = wn.senses(lemma.replace("_", " "), lexicon=lexicon)
    for sense in senses:
        if sense.synset().id == synset_id:
            identifier = sense.metadata().get("identifier")
            return identifier if isinstance(identifier, str) else None
    return None

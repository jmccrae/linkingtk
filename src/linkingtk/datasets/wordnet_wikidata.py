"""Loaders for WordNet-Wikidata, a heterogeneous taxonomic EA benchmark.

Source: https://github.com/jmccrae/wn-wd-entity-align's
``data/{subset}/jape/`` folders -- a URI-keyed relative of the
``ent_ids``/``triples`` zip format shared by
[linkingtk.datasets.dbp15k][]/[linkingtk.datasets.openea][]/
[linkingtk.datasets.icews][] (see ``linkingtk.datasets.kg_zip``'s
module docstring), fetched per-file like [linkingtk.datasets.naisc][]
rather than as a zip, since these files are individually small enough not
to need it.

Per that export's own ``NOTES.md``, ``s_labels``/``t_labels``' third column
is a same-language duplicate of the second (a cross-lingual-translation
slot DBP15K's format expects that doesn't apply here) -- only the second
column is used below.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from linkingtk.core.entity import Entity
from linkingtk.datasets._util import fetch_cached
from linkingtk.datasets.base import GraphDatasetLoader
from linkingtk.utils.graph import Graph, Triple

_BASE_URL = "https://raw.githubusercontent.com/jmccrae/wn-wd-entity-align/main/data"


class _WordNetWikidataDataset(GraphDatasetLoader):
    """Base loader for one WordNet-Wikidata subset's JAPE export."""

    _subset: ClassVar[str]

    def __init__(self, base_url: str | None = None, cache_dir: Path | None = None) -> None:
        """Create the loader.

        Args:
            base_url: Override for where ``s_labels``/``s_triples``/etc.
                are fetched from (a URL or ``file://`` path). Defaults to
                this subset's directory in the source repository.
            cache_dir: Override for the download cache directory. Ignored
                for ``file://`` URLs, which are never cached.
        """
        self.base_url = base_url if base_url is not None else f"{_BASE_URL}/{self._subset}/jape"
        self.cache_dir = cache_dir

    def _fetch(self, name: str) -> str:
        return fetch_cached(f"{self.base_url}/{name}", self.cache_dir).decode("utf-8")

    def _entities(self, side: str) -> list[Entity]:
        entities = []
        for line in self._fetch(f"{side}_labels").splitlines():
            uri, label, _translation = line.split("\t")
            # A comprehension, not `label.split("; ")` directly, so mypy infers
            # `list[str | LabelWithLang]` (Entity.labels' type) instead of the
            # narrower `list[str]` `str.split` returns -- `list` is invariant.
            entities.append(Entity(id=uri, labels=[part for part in label.split("; ")]))
        return entities

    def load(self) -> tuple[list[Entity], list[Entity], list[tuple[str, str]]]:
        entities1 = self._entities("s")
        entities2 = self._entities("t")
        ids1 = {entity.id for entity in entities1}
        ids2 = {entity.id for entity in entities2}
        ground_truth = []
        for line in self._fetch("ent_ILLs").splitlines():
            id1, id2 = line.split("\t")
            if id1 in ids1 and id2 in ids2:
                ground_truth.append((id1, id2))
        return entities1, entities2, ground_truth

    def _triples(self, side: str) -> list[Triple]:
        triples = []
        for line in self._fetch(f"{side}_triples").splitlines():
            subject, predicate, obj = line.split("\t")
            triples.append((subject, predicate, obj))
        return triples

    def load_graphs(self) -> tuple[Graph, Graph]:
        return self._triples("s"), self._triples("t")


class WordNetWikidataLanguagesDataset(_WordNetWikidataDataset):
    """~200 language senses/items (smallest subset -- a good default)."""

    _subset = "languages"


class WordNetWikidataLocationsDataset(_WordNetWikidataDataset):
    """Place-name senses/items."""

    _subset = "locations"


class WordNetWikidataOrganismsDataset(_WordNetWikidataDataset):
    """Species/taxon senses/items, keyed on binomial (scientific) names."""

    _subset = "organisms"


class WordNetWikidataOrganismsHardDataset(_WordNetWikidataDataset):
    """A harder variant of
    [WordNetWikidataOrganismsDataset]
    [linkingtk.datasets.wordnet_wikidata.WordNetWikidataOrganismsDataset].

    See that source repository's README for what makes it harder.
    """

    _subset = "organisms_hard"

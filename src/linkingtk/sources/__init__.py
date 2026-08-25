"""Concrete [EntitySource][linkingtk.core.source.EntitySource] wrappers over external targets."""

from linkingtk.sources.vector_index import Embedder, VectorIndexEntitySource
from linkingtk.sources.wikidata import WikidataEntitySource
from linkingtk.sources.wikipedia import WikipediaEntitySource
from linkingtk.sources.wn import WnEntitySource, sensekey_to_synset_id, synset_id_to_sensekey

__all__ = [
    "Embedder",
    "VectorIndexEntitySource",
    "WikidataEntitySource",
    "WikipediaEntitySource",
    "WnEntitySource",
    "sensekey_to_synset_id",
    "synset_id_to_sensekey",
]

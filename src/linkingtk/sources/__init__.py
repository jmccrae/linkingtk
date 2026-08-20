"""Concrete [EntitySource][linkingtk.core.source.EntitySource] wrappers over external targets."""

from linkingtk.sources.wn import WnEntitySource, sensekey_to_synset_id, synset_id_to_sensekey

__all__ = ["WnEntitySource", "sensekey_to_synset_id", "synset_id_to_sensekey"]

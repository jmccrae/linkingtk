"""Concrete [EntitySource][linkingtk.core.source.EntitySource] wrappers over external targets."""

from linkingtk.sources.wn import WnEntitySource, sensekey_to_synset_id

__all__ = ["WnEntitySource", "sensekey_to_synset_id"]

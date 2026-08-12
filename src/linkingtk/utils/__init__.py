"""Graph utilities wrapping optional NetworkX and RDFLib dependencies."""

from linkingtk.utils.graph import (
    build_id_mappings,
    map_triples_to_ids,
    to_triples,
    train_test_split_triples,
)

__all__ = ["build_id_mappings", "map_triples_to_ids", "to_triples", "train_test_split_triples"]

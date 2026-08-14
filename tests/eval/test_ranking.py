import numpy as np

from linkingtk.core.entity import Entity
from linkingtk.eval.ranking import rank_exhaustive


class _FakeLinker:
    """Structural stub for `_ScoringLinker` -- distinct source/target vectors per id."""

    def __init__(
        self,
        source_vectors: dict[str, list[float]],
        target_vectors: dict[str, list[float]],
    ) -> None:
        self._source_vectors = source_vectors
        self._target_vectors = target_vectors

    def source_embedding(self, entity_id: str) -> np.ndarray:
        return np.array(self._source_vectors[entity_id])

    def target_embedding(self, entity_id: str) -> np.ndarray:
        return np.array(self._target_vectors[entity_id])


def _entities(ids: list[str]) -> list[Entity]:
    return [Entity(id=entity_id, labels=[entity_id]) for entity_id in ids]


def test_ranks_targets_by_descending_similarity() -> None:
    vectors = {
        "s1": [1.0, 0.0],
        "t_best": [1.0, 0.0],
        "t_mid": [1.0, 1.0],
        "t_worst": [0.0, 1.0],
    }
    linker = _FakeLinker(vectors, vectors)

    ranked = rank_exhaustive(linker, _entities(["s1"]), _entities(["t_worst", "t_mid", "t_best"]))

    assert ranked == [("s1", ["t_best", "t_mid", "t_worst"])]


def test_full_list_returned_even_when_true_target_ranks_low() -> None:
    # A truncated top-k version would drop "t_true" entirely, silently
    # turning a low-but-real rank into a false miss -- see the
    # denominator-bug precedent this guards against.
    vectors = {
        "s1": [1.0, 0.0],
        "t_true": [-1.0, 0.0],  # deliberately the worst match
        "t_a": [1.0, 0.0],
        "t_b": [1.0, 0.1],
        "t_c": [1.0, 0.2],
    }
    linker = _FakeLinker(vectors, vectors)

    ranked = rank_exhaustive(linker, _entities(["s1"]), _entities(["t_true", "t_a", "t_b", "t_c"]))

    source_id, ranked_ids = ranked[0]
    assert source_id == "s1"
    assert len(ranked_ids) == 4
    assert ranked_ids[-1] == "t_true"


def test_respects_asymmetric_source_and_target_embeddings() -> None:
    # "e1" scores very differently depending on which side it's on --
    # confirms rank_exhaustive calls source_embedding/target_embedding
    # separately rather than one shared accessor (matches MTransE's/
    # KDCoE's real projected-source-vs-raw-target asymmetry).
    source_vectors = {"e1": [1.0, 0.0]}
    target_vectors = {"e1": [0.0, 1.0], "e2": [1.0, 0.0]}
    linker = _FakeLinker(source_vectors, target_vectors)

    ranked = rank_exhaustive(linker, _entities(["e1"]), _entities(["e1", "e2"]))

    assert ranked == [("e1", ["e2", "e1"])]


def test_empty_source_or_target_returns_empty_lists() -> None:
    linker = _FakeLinker({}, {})

    assert rank_exhaustive(linker, [], _entities(["t1"])) == []
    assert rank_exhaustive(linker, _entities(["s1"]), []) == [("s1", [])]

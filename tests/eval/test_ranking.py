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


def test_manhattan_metric_ranks_by_ascending_l1_distance() -> None:
    vectors = {
        "s1": [0.0, 0.0],
        "t_near": [1.0, 0.0],  # L1 distance 1
        "t_far": [5.0, 5.0],  # L1 distance 10
    }
    linker = _FakeLinker(vectors, vectors)

    ranked = rank_exhaustive(
        linker, _entities(["s1"]), _entities(["t_far", "t_near"]), metric="manhattan"
    )

    assert ranked == [("s1", ["t_near", "t_far"])]


def test_manhattan_and_cosine_can_disagree() -> None:
    # "t_a" is closer by raw L1 distance; "t_b" is more cosine-aligned
    # (same direction, just scaled) -- confirms the metric choice
    # actually changes the ranking, not just the raw score values.
    vectors = {
        "s1": [1.0, 0.0],
        "t_a": [0.9, 0.9],  # L1 dist 1.0, cosine ~0.707
        "t_b": [10.0, 0.0],  # L1 dist 9.0, cosine 1.0
    }
    linker = _FakeLinker(vectors, vectors)

    cosine_ranked = rank_exhaustive(
        linker, _entities(["s1"]), _entities(["t_a", "t_b"]), metric="cosine"
    )
    manhattan_ranked = rank_exhaustive(
        linker, _entities(["s1"]), _entities(["t_a", "t_b"]), metric="manhattan"
    )

    assert cosine_ranked == [("s1", ["t_b", "t_a"])]
    assert manhattan_ranked == [("s1", ["t_a", "t_b"])]


def test_inner_metric_ranks_by_raw_dot_product() -> None:
    vectors = {"s1": [1.0, 0.0], "t_small": [0.1, 0.0], "t_large": [5.0, 0.0]}
    linker = _FakeLinker(vectors, vectors)

    ranked = rank_exhaustive(
        linker, _entities(["s1"]), _entities(["t_small", "t_large"]), metric="inner"
    )

    assert ranked == [("s1", ["t_large", "t_small"])]


def test_unsupported_metric_raises() -> None:
    import pytest

    linker = _FakeLinker({"s1": [1.0]}, {"t1": [1.0]})
    with pytest.raises(ValueError, match="Unsupported metric"):
        rank_exhaustive(linker, _entities(["s1"]), _entities(["t1"]), metric="euclidean")  # type: ignore[arg-type]


def test_csls_matches_manual_formula() -> None:
    # cosine sim matrix is the identity: s1~t1=1, s1~t2=0, s2~t1=0, s2~t2=1.
    # With k=1, each source's/target's own nearest-neighbor mean is 1 (its
    # perfect match), so csls[i, j] = 2 * sim[i, j] - 1 - 1 -- computed by
    # hand here and compared directly against rank_exhaustive's ordering,
    # rather than relying on an intuitive "hub" narrative that's fragile to
    # construct correctly at this tiny (2x2) scale.
    linker = _FakeLinker(
        source_vectors={"s1": [1.0, 0.0], "s2": [0.0, 1.0]},
        target_vectors={"t1": [1.0, 0.0], "t2": [0.0, 1.0]},
    )

    ranked = rank_exhaustive(linker, _entities(["s1", "s2"]), _entities(["t1", "t2"]), csls_k=1)

    # csls[s1, t1] = 2*1 - 2 = 0 > csls[s1, t2] = 2*0 - 2 = -2 -> t1 first.
    assert ranked == [("s1", ["t1", "t2"]), ("s2", ["t2", "t1"])]


def test_csls_k_zero_matches_plain_metric() -> None:
    vectors = {"s1": [1.0, 0.5], "t1": [1.0, 0.0], "t2": [0.0, 1.0]}
    linker = _FakeLinker(vectors, vectors)

    plain = rank_exhaustive(linker, _entities(["s1"]), _entities(["t1", "t2"]))
    csls_off = rank_exhaustive(linker, _entities(["s1"]), _entities(["t1", "t2"]), csls_k=0)

    assert plain == csls_off

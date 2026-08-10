import pytest

from linkingtk.blocking.embedding import EmbeddingSimilarityBlocker
from linkingtk.core.entity import Entity


def _entities(*descriptions: str) -> list[Entity]:
    return [
        Entity(id=str(i), labels=[str(i)], description=description)
        for i, description in enumerate(descriptions)
    ]


def test_finds_candidates_sharing_terms() -> None:
    dataset1 = _entities("a small rodent with a long tail")
    dataset2 = _entities(
        "small furry rodent, often a household pest",
        "wooden club used to strike a ball",
    )

    pairs = EmbeddingSimilarityBlocker(field="description").candidate_pairs(dataset1, dataset2)

    matched_ids = {e2.id for _, e2 in pairs}
    assert matched_ids == {"0"}


def test_no_shared_terms_yields_no_candidates() -> None:
    dataset1 = _entities("apple banana cherry")
    dataset2 = _entities("wooden club ball")

    pairs = EmbeddingSimilarityBlocker(field="description").candidate_pairs(dataset1, dataset2)

    assert pairs == []


def test_top_k_caps_results_per_source() -> None:
    dataset1 = _entities("cat dog bird fish")
    dataset2 = _entities("cat dog", "cat bird", "cat fish", "cat mouse", "cat snake")

    pairs = EmbeddingSimilarityBlocker(field="description", top_k=2).candidate_pairs(
        dataset1, dataset2
    )

    assert len(pairs) == 2


def test_threshold_prunes_weak_candidates() -> None:
    dataset1 = _entities("cat dog bird")
    dataset2 = _entities("cat dog bird", "cat")

    unfiltered = EmbeddingSimilarityBlocker(field="description", top_k=5).candidate_pairs(
        dataset1, dataset2
    )
    filtered = EmbeddingSimilarityBlocker(
        field="description", top_k=5, threshold=0.99
    ).candidate_pairs(dataset1, dataset2)

    assert len(unfiltered) == 2
    assert len(filtered) == 1


def test_ranks_closer_match_first() -> None:
    dataset1 = _entities("small furry rodent that likes cheese")
    dataset2 = _entities(
        "a small furry rodent with a long tail",
        "a small pointing device, not furry at all",
    )

    pairs = EmbeddingSimilarityBlocker(field="description", top_k=2).candidate_pairs(
        dataset1, dataset2
    )

    assert [e2.id for _, e2 in pairs] == ["0", "1"]


def test_rejects_invalid_config() -> None:
    with pytest.raises(ValueError):
        EmbeddingSimilarityBlocker(top_k=0)


def test_max_document_frequency_prunes_overly_common_terms() -> None:
    dataset1 = _entities("common")
    dataset2 = _entities("common one", "common two", "common three")

    unpruned = EmbeddingSimilarityBlocker(field="description").candidate_pairs(dataset1, dataset2)
    pruned = EmbeddingSimilarityBlocker(
        field="description", max_document_frequency=2
    ).candidate_pairs(dataset1, dataset2)

    assert len(unpruned) == 3
    assert pruned == []

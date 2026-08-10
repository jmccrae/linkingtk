import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

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


def test_custom_vectorizer_max_df_prunes_overly_common_terms() -> None:
    dataset1 = _entities("shared apple")
    dataset2 = _entities("shared banana", "shared cherry", "shared date")

    unpruned = EmbeddingSimilarityBlocker(field="description").candidate_pairs(dataset1, dataset2)
    pruned = EmbeddingSimilarityBlocker(
        field="description", vectorizer=TfidfVectorizer(max_df=2)
    ).candidate_pairs(dataset1, dataset2)

    assert len(unpruned) == 3
    assert pruned == []


def test_custom_vectorizer_returning_dense_array_is_supported() -> None:
    class _DenseStubVectorizer:
        def fit_transform(self, raw_documents: list[str]) -> np.ndarray:
            self._vocab = sorted({token for text in raw_documents for token in text.split()})
            return self.transform(raw_documents)

        def transform(self, raw_documents: list[str]) -> np.ndarray:
            return np.array(
                [[float(token in text.split()) for token in self._vocab] for text in raw_documents]
            )

    dataset1 = _entities("cat dog", "elephant")
    dataset2 = _entities("cat dog", "bird")

    pairs = EmbeddingSimilarityBlocker(
        field="description", vectorizer=_DenseStubVectorizer()
    ).candidate_pairs(dataset1, dataset2)

    assert [(e1.id, e2.id) for e1, e2 in pairs] == [("0", "0")]

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from linkingtk.algorithms.wsd._ewiser_text import (  # noqa: E402
    mean_pool_subwords,
    mention_sentence_and_span,
    whitespace_tokenize_with_offsets,
    word_index_for_span,
)
from linkingtk.core.entity import Entity  # noqa: E402


class TestWhitespaceTokenizeWithOffsets:
    def test_splits_and_returns_offsets(self) -> None:
        tokens = whitespace_tokenize_with_offsets("I sat by the bank today")

        assert tokens == [
            ("I", 0, 1),
            ("sat", 2, 5),
            ("by", 6, 8),
            ("the", 9, 12),
            ("bank", 13, 17),
            ("today", 18, 23),
        ]

    def test_collapses_multiple_spaces(self) -> None:
        tokens = whitespace_tokenize_with_offsets("a  b")

        assert tokens == [("a", 0, 1), ("b", 3, 4)]

    def test_empty_string_returns_no_tokens(self) -> None:
        assert whitespace_tokenize_with_offsets("") == []


class TestWordIndexForSpan:
    def test_exact_containment_match(self) -> None:
        tokens = whitespace_tokenize_with_offsets("I sat by the bank today")

        assert word_index_for_span(tokens, 13, 17) == 4

    def test_no_overlap_returns_none(self) -> None:
        assert word_index_for_span([("a", 0, 1)], 5, 6) is None

    def test_partial_overlap_falls_back_to_greatest_overlap(self) -> None:
        tokens = [("banks", 0, 5)]

        # span [0, 4) overlaps "banks" [0, 5) by 4 chars -- picks it even
        # though it doesn't start/end on the token boundary.
        assert word_index_for_span(tokens, 0, 4) == 0


class TestMentionSentenceAndSpan:
    def test_returns_context_tuple_directly(self) -> None:
        mention = Entity(id="m1", labels=["bank"], context=("I sat by the bank", 13, 17))

        assert mention_sentence_and_span(mention) == ("I sat by the bank", 13, 17)

    def test_falls_back_to_label_for_plain_string_context(self) -> None:
        mention = Entity(id="m1", labels=["bank"], context="some text")

        text, start, end = mention_sentence_and_span(mention)

        assert text == "some text"
        assert (start, end) == (0, len(text))

    def test_falls_back_to_label_for_no_context(self) -> None:
        mention = Entity(id="m1", labels=["bank"], context=None)

        text, start, end = mention_sentence_and_span(mention)

        assert text == "bank"
        assert (start, end) == (0, 4)


class TestMeanPoolSubwords:
    def test_averages_subwords_sharing_a_word_index(self) -> None:
        # 2 words; word 0 has 2 subwords (positions 1,2 after a [CLS] at 0),
        # word 1 has 1 subword (position 3), position 0 and 4 are special
        # tokens (word_id=None).
        hidden = torch.zeros(1, 5, 2)
        hidden[0, 1] = torch.tensor([2.0, 0.0])
        hidden[0, 2] = torch.tensor([4.0, 0.0])
        hidden[0, 3] = torch.tensor([10.0, 0.0])
        word_ids_batch: list[list[int | None]] = [[None, 0, 0, 1, None]]

        pooled = mean_pool_subwords(hidden, word_ids_batch, num_words=[2])

        assert pooled[0].shape == (2, 2)
        assert pooled[0][0, 0].item() == pytest.approx(3.0)  # mean(2.0, 4.0)
        assert pooled[0][1, 0].item() == pytest.approx(10.0)

    def test_handles_multiple_sentences_in_a_batch(self) -> None:
        hidden = torch.zeros(2, 3, 1)
        hidden[0, 0] = 5.0
        hidden[1, 0] = 7.0
        hidden[1, 1] = 9.0
        word_ids_batch: list[list[int | None]] = [[0, None, None], [0, 0, None]]

        pooled = mean_pool_subwords(hidden, word_ids_batch, num_words=[1, 1])

        assert pooled[0][0, 0].item() == pytest.approx(5.0)
        assert pooled[1][0, 0].item() == pytest.approx(8.0)  # mean(7.0, 9.0)

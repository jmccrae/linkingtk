"""Sentence/word alignment helpers for `EwiserEncoder`.

EWISER classifies one vector per whitespace word (mean-pooled from BERT
subwords), unlike
[GlossBertEncoder][linkingtk.algorithms.wsd.glossbert.GlossBertEncoder],
which never needs word-level positions. These helpers resolve a mention's
character span to a word index, subword-tokenize each whitespace word in
isolation (see `wordpiece_tokenize_words`'s docstring for why this can't
just be a high-level tokenizer call), and pool subword hidden states back
up to word level.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from linkingtk.core.entity import Entity, label_texts

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


def whitespace_tokenize_with_offsets(text: str) -> list[tuple[str, int, int]]:
    """Split `text` on whitespace, keeping each word's character span."""
    tokens: list[tuple[str, int, int]] = []
    start: int | None = None
    for index, char in enumerate(text):
        if char.isspace():
            if start is not None:
                tokens.append((text[start:index], start, index))
                start = None
        elif start is None:
            start = index
    if start is not None:
        tokens.append((text[start:], start, len(text)))
    return tokens


def word_index_for_span(tokens: list[tuple[str, int, int]], start: int, end: int) -> int | None:
    """The index of `tokens` containing character span ``[start, end)``, or ``None``.

    Prefers an exact containment match (`tok_start <= start < tok_end`);
    falls back to the token with the greatest character overlap for a
    span that doesn't land cleanly on a whitespace-token boundary.
    """
    for index, (_word, tok_start, tok_end) in enumerate(tokens):
        if tok_start <= start < tok_end:
            return index

    best_index: int | None = None
    best_overlap = 0
    for index, (_word, tok_start, tok_end) in enumerate(tokens):
        overlap = min(end, tok_end) - max(start, tok_start)
        if overlap > best_overlap:
            best_index, best_overlap = index, overlap
    return best_index


def mention_sentence_and_span(entity: Entity) -> tuple[str, int, int]:
    """The sentence text and mention character span to run `EwiserEncoder` against.

    Uses `context`'s ``(text, start, end)`` offsets when it's a
    [ContextWithSpan][linkingtk.core.entity.ContextWithSpan] (the shape
    [SemCorDataset][linkingtk.datasets.semcor.SemCorDataset]/
    [UfsacDataset][linkingtk.datasets.ufsac.UfsacDataset] mentions carry --
    every mention from the same sentence shares an identical `text`, which
    is what lets `EwiserEncoder.score` batch by unique sentence instead of
    per pair). Falls back to treating the mention's own label as a
    standalone one-word "sentence" when `context` carries no span.
    """
    context = entity.context
    if isinstance(context, tuple):
        return context
    text = context if isinstance(context, str) else " ".join(label_texts(entity))
    return text, 0, len(text)


def _wordpiece_tokenize_one_word(
    word: str, vocab: dict[str, int], unk_token: str, max_chars: int = 200
) -> list[str]:
    """BERT's standard greedy-longest-match WordPiece algorithm, applied to
    one already-segmented word (no further splitting -- see
    `wordpiece_tokenize_words`)."""
    if len(word) > max_chars:
        return [unk_token]
    tokens: list[str] = []
    start = 0
    while start < len(word):
        end = len(word)
        piece = None
        while start < end:
            candidate = word[start:end]
            if start > 0:
                candidate = "##" + candidate
            if candidate in vocab:
                piece = candidate
                break
            end -= 1
        if piece is None:
            return [unk_token]
        tokens.append(piece)
        start = end
    return tokens


def wordpiece_tokenize_words(
    tokenizer: PreTrainedTokenizerBase, words: list[str]
) -> tuple[list[str], list[int | None]]:
    """Subword-tokenize each whitespace `words` entry in isolation, returning
    ``([CLS] + subwords + [SEP], word_ids)`` (`word_ids[t]` is the index into
    `words` subtoken `t` came from, `None` for `[CLS]`/`[SEP]`).

    Deliberately **not** ``tokenizer(words, is_split_into_words=True)``:
    that high-level call still runs BERT's own punctuation-splitting
    pre-tokenizer on each provided "word" before wordpiece, so a word
    containing internal punctuation (e.g. UFSAC's own single-token
    ``"Oct."``) gets split into two independent pre-tokens (``"Oct"``,
    ``"."``) before wordpiece ever runs, producing a *standalone* ``"."``
    token -- not the ``"##."`` continuation-piece EWISER's own
    reference produces by wordpiece-tokenizing the literal string
    ``"Oct."`` directly, with no pre-splitting step at all. Confirmed
    directly: this single-token divergence (`"."` vs `"##."` are
    different vocabulary entries with unrelated embeddings) was traced,
    layer by layer, as the exact source of a checkpoint-reproduction gap
    against EWISER's own published numbers -- affects any UFSAC word with
    internal punctuation (abbreviations like "Oct."/"Sept.", likely also
    "U.S.", possessives, etc.), a small fraction of tokens but enough to
    measurably shift results once fed through the checkpoint's own
    BatchNorm (calibrated tightly enough to the reference's exact
    activations that even one wrong subtoken's ripple through attention
    measurably shifts other positions' representations).

    Implements BERT's WordPiece algorithm directly (greedy longest-match
    against the tokenizer's own vocab) rather than depending on any
    tokenizer-internal ``wordpiece_tokenizer`` attribute -- not exposed by
    every `transformers` version/backend.
    """
    vocab = tokenizer.get_vocab()
    unk_token = tokenizer.unk_token
    subwords: list[str] = []
    word_ids: list[int | None] = []
    for word_index, word in enumerate(words):
        for piece in _wordpiece_tokenize_one_word(word, vocab, unk_token):
            subwords.append(piece)
            word_ids.append(word_index)
    tokens = [tokenizer.cls_token, *subwords, tokenizer.sep_token]
    return tokens, [None, *word_ids, None]


def mean_pool_subwords(
    hidden: torch.Tensor, word_ids_batch: list[list[int | None]], num_words: list[int]
) -> list[torch.Tensor]:
    """Mean-pool `hidden` (``[batch, seq, H]``) subword vectors back to one vector per word.

    `word_ids_batch[b][t]` is the word index subtoken `t` of sequence `b`
    belongs to (``None`` for special/padding tokens), from the tokenizer's
    own fast `word_ids()` alignment.
    """
    pooled: list[torch.Tensor] = []
    for batch_index, word_ids in enumerate(word_ids_batch):
        n = num_words[batch_index]
        real = [(t, wid) for t, wid in enumerate(word_ids) if wid is not None and wid < n]
        positions = torch.tensor([t for t, _wid in real], dtype=torch.long, device=hidden.device)
        word_index = torch.tensor([wid for _t, wid in real], dtype=torch.long, device=hidden.device)
        vectors = hidden[batch_index, positions]

        sums = hidden.new_zeros(n, hidden.shape[-1]).index_add_(0, word_index, vectors)
        counts = hidden.new_zeros(n).index_add_(
            0, word_index, torch.ones_like(word_index, dtype=hidden.dtype)
        )
        pooled.append(sums / counts.clamp(min=1.0).unsqueeze(-1))
    return pooled

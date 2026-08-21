"""Sentence/word alignment helpers for `EwiserEncoder`.

EWISER classifies one vector per whitespace word (mean-pooled from BERT
subwords), unlike
[GlossBertEncoder][linkingtk.algorithms.wsd.glossbert.GlossBertEncoder],
which never needs word-level positions. These helpers resolve a mention's
character span to a word index and pool subword hidden states back up to
that word level.
"""

from __future__ import annotations

import torch

from linkingtk.core.entity import Entity, label_texts


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

"""Joint-sequence construction helpers for `EscEncoder`.

ESC scores a mention's candidate senses with a single extractive-QA forward
pass per mention: the context sentence (with the target word bracketed by
literal ``<classify>``/``</classify>`` markers) followed by every
candidate's gloss, concatenated, tokenized as one ``(sequence1,
sequence2)`` pair. The correct candidate is identified by a token span
within `sequence2` -- these helpers build that joint sequence and resolve
each candidate's span in it.

Reimplemented from understanding the reference's own
``esc/utils/definitions_tokenizer.py`` (``prepare_sample_without_st``), not
copied -- the reference repo is CC-BY-NC-SA 4.0, incompatible with
redistributing literal code under this package's Apache-2.0 license (see
[EwiserEncoder][linkingtk.algorithms.wsd.ewiser.EwiserEncoder]'s module
docstring for the same constraint on the EWISER port). Uses the fast
tokenizer's own `Encoding.sequence_ids` property to tell `sequence1`'s
tokens from `sequence2`'s, rather than the reference's own heuristic of
scanning for two consecutive ``(0, 0)``-offset tokens in a row -- a more
direct, less fragile way to find the same boundary.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from linkingtk.core.entity import Entity, description_text

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

CLASSIFY_START = "<classify>"
CLASSIFY_END = "</classify>"


def candidate_gloss(sense: Entity) -> str:
    """``Definition text, capitalized, with a trailing period`` -- matches
    the reference's own gloss formatting (`synset.definition().capitalize()
    + "."`)."""
    text = description_text(sense).strip()
    if not text:
        return "."
    text = text[0].upper() + text[1:]
    return text if text.endswith(".") else f"{text}."


def insert_classify_markers(text: str, start: int, end: int) -> str:
    """Bracket `text`'s ``[start, end)`` mention span with literal
    ``<classify>``/``</classify>`` markers (plain text, not added special
    tokens -- matches the published checkpoint's own
    ``use_special_tokens=False`` training configuration).
    """
    return f"{text[:start]}{CLASSIFY_START} {text[start:end]} {CLASSIFY_END}{text[end:]}"


def build_joint_sequence(
    tokenizer: PreTrainedTokenizerBase,
    context_sentence: str,
    candidate_glosses: list[str],
    max_length: int,
) -> tuple[torch.Tensor, torch.Tensor, list[tuple[int, int] | None]]:
    """Tokenize ``(context_sentence, " ".join(candidate_glosses))`` as one
    pair sequence and resolve each candidate's ``(start_token, end_token)``
    span within it.

    Returns ``(input_ids, attention_mask, spans)`` for **one** sequence
    (batching/padding happens at the caller, matching
    `EwiserEncoder._encode_chunk`'s existing pattern) -- `input_ids`/
    `attention_mask` are 1-D. `spans[i]` is `None` if candidate `i`'s gloss
    span couldn't be resolved to token boundaries, or falls outside a
    sequence truncated to `max_length` (the caller treats this the same
    way `EwiserEncoder.score` treats an unresolved candidate: unscoreable,
    not a crash).
    """
    definitions_seq = " ".join(candidate_glosses)
    definitions_offsets: list[tuple[int, int]] = []
    cursor = 0
    for gloss in candidate_glosses:
        start = cursor
        end = start + len(gloss)
        definitions_offsets.append((start, end))
        cursor = end + 1  # +1 for the joining space

    encoded = tokenizer(
        context_sentence,
        definitions_seq,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    input_ids = encoded["input_ids"][0]
    attention_mask = encoded["attention_mask"][0]
    encoding = encoded.encodings[0]
    offsets = encoding.offsets
    sequence_ids = encoding.sequence_ids

    char_start_to_token: dict[int, int] = {}
    char_end_to_token: dict[int, int] = {}
    for token_index, (sequence_id, (char_start, char_end)) in enumerate(
        zip(sequence_ids, offsets, strict=True)
    ):
        if sequence_id != 1:
            continue
        char_start_to_token.setdefault(char_start, token_index)
        char_end_to_token.setdefault(char_end, token_index)

    spans: list[tuple[int, int] | None] = []
    for gloss_start, gloss_end in definitions_offsets:
        start_token = char_start_to_token.get(gloss_start)
        end_token = char_end_to_token.get(gloss_end)
        if start_token is None or end_token is None:
            spans.append(None)
        else:
            spans.append((start_token, end_token))

    return input_ids, attention_mask, spans


def pad_encoded_sequences(
    sequences: list[tuple[torch.Tensor, torch.Tensor]], pad_token_id: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad a list of ``(input_ids, attention_mask)`` 1-D tensor pairs
    (e.g. from repeated `build_joint_sequence` calls) into one batch.

    Shared by [EscEncoder][linkingtk.algorithms.wsd.esc.EscEncoder]'s
    scoring path and [EscTrainer][linkingtk.train.esc_trainer.EscTrainer]'s
    training path, so both batch mentions of differing joint-sequence
    length the same way.
    """
    max_len = max(ids.size(0) for ids, _mask in sequences)
    batch_input_ids = torch.full((len(sequences), max_len), pad_token_id, dtype=torch.long)
    batch_attention_mask = torch.zeros((len(sequences), max_len), dtype=torch.long)
    for row, (ids, mask) in enumerate(sequences):
        batch_input_ids[row, : ids.size(0)] = ids
        batch_attention_mask[row, : mask.size(0)] = mask
    return batch_input_ids, batch_attention_mask

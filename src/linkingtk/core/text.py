"""Shared plain-text tokenization and field selection for string- and vector-based algorithms."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Literal

from linkingtk.core.entity import Entity, context_text, description_text, label_texts

_TOKEN_RE = re.compile(r"[a-zA-Z]+")

_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "of",
        "to",
        "in",
        "on",
        "at",
        "for",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "it",
        "its",
        "this",
        "that",
        "with",
        "as",
        "by",
        "from",
        "has",
        "have",
        "had",
        "not",
        "he",
        "she",
        "they",
        "we",
        "you",
        "i",
    }
)


def tokenize(text: str) -> list[str]:
    """Lowercase, alphabetic tokens with stopwords removed; duplicates preserved."""
    return [
        lowered for token in _TOKEN_RE.findall(text) if (lowered := token.lower()) not in _STOPWORDS
    ]


FieldName = Literal["label", "description", "context"]
Field = FieldName | Callable[[Entity], str]

_FIELD_EXTRACTORS: dict[FieldName, Callable[[Entity], str]] = {
    "label": lambda entity: " ".join(label_texts(entity)),
    "description": description_text,
    "context": context_text,
}


def resolve_field(field: Field) -> Callable[[Entity], str]:
    """Resolve a ``Field`` into a callable extracting text from an ``Entity``."""
    return _FIELD_EXTRACTORS[field] if isinstance(field, str) else field

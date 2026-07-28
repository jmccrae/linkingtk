"""Core data model shared by all linking tasks."""

from __future__ import annotations

from dataclasses import dataclass, field

LabelWithLang = tuple[str, str]
ContextWithSpan = tuple[str, int, int]


@dataclass
class Entity:
    """A single entity/sense/mention participating in a linking task.

    Attributes:
        id: An identifier for the entity.
        labels: Plain strings such as ``"cat"``, or pairs with a language
            such as ``("cat", "en")``.
        description: The main definition of the entity, optionally tagged
            with a language.
        context: An occurrence of the label in text, optionally with
            character offsets ``(text, start, end)`` indicating the mention.
        properties: Extra key/value properties describing the entity.
    """

    id: str
    labels: list[str | LabelWithLang]
    description: str | LabelWithLang | None = None
    context: str | ContextWithSpan | None = None
    properties: dict[str, str] = field(default_factory=dict)

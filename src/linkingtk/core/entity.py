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


def label_texts(entity: Entity) -> list[str]:
    """Extract the plain text of each of an entity's labels.

    Language tags are dropped; duplicates are preserved so callers can
    weight by how often a label occurs.
    """
    return [label[0] if isinstance(label, tuple) else label for label in entity.labels]


def description_text(entity: Entity) -> str:
    """Extract the plain text of an entity's description, ignoring its language tag.

    Returns an empty string if the entity has no description.
    """
    if entity.description is None:
        return ""
    return entity.description if isinstance(entity.description, str) else entity.description[0]


def context_text(entity: Entity) -> str:
    """Extract the plain text of an entity's context, ignoring any character offsets.

    Returns an empty string if the entity has no context.
    """
    if entity.context is None:
        return ""
    return entity.context if isinstance(entity.context, str) else entity.context[0]

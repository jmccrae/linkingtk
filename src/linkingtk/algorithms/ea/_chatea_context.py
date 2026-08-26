"""Per-entity context (neighbor triples, names, LLM-generated descriptions)
for [ChatEALinker][linkingtk.algorithms.ea.chatea.ChatEALinker]'s prompts.

Ports the neighbor-rendering half of the reference's
`tools_for_ChatEA.py`'s `NeighborGenerator.get_neighbors` (as
[NeighborIndex][linkingtk.algorithms.ea._chatea_context.NeighborIndex])
and the description pre-compute of `preobtain_description.py`'s
`generate_prompt`/`get_description` (as
[generate_descriptions][linkingtk.algorithms.ea._chatea_context.generate_descriptions]).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from linkingtk.llm.client import LlmClient, LlmMessage
from linkingtk.utils.graph import Triple

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = logging.getLogger("linkingtk")

NeighborTriple = tuple[str, str, str, "str | None", "str | None"]
"""``(subject_name, relation_name, object_name, start_label, end_label)``
-- time labels are ``None`` for non-temporal datasets/triples."""

TemporalTriple = tuple[str, str, str, "str | None", "str | None"]

_DESCRIPTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"description": {"type": "string"}},
    "required": ["description"],
}


@dataclass
class EntityContext:
    """One entity's rendered prompt content."""

    entity_id: str
    name: str
    description: str
    neighbors: list[NeighborTriple] = field(default_factory=list)


class NeighborIndex:
    """Looks up an entity's name and (capped) rendered neighbor triples.

    Args:
        triples: Combined relational triples, used only when
            `temporal_triples` is `None`.
        temporal_triples: Combined temporal triples (preferred over
            `triples` when given, since they carry strictly more
            information -- the same relation/entity ids, plus time).
        entity_names: `entity_id` -> display name (from `Entity.labels`).
        relation_names: `relation_id` -> display name (e.g. from
            [IcewsWikiDataset.load_relation_labels][linkingtk.datasets.icews.IcewsWikiDataset.load_relation_labels]).
            Falls back to the raw id for any relation not present.
        neigh_num: Max neighbor triples kept per entity (both as subject
            and object), mirroring the reference's own `--neigh` cap.
    """

    def __init__(
        self,
        triples: list[Triple],
        temporal_triples: list[TemporalTriple] | None,
        entity_names: dict[str, str],
        relation_names: dict[str, str],
        neigh_num: int,
    ) -> None:
        self._entity_names = entity_names
        self._relation_names = relation_names
        self._neigh_num = neigh_num
        self._raw: dict[str, list[TemporalTriple]] = defaultdict(list)
        source: Iterable[TemporalTriple] = (
            temporal_triples
            if temporal_triples is not None
            else [(s, r, o, None, None) for s, r, o in triples]
        )
        for subject_id, relation_id, object_id, start, end in source:
            fact = (subject_id, relation_id, object_id, start, end)
            self._raw[subject_id].append(fact)
            self._raw[object_id].append(fact)

    def name(self, entity_id: str) -> str:
        return self._entity_names.get(entity_id, entity_id)

    def neighbors(self, entity_id: str) -> list[NeighborTriple]:
        raw = self._raw.get(entity_id, [])[: self._neigh_num]
        return [
            (self.name(s), self._relation_names.get(r, r), self.name(o), start, end)
            for s, r, o, start, end in raw
        ]


def format_triple(triple: NeighborTriple) -> str:
    """Renders one neighbor triple as prompt text, e.g. ``"(A, rel, B, 2005-11, 2005-11)"``."""
    subject, relation, obj, start, end = triple
    if start is None and end is None:
        return f"({subject}, {relation}, {obj})"
    return f"({subject}, {relation}, {obj}, {start or '?'}, {end or '?'})"


def _description_prompt(name: str, neighbors: list[NeighborTriple]) -> list[LlmMessage]:
    tuple_text = ", ".join(format_triple(t) for t in neighbors)
    instruction = (
        "Give a one-sentence brief introduction for the given entity, based on "
        "1. YOUR OWN KNOWLEDGE; 2. the given knowledge tuples. The introduction "
        "must be a single sentence, under 50 tokens."
    )
    user_content = (
        f"Entity: {name}\n"
        f"Known tuples: [{tuple_text}]\n"
        f"What is {name}? Give a one-sentence brief introduction based on your "
        "own knowledge and the tuples above."
    )
    return [
        LlmMessage(role="system", content=instruction),
        LlmMessage(role="user", content=user_content),
    ]


def generate_descriptions(
    client: LlmClient,
    entity_ids: Iterable[str],
    index: NeighborIndex,
    *,
    max_tokens: int = 80,
) -> dict[str, str]:
    """One cached LLM-generated one-sentence description per id in `entity_ids`.

    Ports `preobtain_description.py`'s `get_description` -- called once
    per entity actually needed (a source entity or one of its top-k
    candidates), not once per comparison.
    """
    descriptions: dict[str, str] = {}
    for entity_id in entity_ids:
        if entity_id in descriptions:
            continue
        messages = _description_prompt(index.name(entity_id), index.neighbors(entity_id))
        try:
            response = client.complete_structured(
                messages, schema=_DESCRIPTION_SCHEMA, max_tokens=max_tokens
            )
            descriptions[entity_id] = str(response.get("description", "")).strip()
        except Exception:
            logger.warning(
                "ChatEALinker: description generation failed for entity %r, "
                "using empty description",
                entity_id,
                exc_info=True,
            )
            descriptions[entity_id] = ""
    return descriptions

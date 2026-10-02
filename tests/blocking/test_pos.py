from __future__ import annotations

from linkingtk.blocking.base import BlockingStrategy
from linkingtk.blocking.pos import WordNetPosFilter
from linkingtk.core.entity import Entity


class _AllPairs(BlockingStrategy):
    def candidate_pairs(
        self, dataset1: list[Entity], dataset2: list[Entity]
    ) -> list[tuple[Entity, Entity]]:
        return [(e1, e2) for e1 in dataset1 for e2 in dataset2]


_SENSES = [
    Entity(id="omw-en-00001-n", labels=["play"]),
    Entity(id="omw-en-00002-v", labels=["play"]),
    Entity(id="omw-en-00003-a", labels=["play"]),
    Entity(id="omw-en-00004-s", labels=["play"]),
    Entity(id="omw-en-00005-r", labels=["play"]),
]


def _targets(pos: str | None) -> list[str]:
    properties = {"pos": pos} if pos is not None else {}
    mention = Entity(id="m", labels=["play"], properties=properties)
    pairs = WordNetPosFilter(_AllPairs()).candidate_pairs([mention], _SENSES)
    return [sense.id for _, sense in pairs]


def test_keeps_only_matching_pos_in_order() -> None:
    assert _targets("VBD") == ["omw-en-00002-v"]
    assert _targets("NNS") == ["omw-en-00001-n"]
    assert _targets("RB") == ["omw-en-00005-r"]


def test_adjectives_include_satellites() -> None:
    assert _targets("JJ") == ["omw-en-00003-a", "omw-en-00004-s"]


def test_unknown_or_missing_tag_keeps_everything() -> None:
    assert len(_targets(None)) == 5
    assert len(_targets("IN")) == 5

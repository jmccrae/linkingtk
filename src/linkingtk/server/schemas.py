"""Pydantic request/response models for the REST server.

These mirror the plain dataclasses in [linkingtk.core.entity][] and
[linkingtk.core.result][] so the API has a typed, OpenAPI-documented
contract, with small converters to/from the real dataclasses used
everywhere else in the library.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from linkingtk.core.entity import Entity
from linkingtk.core.result import AlignmentResult


class SchemaEntity(BaseModel):
    """JSON mirror of [Entity][linkingtk.core.entity.Entity]."""

    id: str
    labels: list[str | tuple[str, str]]
    description: str | tuple[str, str] | None = None
    context: str | tuple[str, int, int] | None = None
    properties: dict[str, str] = Field(default_factory=dict)

    def to_entity(self) -> Entity:
        return Entity(
            id=self.id,
            labels=self.labels,
            description=self.description,
            context=self.context,
            properties=self.properties,
        )


class SchemaAlignmentResult(BaseModel):
    """JSON mirror of [AlignmentResult][linkingtk.core.result.AlignmentResult]."""

    source_id: str
    target_id: str
    score: float = 1.0
    alternatives: list[str] = Field(default_factory=list)

    @classmethod
    def from_result(cls, result: AlignmentResult) -> SchemaAlignmentResult:
        return cls(
            source_id=result.source_id,
            target_id=result.target_id,
            score=result.score,
            alternatives=result.alternatives,
        )


class LinkRequestBody(BaseModel):
    """The JSON ``request`` part of a ``POST /link/{linker_name}`` call.

    Exactly one of ``dataset2_entities`` (an inline target list) or
    ``dataset2_source`` (the name of a pre-registered
    [EntitySource][linkingtk.core.source.EntitySource]) must be set.
    """

    dataset1: list[SchemaEntity]
    dataset2_entities: list[SchemaEntity] | None = None
    dataset2_source: str | None = None

    @model_validator(mode="after")
    def _check_dataset2(self) -> LinkRequestBody:
        if (self.dataset2_entities is None) == (self.dataset2_source is None):
            raise ValueError("Exactly one of 'dataset2_entities' or 'dataset2_source' must be set.")
        return self

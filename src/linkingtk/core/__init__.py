"""Core dataclasses and interfaces shared across LinkingTK tasks."""

from linkingtk.core.entity import Entity
from linkingtk.core.result import AlignmentResult
from linkingtk.core.source import CachingEntitySource, EntitySource

__all__ = ["Entity", "AlignmentResult", "EntitySource", "CachingEntitySource"]

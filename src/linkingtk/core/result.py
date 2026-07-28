"""Result types returned by linking algorithms."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AlignmentResult:
    """A single predicted link between two entities.

    Attributes:
        source_id: Identifier of the entity in ``dataset1``.
        target_id: Identifier of the matched entity in ``dataset2``.
        score: Confidence score assigned by the linker, higher is better.
        alternatives: Additional candidate target ids ranked after
            ``target_id``, used for ranked evaluation (e.g. Hits@k, MRR).
    """

    source_id: str
    target_id: str
    score: float = 1.0
    alternatives: list[str] = field(default_factory=list)

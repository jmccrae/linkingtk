"""Result container returned by :class:`linkingtk.eval.Evaluator`."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EvaluationReport:
    """Holds computed metrics for a single evaluation run.

    Attributes:
        metrics: Mapping of metric name (e.g. ``"precision@1"``, ``"f1"``,
            ``"Hits@1"``, ``"MRR"``) to its computed value.
    """

    metrics: dict[str, float]

"""Result container returned by [linkingtk.eval.evaluator.Evaluator][]."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EvaluationReport:
    """Holds computed metrics for a single evaluation run.

    Attributes:
        metrics: Mapping of metric name (e.g. ``"precision@1"``, ``"f1"``,
            ``"Hits@1"``, ``"MRR"``) to its computed value. A metric is
            ``None`` when it isn't computable for the given inputs (e.g.
            ``"reduction_ratio"`` when
            [evaluate_blocking][linkingtk.eval.evaluator.Evaluator.evaluate_blocking]
            is called without a ``dataset2_size``).
    """

    metrics: dict[str, float | None]

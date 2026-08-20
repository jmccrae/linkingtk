"""Standardized evaluation metrics for EA, EL, WSD and WSA."""

from __future__ import annotations

from linkingtk.eval.report import EvaluationReport


class Evaluator:
    """Computes standard linking metrics from predictions and ground truth."""

    @staticmethod
    def evaluate(
        predictions: list[tuple[str, str]],
        ground_truth: list[tuple[str, str]],
    ) -> EvaluationReport:
        """Compute micro precision@1, recall and F1 for unranked predictions.

        Args:
            predictions: List of ``(source_id, predicted_target_id)`` pairs.
            ground_truth: List of ``(source_id, target_id)`` pairs.

        Returns:
            A report containing ``precision@1``, ``recall`` and ``f1``.
        """
        truth_by_source = dict(ground_truth)
        correct = sum(
            1 for source_id, target_id in predictions if truth_by_source.get(source_id) == target_id
        )
        precision = correct / len(predictions) if predictions else 0.0
        recall = correct / len(ground_truth) if ground_truth else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return EvaluationReport(metrics={"precision@1": precision, "recall": recall, "f1": f1})

    @staticmethod
    def evaluate_ranked(
        ranked_predictions: list[tuple[str, list[str]]],
        ground_truth: list[tuple[str, str]],
        top_k: list[int],
    ) -> EvaluationReport:
        """Compute Hits@k and MRR for ranked predictions.

        Iterates over ``ground_truth`` (not ``ranked_predictions``), so the
        denominator is always exactly ``len(ground_truth)`` -- the number of
        queries actually being scored. This matters for callers (e.g. every
        EA-linker benchmark script under ``examples/``) that call
        ``linker.link()`` over a *superset* of the evaluated split (train +
        val + test entities, since which entities need embeddings is
        determined before the split is known): a ground-truth source with no
        matching entry in ``ranked_predictions`` at all (e.g. blocking found
        no candidates for it) counts as a miss, exactly like a source whose
        ranked list doesn't contain the correct target -- neither case is
        silently excluded, and no non-ground-truth prediction can dilute the
        denominator.

        Args:
            ranked_predictions: List of ``(source_id, ranked_target_ids)``
                pairs, best candidate first. May contain entries for
                ``source_id``s absent from ``ground_truth`` -- those are
                ignored rather than counted.
            ground_truth: List of ``(source_id, target_id)`` pairs.
            top_k: The cut-offs to compute Hits@k for.

        Returns:
            A report containing ``Hits@k`` for each ``k`` in ``top_k`` and
            ``MRR``.
        """
        ranked_ids_by_source = dict(ranked_predictions)
        hits = {k: 0 for k in top_k}
        reciprocal_ranks = []

        for source_id, target_id in ground_truth:
            ranked_ids = ranked_ids_by_source.get(source_id, [])
            if target_id in ranked_ids:
                rank = ranked_ids.index(target_id) + 1
                reciprocal_ranks.append(1.0 / rank)
                for k in top_k:
                    if rank <= k:
                        hits[k] += 1
            else:
                reciprocal_ranks.append(0.0)

        total = len(ground_truth) or 1
        metrics: dict[str, float | None] = {f"Hits@{k}": hits[k] / total for k in top_k}
        metrics["MRR"] = sum(reciprocal_ranks) / total
        return EvaluationReport(metrics=metrics)

    @staticmethod
    def evaluate_blocking(
        candidate_pairs: list[tuple[str, str]],
        ground_truth: list[tuple[str, str]],
        dataset1_size: int,
        dataset2_size: int | None = None,
    ) -> EvaluationReport:
        """Compute Pair Completeness and Reduction Ratio for a blocking pass.

        These metrics assess a [BlockingStrategy][linkingtk.blocking.base.BlockingStrategy]
        in isolation, before any linker scores the candidate pairs it produces.

        Args:
            candidate_pairs: ``(source_id, target_id)`` candidate pairs
                produced by a blocking strategy.
            ground_truth: List of ``(source_id, target_id)`` true matching
                pairs.
            dataset1_size: Number of entities in the first dataset.
            dataset2_size: Number of entities in the second dataset, or
                ``None`` if unknown -- e.g. blocking targeted an
                [EntitySource][linkingtk.core.source.EntitySource] with no
                fixed size. ``reduction_ratio`` is then not computable.

        Returns:
            A report containing ``pair_completeness`` (the fraction of true
            matches present among the candidate pairs, i.e. blocking recall)
            and ``reduction_ratio`` (the fraction of the full ``dataset1_size
            * dataset2_size`` cross-product eliminated by blocking, or
            ``None`` if ``dataset2_size`` wasn't given).
        """
        candidate_set = set(candidate_pairs)
        true_positives = sum(1 for pair in ground_truth if pair in candidate_set)
        pair_completeness = true_positives / len(ground_truth) if ground_truth else 0.0

        reduction_ratio: float | None = None
        if dataset2_size is not None:
            total_possible = dataset1_size * dataset2_size
            reduction_ratio = 1 - len(candidate_pairs) / total_possible if total_possible else 0.0
        return EvaluationReport(
            metrics={"pair_completeness": pair_completeness, "reduction_ratio": reduction_ratio}
        )

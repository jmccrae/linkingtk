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

        Args:
            ranked_predictions: List of ``(source_id, ranked_target_ids)``
                pairs, best candidate first.
            ground_truth: List of ``(source_id, target_id)`` pairs.
            top_k: The cut-offs to compute Hits@k for.

        Returns:
            A report containing ``Hits@k`` for each ``k`` in ``top_k`` and
            ``MRR``.
        """
        truth_by_source = dict(ground_truth)
        hits = {k: 0 for k in top_k}
        reciprocal_ranks = []

        for source_id, ranked_ids in ranked_predictions:
            target_id = truth_by_source.get(source_id)
            if target_id is None:
                continue
            if target_id in ranked_ids:
                rank = ranked_ids.index(target_id) + 1
                reciprocal_ranks.append(1.0 / rank)
                for k in top_k:
                    if rank <= k:
                        hits[k] += 1
            else:
                reciprocal_ranks.append(0.0)

        total = len(ranked_predictions) or 1
        metrics = {f"Hits@{k}": hits[k] / total for k in top_k}
        metrics["MRR"] = sum(reciprocal_ranks) / total
        return EvaluationReport(metrics=metrics)

    @staticmethod
    def evaluate_blocking(
        candidate_pairs: list[tuple[str, str]],
        ground_truth: list[tuple[str, str]],
        dataset1_size: int,
        dataset2_size: int,
    ) -> EvaluationReport:
        """Compute Pair Completeness and Reduction Ratio for a blocking pass.

        These metrics assess a :class:`~linkingtk.blocking.base.BlockingStrategy`
        in isolation, before any linker scores the candidate pairs it produces.

        Args:
            candidate_pairs: ``(source_id, target_id)`` candidate pairs
                produced by a blocking strategy.
            ground_truth: List of ``(source_id, target_id)`` true matching
                pairs.
            dataset1_size: Number of entities in the first dataset.
            dataset2_size: Number of entities in the second dataset.

        Returns:
            A report containing ``pair_completeness`` (the fraction of true
            matches present among the candidate pairs, i.e. blocking recall)
            and ``reduction_ratio`` (the fraction of the full ``dataset1_size
            * dataset2_size`` cross-product eliminated by blocking).
        """
        candidate_set = set(candidate_pairs)
        true_positives = sum(1 for pair in ground_truth if pair in candidate_set)
        pair_completeness = true_positives / len(ground_truth) if ground_truth else 0.0

        total_possible = dataset1_size * dataset2_size
        reduction_ratio = 1 - len(candidate_pairs) / total_possible if total_possible else 0.0
        return EvaluationReport(
            metrics={"pair_completeness": pair_completeness, "reduction_ratio": reduction_ratio}
        )

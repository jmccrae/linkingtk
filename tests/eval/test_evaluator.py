from linkingtk.eval.evaluator import Evaluator


def test_evaluate_precision_recall_f1() -> None:
    report = Evaluator.evaluate(
        predictions=[("e1", "e1_target"), ("e2", "e2_wrong")],
        ground_truth=[("e1", "e1_target"), ("e2", "e2_correct")],
    )
    assert report.metrics == {"precision@1": 0.5, "recall": 0.5, "f1": 0.5}


def test_evaluate_ranked_hits_and_mrr() -> None:
    report = Evaluator.evaluate_ranked(
        ranked_predictions=[
            ("e1", ["e1_target", "e1_alt"]),
            ("e2", ["e2_wrong", "e2_correct"]),
        ],
        ground_truth=[("e1", "e1_target"), ("e2", "e2_correct")],
        top_k=[1, 5],
    )
    assert report.metrics["Hits@1"] == 0.5
    assert report.metrics["Hits@5"] == 1.0
    assert report.metrics["MRR"] == 0.75


def test_evaluate_ranked_ignores_predictions_outside_ground_truth() -> None:
    # Regression test: a caller that links() a superset of the evaluated
    # split (e.g. train+val+test entities, common in the EA-linker
    # benchmark scripts under examples/) must not have those extra
    # predictions dilute the denominator -- only `len(ground_truth)`
    # queries are ever being scored.
    report = Evaluator.evaluate_ranked(
        ranked_predictions=[
            ("e1", ["e1_target"]),
            ("train_only_1", ["whatever"]),
            ("train_only_2", ["whatever_else"]),
        ],
        ground_truth=[("e1", "e1_target")],
        top_k=[1],
    )
    assert report.metrics["Hits@1"] == 1.0
    assert report.metrics["MRR"] == 1.0


def test_evaluate_ranked_counts_missing_prediction_as_a_miss() -> None:
    # The flip side of the same bug: a ground-truth source with *no*
    # matching entry in ranked_predictions at all (e.g. blocking found no
    # candidates for it) must still count in the denominator as a miss,
    # not be silently excluded.
    report = Evaluator.evaluate_ranked(
        ranked_predictions=[("e1", ["e1_target"])],
        ground_truth=[("e1", "e1_target"), ("e2", "e2_target")],
        top_k=[1],
    )
    assert report.metrics["Hits@1"] == 0.5
    assert report.metrics["MRR"] == 0.5


def test_evaluate_blocking_pair_completeness_and_reduction_ratio() -> None:
    report = Evaluator.evaluate_blocking(
        candidate_pairs=[("s1", "t1"), ("s2", "t2"), ("s2", "t3")],
        ground_truth=[("s1", "t1"), ("s2", "t2"), ("s3", "t4")],
        dataset1_size=3,
        dataset2_size=4,
    )
    assert report.metrics["pair_completeness"] == 2 / 3
    assert report.metrics["reduction_ratio"] == 1 - 3 / 12


def test_evaluate_blocking_without_dataset2_size_omits_reduction_ratio() -> None:
    report = Evaluator.evaluate_blocking(
        candidate_pairs=[("s1", "t1"), ("s2", "t2")],
        ground_truth=[("s1", "t1"), ("s2", "t2"), ("s3", "t4")],
        dataset1_size=3,
        dataset2_size=None,
    )
    assert report.metrics["reduction_ratio"] is None
    assert report.metrics["pair_completeness"] == 2 / 3


def test_evaluate_blocking_empty_ground_truth_and_no_candidates() -> None:
    report = Evaluator.evaluate_blocking(
        candidate_pairs=[],
        ground_truth=[],
        dataset1_size=0,
        dataset2_size=0,
    )
    assert report.metrics == {"pair_completeness": 0.0, "reduction_ratio": 0.0}

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

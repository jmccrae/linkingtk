from linkingtk.algorithms.string_similarity import StringSimilarityLinker
from linkingtk.algorithms.wsd import LeskLinker
from linkingtk.datasets.toy import ToyELDataset, ToyWSDDataset
from linkingtk.eval import Evaluator


def test_toy_wsd_dataset_shape() -> None:
    mentions, senses, ground_truth = ToyWSDDataset().load()

    assert len(mentions) == 4
    assert len(senses) == 4
    assert len(ground_truth) == 4

    mention_ids = {entity.id for entity in mentions}
    sense_ids = {entity.id for entity in senses}
    for source_id, target_id in ground_truth:
        assert source_id in mention_ids
        assert target_id in sense_ids


def test_toy_wsd_dataset_solvable_by_lesk() -> None:
    mentions, senses, ground_truth = ToyWSDDataset().load()

    results = LeskLinker().link(mentions, senses)
    predictions = [(result.source_id, result.target_id) for result in results]

    report = Evaluator.evaluate(predictions=predictions, ground_truth=ground_truth)
    assert report.metrics["precision@1"] == 1.0


def test_toy_el_dataset_shape() -> None:
    mentions, knowledge_base, ground_truth = ToyELDataset().load()

    assert len(mentions) == 6
    assert len(knowledge_base) == 6
    assert len(ground_truth) == 6

    mention_ids = {entity.id for entity in mentions}
    kb_ids = {entity.id for entity in knowledge_base}
    for source_id, target_id in ground_truth:
        assert source_id in mention_ids
        assert target_id in kb_ids


def test_toy_el_dataset_solvable_by_string_similarity() -> None:
    mentions, knowledge_base, ground_truth = ToyELDataset().load()

    linker = StringSimilarityLinker(
        source_field="context", target_field="description", metric="word_overlap"
    )
    results = linker.link(mentions, knowledge_base)
    predictions = [(result.source_id, result.target_id) for result in results]

    report = Evaluator.evaluate(predictions=predictions, ground_truth=ground_truth)
    assert report.metrics["precision@1"] == 1.0

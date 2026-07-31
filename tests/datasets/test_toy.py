from linkingtk.algorithms.string_similarity import StringSimilarityLinker
from linkingtk.algorithms.wsd import LeskLinker
from linkingtk.blocking import LabelOverlap
from linkingtk.datasets.toy import ToyEADataset, ToyELDataset, ToyWSADataset, ToyWSDDataset
from linkingtk.eval import Evaluator


def test_toy_ea_dataset_shape() -> None:
    kg1, kg2, ground_truth = ToyEADataset().load()

    assert len(kg1) == 3
    assert len(kg2) == 3
    assert len(ground_truth) == 3

    kg1_ids = {entity.id for entity in kg1}
    kg2_ids = {entity.id for entity in kg2}
    for source_id, target_id in ground_truth:
        assert source_id in kg1_ids
        assert target_id in kg2_ids


def test_toy_ea_dataset_solvable_by_string_similarity() -> None:
    kg1, kg2, ground_truth = ToyEADataset().load()

    linker = StringSimilarityLinker(source_field="label", target_field="label", metric="jaccard")
    results = linker.link(kg1, kg2, blocking=LabelOverlap(max_matches=3))
    predictions = [(result.source_id, result.target_id) for result in results]

    report = Evaluator.evaluate(predictions=predictions, ground_truth=ground_truth)
    assert report.metrics["precision@1"] == 1.0


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


def test_toy_wsa_dataset_shape() -> None:
    dict1, dict2, ground_truth = ToyWSADataset().load()

    assert len(dict1) == 4
    assert len(dict2) == 4
    assert len(ground_truth) == 4

    dict1_ids = {entity.id for entity in dict1}
    dict2_ids = {entity.id for entity in dict2}
    for source_id, target_id in ground_truth:
        assert source_id in dict1_ids
        assert target_id in dict2_ids


def test_toy_wsa_dataset_solvable_by_string_similarity() -> None:
    dict1, dict2, ground_truth = ToyWSADataset().load()

    linker = StringSimilarityLinker(
        source_field="description", target_field="description", metric="word_overlap"
    )
    results = linker.link(dict1, dict2)
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

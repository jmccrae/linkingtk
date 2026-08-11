from linkingtk.algorithms.ea import EntMatcherLinker
from linkingtk.algorithms.feature_classifier import FeatureClassifierLinker
from linkingtk.blocking import LabelOverlap
from linkingtk.datasets.toy import ToyEADataset
from linkingtk.eval import Evaluator


def test_entmatcher_is_a_preconfigured_feature_classifier_linker() -> None:
    linker = EntMatcherLinker()
    assert isinstance(linker, FeatureClassifierLinker)
    assert linker.matching == "optimal"


def test_entmatcher_fits_and_links_toy_ea_dataset() -> None:
    kg1, kg2, ground_truth = ToyEADataset().load()
    blocking = LabelOverlap(ngram_size=1, max_matches=10)

    linker = EntMatcherLinker().fit(kg1, kg2, ground_truth, blocking=blocking, random_state=0)
    results = linker.link(kg1, kg2, blocking=blocking)
    predictions = [(r.source_id, r.target_id) for r in results]

    report = Evaluator.evaluate(predictions=predictions, ground_truth=ground_truth)
    assert report.metrics["precision@1"] == 1.0

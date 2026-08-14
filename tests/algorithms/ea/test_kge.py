import numpy as np
import pytest
from pykeen.triples import TriplesFactory

from linkingtk.algorithms.ea import KGELinker
from linkingtk.blocking.base import BlockingStrategy
from linkingtk.core.entity import Entity
from linkingtk.eval import Evaluator
from linkingtk.exceptions import LinkingTKError
from linkingtk.utils.graph import build_id_mappings, map_triples_to_ids

# Two isomorphic 4-node chains ("next"-linked), fully seeded (every
# ground-truth pair is given to fit() as a seed alignment triple) -- this
# is a pipeline-correctness check (does training/scoring/matching wire up
# correctly and recover what it was directly taught), not a generalization
# benchmark, matching examples/feature_classifier_ea.py's framing. See
# tests/algorithms/ea/test_entmatcher.py for the equivalent
# non-KGE-specific pattern.
_KG1 = [Entity(id=f"kg1:{c}", labels=[c]) for c in "abcd"]
_KG2 = [Entity(id=f"kg2:{c}", labels=[c]) for c in "wxyz"]
_GRAPH = [
    ("kg1:a", "next", "kg1:b"),
    ("kg1:b", "next", "kg1:c"),
    ("kg1:c", "next", "kg1:d"),
    ("kg2:w", "next", "kg2:x"),
    ("kg2:x", "next", "kg2:y"),
    ("kg2:y", "next", "kg2:z"),
]
_GROUND_TRUTH = [("kg1:a", "kg2:w"), ("kg1:b", "kg2:x"), ("kg1:c", "kg2:y"), ("kg1:d", "kg2:z")]


class _AllPairs(BlockingStrategy):
    """Blocking strategy that lets every pair through -- entities here don't
    share labels across KGs, so ExactMatch's default blocking would find
    nothing.
    """

    def candidate_pairs(
        self, dataset1: list[Entity], dataset2: list[Entity]
    ) -> list[tuple[Entity, Entity]]:
        return [(e1, e2) for e1 in dataset1 for e2 in dataset2]


class TestFitAndLink:
    def test_recovers_seeded_alignment(self) -> None:
        linker = KGELinker(embedding_dim=16, num_epochs=100)
        linker.fit(_KG1, _KG2, _GROUND_TRUTH, graph=_GRAPH, random_state=0)

        results = linker.link(_KG1, _KG2, blocking=_AllPairs())
        predictions = [(r.source_id, r.target_id) for r in results]

        report = Evaluator.evaluate(predictions=predictions, ground_truth=_GROUND_TRUTH)
        assert report.metrics["precision@1"] == 1.0

    def test_source_and_target_embedding_are_identical(self) -> None:
        # KGELinker has no source/target asymmetry (unlike MTransE/KDCoE's
        # projected-source scoring) -- both accessors return the same
        # underlying vector.
        linker = KGELinker(embedding_dim=16, num_epochs=20)
        linker.fit(_KG1, _KG2, _GROUND_TRUTH, graph=_GRAPH, random_state=0)

        entity_id = _KG1[0].id
        assert np.array_equal(
            linker.source_embedding(entity_id), linker.target_embedding(entity_id)
        )

    @pytest.mark.parametrize("model", ["DistMult", "RotatE"])
    def test_other_pykeen_models_train_and_score_without_error(self, model: str) -> None:
        # Not asserting accuracy here (that's covered for the default
        # TransE above) -- just that the model_resolver-based model choice
        # and, for RotatE, the complex-embedding-flattening path both work
        # end to end without crashing.
        linker = KGELinker(model=model, embedding_dim=8, num_epochs=5)
        linker.fit(_KG1, _KG2, _GROUND_TRUTH, graph=_GRAPH, random_state=0)

        results = linker.link(_KG1, _KG2, blocking=_AllPairs())

        assert {r.source_id for r in results} == {entity.id for entity in _KG1}


class TestIdMappingMatchesPykeen:
    def test_fit_uses_pykeen_compatible_id_mapping(self) -> None:
        # Guards the fit() refactor from #12: build_id_mappings/
        # map_triples_to_ids must reproduce exactly what
        # TriplesFactory.from_labeled_triples built internally before that
        # refactor, so training behavior is unchanged.
        triples = _GRAPH + [(s, "__seed_alignment__", t) for s, t in _GROUND_TRUTH]
        entity_to_id, relation_to_id = build_id_mappings(triples)
        mapped = map_triples_to_ids(triples, entity_to_id, relation_to_id)

        reference = TriplesFactory.from_labeled_triples(np.array(triples, dtype=str))

        assert entity_to_id == reference.entity_to_id
        assert relation_to_id == reference.relation_to_id
        # Row order isn't semantically meaningful (pykeen's own row order
        # here is a sorted-triples side effect, not a training
        # requirement -- SLCWATrainingLoop shuffles anyway), so compare as
        # sets of rows rather than exact list equality.
        assert {tuple(row) for row in mapped.tolist()} == {
            tuple(row) for row in reference.mapped_triples.tolist()
        }


class TestErrors:
    def test_link_before_fit_raises(self) -> None:
        linker = KGELinker()
        with pytest.raises(LinkingTKError, match="before fit"):
            linker.link(_KG1, _KG2, blocking=_AllPairs())

    def test_fit_with_no_ground_truth_ids_in_datasets_raises(self) -> None:
        linker = KGELinker()
        bogus_ground_truth = [("not:an:id", "also:not:an:id")]
        with pytest.raises(LinkingTKError, match="no seed alignment triples"):
            linker.fit(_KG1, _KG2, bogus_ground_truth, graph=_GRAPH, random_state=0)

    def test_link_with_entity_absent_from_training_graph_raises(self) -> None:
        linker = KGELinker(embedding_dim=8, num_epochs=5)
        linker.fit(_KG1, _KG2, _GROUND_TRUTH, graph=_GRAPH, random_state=0)

        unseen = Entity(id="kg1:unseen", labels=["unseen"])
        with pytest.raises(LinkingTKError, match="no trained embedding"):
            linker.link([unseen], _KG2, blocking=_AllPairs())

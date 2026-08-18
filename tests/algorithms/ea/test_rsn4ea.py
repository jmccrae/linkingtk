import numpy as np
import pytest
import torch

from linkingtk.algorithms.ea import RSN4EALinker
from linkingtk.blocking.base import BlockingStrategy
from linkingtk.core.entity import Entity
from linkingtk.eval import Evaluator
from linkingtk.exceptions import LinkingTKError

# Two isomorphic 4-node directed cycles ("next"-linked) -- unlike siblings'
# open-chain fixtures, RSN4EA's path sampler needs every entity to have
# out-degree >= 1 (a walk that dead-ends before max_length is dropped
# entirely), which a cycle guarantees even before reverse-edge doubling.
# Fully seeded (every ground-truth pair given to fit() as a seed pair) --
# a pipeline-correctness check, not a generalization benchmark. See
# test_rsn4ea_benchmark.py for held-out generalization.
_KG1 = [Entity(id=f"kg1:{c}", labels=[c]) for c in "abcd"]
_KG2 = [Entity(id=f"kg2:{c}", labels=[c]) for c in "wxyz"]
_GRAPH = [
    ("kg1:a", "next", "kg1:b"),
    ("kg1:b", "next", "kg1:c"),
    ("kg1:c", "next", "kg1:d"),
    ("kg1:d", "next", "kg1:a"),
    ("kg2:w", "next", "kg2:x"),
    ("kg2:x", "next", "kg2:y"),
    ("kg2:y", "next", "kg2:z"),
    ("kg2:z", "next", "kg2:w"),
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


def _linker(**overrides: object) -> RSN4EALinker:
    params: dict[str, object] = dict(
        hidden_size=16,
        num_layers=2,
        max_length=7,
        num_epochs=200,
        batch_size=8,
        repeat_times=6,
        learning_rate=0.01,
    )
    params.update(overrides)
    return RSN4EALinker(**params)  # type: ignore[arg-type]


class TestFitAndLink:
    def test_recovers_seeded_alignment(self) -> None:
        linker = _linker()
        linker.fit(_KG1, _KG2, _GROUND_TRUTH, graph=_GRAPH, random_state=0)

        results = linker.link(_KG1, _KG2, blocking=_AllPairs())
        predictions = [(r.source_id, r.target_id) for r in results]

        report = Evaluator.evaluate(predictions=predictions, ground_truth=_GROUND_TRUTH)
        assert report.metrics["precision@1"] == 1.0

    def test_early_stopping_runs_without_error(self) -> None:
        # Not asserting an exact stopping epoch (an internal,
        # timing-sensitive detail) -- just that passing val_ground_truth
        # exercises the early-stopping path end to end without crashing
        # and still produces a usable fitted linker.
        linker = _linker(num_epochs=50)
        linker.fit(
            _KG1,
            _KG2,
            _GROUND_TRUTH,
            graph=_GRAPH,
            random_state=0,
            val_ground_truth=_GROUND_TRUTH,
            patience=1,
            eval_every=5,
        )

        results = linker.link(_KG1, _KG2, blocking=_AllPairs())
        assert {r.source_id for r in results} == {entity.id for entity in _KG1}

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
    def test_recovers_seeded_alignment_on_cuda(self) -> None:
        linker = _linker(device="cuda")
        linker.fit(_KG1, _KG2, _GROUND_TRUTH, graph=_GRAPH, random_state=0)

        results = linker.link(_KG1, _KG2, blocking=_AllPairs())
        predictions = [(r.source_id, r.target_id) for r in results]

        report = Evaluator.evaluate(predictions=predictions, ground_truth=_GROUND_TRUTH)
        assert report.metrics["precision@1"] == 1.0


class TestErrors:
    def test_invalid_max_length_raises(self) -> None:
        with pytest.raises(LinkingTKError, match="max_length"):
            RSN4EALinker(max_length=4)
        with pytest.raises(LinkingTKError, match="max_length"):
            RSN4EALinker(max_length=1)

    def test_link_before_fit_raises(self) -> None:
        linker = _linker()
        with pytest.raises(LinkingTKError, match="before fit"):
            linker.link(_KG1, _KG2, blocking=_AllPairs())

    def test_fit_with_no_ground_truth_ids_in_graph_raises(self) -> None:
        linker = _linker()
        bogus_ground_truth = [("not:an:id", "also:not:an:id")]
        with pytest.raises(LinkingTKError, match="no seed pairs"):
            linker.fit(_KG1, _KG2, bogus_ground_truth, graph=_GRAPH, random_state=0)

    def test_link_with_entity_absent_from_training_graph_raises(self) -> None:
        linker = _linker(num_epochs=5)
        linker.fit(_KG1, _KG2, _GROUND_TRUTH, graph=_GRAPH, random_state=0)

        unseen = Entity(id="kg1:unseen", labels=["unseen"])
        with pytest.raises(LinkingTKError, match="no trained embedding"):
            linker.link([unseen], _KG2, blocking=_AllPairs())


class TestNumpyArrayReturn:
    def test_embedding_is_numpy_array(self) -> None:
        linker = _linker(num_epochs=5)
        linker.fit(_KG1, _KG2, _GROUND_TRUTH, graph=_GRAPH, random_state=0)

        assert isinstance(linker.source_embedding("kg1:a"), np.ndarray)
        assert linker.source_embedding("kg1:a").shape == (16,)

import numpy as np
import pytest

from linkingtk.algorithms.ea._simple_hhea_structure import build_structure_embeddings
from linkingtk.algorithms.ea._simple_hhea_torch import build_time_histogram
from linkingtk.algorithms.ea.simple_hhea import SimpleHHEALinker
from linkingtk.blocking.base import BlockingStrategy
from linkingtk.core.entity import Entity
from linkingtk.eval import Evaluator, rank_exhaustive
from linkingtk.exceptions import LinkingTKError

_TINY_MODEL = "hf-internal-testing/tiny-random-AlbertModel"

# Two isomorphic 4-node chains ("next"-linked), fully seeded -- same
# pipeline-correctness framing as tests/algorithms/ea/test_kge.py (does
# fit/link/matching wire up correctly and recover what it was directly
# taught), not a generalization benchmark. Name embeddings come from an
# untrained tiny ALBERT (pure noise), so recovering the alignment here
# genuinely exercises the structural/training path, not the name branch.
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
        linker = SimpleHHEALinker(
            name_model=_TINY_MODEL,
            emb_size=8,
            structure_size=4,
            structure_dim=8,
            num_epochs=1000,
            learning_rate=0.05,
        )
        linker.fit(_KG1, _KG2, _GROUND_TRUTH, graph=_GRAPH, random_state=0)

        results = linker.link(_KG1, _KG2, blocking=_AllPairs())
        predictions = [(r.source_id, r.target_id) for r in results]

        report = Evaluator.evaluate(predictions=predictions, ground_truth=_GROUND_TRUTH)
        assert report.metrics["precision@1"] == 1.0

    def test_source_and_target_embedding_work_with_rank_exhaustive(self) -> None:
        linker = SimpleHHEALinker(
            name_model=_TINY_MODEL, emb_size=8, structure_size=4, structure_dim=8, num_epochs=20
        )
        linker.fit(_KG1, _KG2, _GROUND_TRUTH, graph=_GRAPH, random_state=0)

        ranked = rank_exhaustive(linker, _KG1, _KG2)

        assert {source_id for source_id, _ in ranked} == {e.id for e in _KG1}
        assert all(len(targets) == len(_KG2) for _, targets in ranked)


class TestTemporalBranch:
    def test_none_temporal_triples_disables_time_branch(self) -> None:
        linker = SimpleHHEALinker(
            name_model=_TINY_MODEL, emb_size=4, structure_size=2, structure_dim=4, num_epochs=5
        )
        linker.fit(_KG1, _KG2, _GROUND_TRUTH, graph=_GRAPH, temporal_triples=None, random_state=0)

        assert linker.source_embedding("kg1:a").shape == (4,)

    def test_temporal_triples_enable_time_branch(self) -> None:
        temporal_triples = [
            ("kg1:a", "next", "kg1:b", "2012-01", "2012-03"),
            ("kg2:w", "next", "kg2:x", "2012-01", None),
        ]
        linker = SimpleHHEALinker(
            name_model=_TINY_MODEL,
            emb_size=4,
            structure_size=2,
            time_size=2,
            structure_dim=4,
            num_epochs=5,
        )
        linker.fit(
            _KG1,
            _KG2,
            _GROUND_TRUTH,
            graph=_GRAPH,
            temporal_triples=temporal_triples,
            random_state=0,
        )

        assert linker.source_embedding("kg1:a").shape == (4,)


class TestErrors:
    def test_link_before_fit_raises(self) -> None:
        linker = SimpleHHEALinker()
        with pytest.raises(LinkingTKError, match="before fit"):
            linker.link(_KG1, _KG2, blocking=_AllPairs())

    def test_fit_with_no_ground_truth_ids_in_datasets_raises(self) -> None:
        linker = SimpleHHEALinker(name_model=_TINY_MODEL)
        bogus_ground_truth = [("not:an:id", "also:not:an:id")]
        with pytest.raises(LinkingTKError, match="no training pairs"):
            linker.fit(_KG1, _KG2, bogus_ground_truth, graph=_GRAPH, random_state=0)


class TestStructureEmbeddings:
    def test_train_pair_merge_gives_identical_vectors(self) -> None:
        entity_ids = [e.id for e in _KG1 + _KG2]
        embeddings = build_structure_embeddings(
            entity_ids,
            _GRAPH,
            _GROUND_TRUTH,
            dimensions=4,
            walk_length=5,
            num_walks=2,
            random_state=0,
        )

        assert np.array_equal(embeddings["kg1:a"], embeddings["kg2:w"])
        assert np.array_equal(embeddings["kg1:b"], embeddings["kg2:x"])

    def test_entity_absent_from_triples_gets_a_fallback_vector(self) -> None:
        embeddings = build_structure_embeddings(
            ["kg1:a", "kg1:b", "lonely:1"],
            [("kg1:a", "next", "kg1:b")],
            [],
            dimensions=4,
            walk_length=5,
            num_walks=2,
            random_state=0,
        )

        assert embeddings["lonely:1"].shape == (4,)
        assert not np.array_equal(embeddings["lonely:1"], embeddings["kg1:a"])


class TestTimeHistogram:
    def test_range_increments_every_month_bin(self) -> None:
        histogram = build_time_histogram(["e1"], [("e1", "r", "t", "2012-01", "2012-03")])

        nonzero = np.flatnonzero(histogram[0])
        assert list(nonzero) == list(range(nonzero[0], nonzero[0] + 3))

    def test_unresolvable_end_falls_back_to_single_point(self) -> None:
        histogram = build_time_histogram(["e1"], [("e1", "r", "t", "2012-01", None)])

        assert histogram[0].sum() == 1.0

    def test_both_unresolvable_is_skipped(self) -> None:
        histogram = build_time_histogram(["e1"], [("e1", "r", "t", None, None)])

        assert histogram[0].sum() == 0.0

    def test_pre_1995_falls_into_catch_all_bin(self) -> None:
        histogram = build_time_histogram(["e1"], [("e1", "r", "t", "1980-01", "1980-01")])

        assert histogram[0][0] == 1.0

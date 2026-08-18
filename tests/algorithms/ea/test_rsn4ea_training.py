"""Unit tests for the pure-numpy pieces of
[RSN4EALinker][linkingtk.algorithms.ea.rsn4ea.RSN4EALinker] --
independently testable without torch installed, same precedent as
test_bootea.py's private-helper tests.
"""

from __future__ import annotations

import numpy as np

from linkingtk.algorithms.ea._rsn4ea_training import build_augmented_kb, sample_paths

# Two isomorphic 4-node directed cycles (ids 0-3 and 10-13), "next"-linked
# with distinct relation ids (0 for KG1, 1 for KG2) -- directed cycles
# guarantee every entity has out-degree >= 1 even before reverse-edge
# doubling, which sample_paths needs to avoid dead-ending every walk.
_TRIPLES = np.array(
    [
        [0, 0, 1],
        [1, 0, 2],
        [2, 0, 3],
        [3, 0, 0],
        [10, 1, 11],
        [11, 1, 12],
        [12, 1, 13],
        [13, 1, 10],
    ],
    dtype=np.int64,
)
_SEED_PAIRS = [(0, 10), (1, 11), (2, 12), (3, 13)]


def _augment(seed_pairs: list[tuple[int, int]]) -> tuple[np.ndarray, int]:
    return build_augmented_kb(_TRIPLES, num_entities=14, num_relations=2, seed_pairs=seed_pairs)


class TestBuildAugmentedKb:
    def test_reverse_edges_added_without_seed_pairs(self) -> None:
        kb, num_relations = _augment([])

        assert num_relations == 4
        # No aliasing -> only the 8 original + 8 reverse triples survive.
        assert kb.shape == (16, 3)
        assert (1, 2, 0) in {tuple(row) for row in kb}  # reverse of (0, 0, 1)

    def test_alias_variants_added_with_seed_pairs(self) -> None:
        kb, num_relations = _augment(_SEED_PAIRS)

        rows = {tuple(row) for row in kb}
        # (0, 0, 1) aliased via 0->10, 1->11: (10, 0, 1), (0, 0, 11), (10, 0, 11).
        assert (0, 0, 1) in rows
        assert (10, 0, 1) in rows
        assert (0, 0, 11) in rows
        assert (10, 0, 11) in rows

    def test_relation_never_aliased(self) -> None:
        # relation ids in every row must come only from {0, 1, 2, 3}
        # (the doubled original vocabulary) -- never substituted.
        kb, num_relations = _augment(_SEED_PAIRS)
        assert set(kb[:, 1].tolist()) <= set(range(num_relations))

    def test_empty_triples(self) -> None:
        kb, num_relations = build_augmented_kb(
            np.empty((0, 3), dtype=np.int64), num_entities=4, num_relations=2, seed_pairs=[]
        )
        assert kb.shape == (0, 3)
        assert num_relations == 4


class TestSamplePaths:
    def test_output_shape_and_alternation(self) -> None:
        kb, _ = _augment(_SEED_PAIRS)
        rng = np.random.default_rng(0)

        paths = sample_paths(
            kb, _SEED_PAIRS, rng, max_length=5, alpha=0.7, beta=0.7, repeat_times=3
        )

        assert paths.ndim == 2
        assert paths.shape[1] == 5
        assert len(paths) > 0
        known_entities = set(kb[:, 0].tolist()) | set(kb[:, 2].tolist())
        known_relations = set(kb[:, 1].tolist())
        assert set(paths[:, 0::2].flatten().tolist()) <= known_entities
        assert set(paths[:, 1::2].flatten().tolist()) <= known_relations

    def test_paths_cross_between_kgs(self) -> None:
        # With every entity seed-paired, some sampled walk should visit
        # both the 0-3 and 10-13 id ranges -- otherwise the alias
        # expansion isn't actually connecting the two KGs.
        kb, _ = _augment(_SEED_PAIRS)
        rng = np.random.default_rng(1)

        paths = sample_paths(
            kb, _SEED_PAIRS, rng, max_length=7, alpha=0.7, beta=0.7, repeat_times=8
        )

        crosses = [row for row in paths if (row < 10).any() and (row >= 10).any()]
        assert crosses

    def test_max_paths_caps_output(self) -> None:
        kb, _ = _augment(_SEED_PAIRS)
        rng = np.random.default_rng(0)

        paths = sample_paths(
            kb, _SEED_PAIRS, rng, max_length=5, alpha=0.7, beta=0.7, repeat_times=3, max_paths=5
        )

        assert len(paths) == 5

    def test_dead_end_drops_walk(self) -> None:
        # A single one-way edge, no cycle -- no path of length 5 can be
        # sampled at all (the walk dead-ends after the first hop).
        triples = np.array([[0, 0, 1]], dtype=np.int64)
        rng = np.random.default_rng(0)

        paths = sample_paths(triples, [], rng, max_length=5, alpha=0.7, beta=0.7, repeat_times=3)

        assert len(paths) == 0

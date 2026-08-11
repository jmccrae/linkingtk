from linkingtk.algorithms.matching import GreedyMatcher, OptimalMatcher


class TestMatchingStrategies:
    """Unit tests isolating the matching algorithms with hand-crafted scores
    under exact control -- this is where the value-add of a globally
    optimal assignment over independent per-source argmax is
    deterministically demonstrable."""

    def test_greedy_allows_a_target_collision(self) -> None:
        candidates = {
            "A1": [("T1", 0.9), ("T2", 0.1)],
            "A2": [("T1", 0.95), ("T2", 0.80)],
        }

        results = GreedyMatcher().match(candidates)

        assert {(r.source_id, r.target_id) for r in results} == {("A1", "T1"), ("A2", "T1")}

    def test_optimal_resolves_the_same_collision(self) -> None:
        candidates = {
            "A1": [("T1", 0.9), ("T2", 0.1)],
            "A2": [("T1", 0.95), ("T2", 0.80)],
        }

        results = OptimalMatcher().match(candidates)

        assert {(r.source_id, r.target_id) for r in results} == {("A1", "T1"), ("A2", "T2")}

    def test_optimal_drops_sources_left_with_no_real_candidate(self) -> None:
        # 2 sources compete for 1 target; whichever loses the assignment
        # should be dropped entirely, not paired with a nonexistent candidate.
        candidates = {
            "A1": [("T1", 0.9)],
            "A2": [("T1", 0.95)],
        }

        results = OptimalMatcher().match(candidates)

        assert len(results) == 1
        assert results[0].source_id == "A2"
        assert results[0].target_id == "T1"

    def test_optimal_matches_empty_input(self) -> None:
        assert OptimalMatcher().match({}) == []

    def test_optimal_resolves_collision_with_unbounded_scores(self) -> None:
        # word_overlap-style raw counts, not bounded to [0, 1] -- exercises
        # the dynamically-computed sentinel cost rather than a fixed one.
        candidates = {
            "A1": [("T1", 5.0), ("T2", 1.0)],
            "A2": [("T1", 6.0), ("T2", 4.0)],
        }

        results = OptimalMatcher().match(candidates)

        assert {(r.source_id, r.target_id) for r in results} == {("A1", "T1"), ("A2", "T2")}

"""Benchmarks KevLinker (zero-shot pairwise "same entity?" questions on a
Kev decision model) on Entity Linking and Word Sense Disambiguation.

Needs a running Kev server -- see
[linkingtk.algorithms.kev][linkingtk.algorithms.kev] for setup; point
`--kev-url` at it and pass `--label` (e.g. `kev-4b`) to name the row.

Datasets and candidate generation match the rest of the repo's EL/WSD
benchmarks so numbers are comparable:

- EL: `AidaConllDataset`'s test split against the test split's own gold
  KB entities, with `LabelOverlap(ngram_size=3, max_matches=30)` blocking
  (same as `comparative_benchmark.py`/`refined_benchmark.py`). Baseline:
  `StringSimilarityLinker` (the comparative harness's MVP tier).
- WSD: SemEval-2007 via `UfsacDataset` against a real `WnEntitySource`,
  with `ExactMatch(top_k=50)` blocking (same as `llm_benchmark.py`),
  filtered to the mention's own part of speech. Baseline: most-frequent
  sense, i.e. the first WordNet candidate of that POS.

Both are subsampled to `--max-mentions` sources (seeded; `0` = all).
Besides P@1/F1 and Hits@k/MRR, each task reports its blocking recall
ceiling and the P(yes) distribution for gold vs. non-gold candidate
pairs, to check Kev's scores actually separate the two rather than
saturating.

Run with: `uv run python examples/kev_benchmark.py --task both --label kev-0.8b`
"""

from __future__ import annotations

import argparse
import random
import statistics
from functools import partial
from pathlib import Path

from linkingtk.algorithms.kev import KevClient, KevLinker, QuestionType
from linkingtk.algorithms.string_similarity import StringSimilarityLinker
from linkingtk.blocking.base import BlockingStrategy
from linkingtk.blocking.exact import ExactMatch
from linkingtk.blocking.label_overlap import LabelOverlap
from linkingtk.blocking.pos import WordNetPosFilter
from linkingtk.core.entity import Entity
from linkingtk.core.source import EntitySource
from linkingtk.datasets.aida_conll import AidaConllDataset
from linkingtk.datasets.ufsac import UfsacDataset
from linkingtk.eval import Evaluator
from linkingtk.eval.harness import BenchmarkRun, format_table, run_benchmarks
from linkingtk.eval.report import EvaluationReport
from linkingtk.matchers import GreedyMatcher

_SEED = 20260827
_TOP_K = [1, 5]


def _sample(
    mentions: list[Entity], ground_truth: list[tuple[str, str]], max_mentions: int | None
) -> tuple[list[Entity], list[tuple[str, str]]]:
    gold_sources = {s for s, _ in ground_truth}
    mentions = [m for m in mentions if m.id in gold_sources]
    if max_mentions is not None and max_mentions < len(mentions):
        mentions = random.Random(_SEED).sample(mentions, max_mentions)
    ids = {m.id for m in mentions}
    return mentions, [(s, t) for s, t in ground_truth if s in ids]


def _report_from_scores(
    scores: dict[str, list[tuple[str, float]]], ground_truth: list[tuple[str, str]]
) -> EvaluationReport:
    results = GreedyMatcher().match(scores)
    unranked = Evaluator.evaluate([(r.source_id, r.target_id) for r in results], ground_truth)
    # Not Evaluator.evaluate_ranked: it counts every gold row, so a WSD
    # instance with several gold senses would be scored more than once.
    # Rank of the first gold hit per source instead, one query per source.
    gold: dict[str, set[str]] = {}
    for s, t in ground_truth:
        gold.setdefault(s, set()).add(t)
    ranked = {r.source_id: [r.target_id, *r.alternatives] for r in results}
    ranks = [
        next((i + 1 for i, t in enumerate(ranked.get(s, [])) if t in targets), None)
        for s, targets in gold.items()
    ]
    metrics = dict(unranked.metrics)
    for k in _TOP_K:
        metrics[f"Hits@{k}"] = sum(1 for r in ranks if r is not None and r <= k) / len(gold)
    metrics["MRR"] = sum(1 / r for r in ranks if r is not None) / len(gold)
    return EvaluationReport(metrics=metrics)


def _require_no_failures(linker: KevLinker) -> None:
    """Abort rather than report metrics where failed requests silently scored 0.0."""
    if linker.failed_requests:
        raise SystemExit(
            f"{linker.failed_requests}/{linker.total_requests} Kev requests failed "
            "(see the server log, e.g. out of GPU memory); results would be invalid"
        )


def _diagnostics(
    name: str,
    mentions: list[Entity],
    dataset2: list[Entity] | EntitySource,
    blocking: BlockingStrategy,
    ground_truth: list[tuple[str, str]],
    scores: dict[str, list[tuple[str, float]]],
) -> None:
    gold: dict[str, set[str]] = {}
    for s, t in ground_truth:
        gold.setdefault(s, set()).add(t)
    candidates: dict[str, set[str]] = {}
    for e1, e2 in blocking.candidate_pairs(mentions, dataset2):
        candidates.setdefault(e1.id, set()).add(e2.id)
    ceiling = sum(1 for s, ts in gold.items() if ts & candidates.get(s, set())) / len(gold)
    mean_candidates = statistics.mean(len(c) for c in candidates.values()) if candidates else 0.0
    gold_p = [p for s, pairs in scores.items() for t, p in pairs if t in gold.get(s, set())]
    other_p = [p for s, pairs in scores.items() for t, p in pairs if t not in gold.get(s, set())]

    def quantiles(values: list[float]) -> str:
        if len(values) < 2:
            return "n/a"
        q = statistics.quantiles(values, n=4)
        mean = statistics.mean(values)
        return f"n={len(values)} mean={mean:.3f} q1/med/q3={q[0]:.3f}/{q[1]:.3f}/{q[2]:.3f}"

    print(f"\n[{name}] sources={len(gold)} mean candidates={mean_candidates:.1f}")
    print(f"[{name}] blocking recall ceiling (gold in candidates): {ceiling:.3f}")
    print(f"[{name}] P(yes) gold:     {quantiles(gold_p)}")
    print(f"[{name}] P(yes) non-gold: {quantiles(other_p)}")


def el_runs(
    client: KevClient, label: str, question_types: list[QuestionType], max_mentions: int | None
) -> list[BenchmarkRun]:
    dataset = AidaConllDataset()
    mentions, kb, _ = dataset.load()
    _train, test_pairs, _val = dataset.load_splits()
    kb_by_id = {e.id: e for e in kb}
    test_kb = [kb_by_id[t] for t in sorted({t for _, t in test_pairs})]
    mentions, ground_truth = _sample(mentions, test_pairs, max_mentions)
    blocking = LabelOverlap(ngram_size=3, max_matches=30)

    def baseline() -> EvaluationReport:
        results = StringSimilarityLinker().link(mentions, test_kb, blocking=blocking)
        scores = {r.source_id: [(r.target_id, 1.0)] for r in results}
        return _report_from_scores(scores, ground_truth)

    def kev(question_type: QuestionType) -> EvaluationReport:
        linker = KevLinker(client, task="el", question_type=question_type)
        scores = linker.score_candidates(mentions, test_kb, blocking)
        _require_no_failures(linker)
        _diagnostics(f"EL/{question_type}", mentions, test_kb, blocking, ground_truth, scores)
        return _report_from_scores(scores, ground_truth)

    return [
        BenchmarkRun("EL", "MVP", "StringSimilarityLinker", "AIDA-CoNLL test", baseline),
        *(
            BenchmarkRun(
                "EL",
                "Decision model",
                f"KevLinker ({label}, {qt})",
                "AIDA-CoNLL test",
                partial(kev, qt),
            )
            for qt in question_types
        ),
    ]


def wsd_runs(
    client: KevClient,
    label: str,
    question_types: list[QuestionType],
    ufsac_path: Path,
    max_mentions: int | None,
) -> list[BenchmarkRun]:
    mentions, senses, ground_truth = UfsacDataset(source=str(ufsac_path)).load()
    mentions, ground_truth = _sample(mentions, ground_truth, max_mentions)
    blocking = WordNetPosFilter(ExactMatch(top_k=50))

    def baseline() -> EvaluationReport:
        # WordNet lists senses most-frequent first; decreasing scores keep that order.
        scores: dict[str, list[tuple[str, float]]] = {}
        for e1, e2 in blocking.candidate_pairs(mentions, senses):
            ranked = scores.setdefault(e1.id, [])
            ranked.append((e2.id, -float(len(ranked))))
        return _report_from_scores(scores, ground_truth)

    def kev(question_type: QuestionType) -> EvaluationReport:
        linker = KevLinker(client, task="wsd", question_type=question_type)
        scores = linker.score_candidates(mentions, senses, blocking)
        _require_no_failures(linker)
        _diagnostics(f"WSD/{question_type}", mentions, senses, blocking, ground_truth, scores)
        return _report_from_scores(scores, ground_truth)

    return [
        BenchmarkRun("WSD", "MVP", "Most frequent sense", "SemEval-2007", baseline),
        *(
            BenchmarkRun(
                "WSD",
                "Decision model",
                f"KevLinker ({label}, {qt})",
                "SemEval-2007",
                partial(kev, qt),
            )
            for qt in question_types
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["el", "wsd", "both"], default="both")
    parser.add_argument("--kev-url", default="http://127.0.0.1:8008")
    parser.add_argument("--label", default="kev")
    parser.add_argument(
        "--question-types",
        nargs="+",
        choices=["noul", "choice"],
        default=["noul", "choice"],
        help="KevLinker question types to run, one result row each",
    )
    parser.add_argument("--max-mentions", type=int, default=200)
    parser.add_argument(
        "--ufsac-path",
        type=Path,
        default=Path("~/data/ufsac-public-2.1/raganato_semeval2007.xml").expanduser(),
    )
    args = parser.parse_args()
    max_mentions = args.max_mentions if args.max_mentions > 0 else None

    client = KevClient(args.kev_url)
    client.health()

    runs: list[BenchmarkRun] = []
    if args.task in ("el", "both"):
        runs += el_runs(client, args.label, args.question_types, max_mentions)
    if args.task in ("wsd", "both"):
        runs += wsd_runs(client, args.label, args.question_types, args.ufsac_path, max_mentions)
    print("\n" + format_table(run_benchmarks(runs)))


if __name__ == "__main__":
    main()

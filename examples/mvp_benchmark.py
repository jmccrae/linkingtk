"""String-similarity baselines on all four tasks.

Runs the same StringSimilarityLinker baseline (LeskLinker for WSD is a
preconfigured instance of it) against a bundled toy dataset for each of Entity
Alignment (EA), Entity Linking (EL), Word Sense Disambiguation (WSD) and Word
Sense Alignment (WSA), scoring each with Evaluator.evaluate.

Run with: `uv run python examples/mvp_benchmark.py`
"""

from __future__ import annotations

from linkingtk.algorithms.string_similarity import StringSimilarityLinker
from linkingtk.algorithms.wsd import LeskLinker
from linkingtk.blocking import LabelOverlap
from linkingtk.core import AlignmentResult
from linkingtk.datasets import ToyEADataset, ToyELDataset, ToyWSADataset, ToyWSDDataset
from linkingtk.eval import Evaluator


def _score(results: list[AlignmentResult], ground_truth: list[tuple[str, str]]) -> dict[str, float]:
    predictions = [(r.source_id, r.target_id) for r in results]
    return Evaluator.evaluate(predictions=predictions, ground_truth=ground_truth).metrics


def run_ea() -> dict[str, float]:
    kg1, kg2, ground_truth = ToyEADataset().load()
    linker = StringSimilarityLinker(source_field="label", target_field="label", metric="jaccard")
    results = linker.link(kg1, kg2, blocking=LabelOverlap(max_matches=3))
    return _score(results, ground_truth)


def run_el() -> dict[str, float]:
    mentions, kb, ground_truth = ToyELDataset().load()
    linker = StringSimilarityLinker(
        source_field="context", target_field="description", metric="word_overlap"
    )
    results = linker.link(mentions, kb)
    return _score(results, ground_truth)


def run_wsd() -> dict[str, float]:
    mentions, senses, ground_truth = ToyWSDDataset().load()
    results = LeskLinker().link(mentions, senses)
    return _score(results, ground_truth)


def run_wsa() -> dict[str, float]:
    dict1, dict2, ground_truth = ToyWSADataset().load()
    linker = StringSimilarityLinker(
        source_field="description", target_field="description", metric="word_overlap"
    )
    results = linker.link(dict1, dict2)
    return _score(results, ground_truth)


def main() -> None:
    tasks = {"EA": run_ea, "EL": run_el, "WSD": run_wsd, "WSA": run_wsa}
    for name, task in tasks.items():
        print(f"{name}: {task()}")


if __name__ == "__main__":
    main()

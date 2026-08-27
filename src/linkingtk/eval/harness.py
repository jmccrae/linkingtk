"""A small comparison harness for running many linkers/datasets side by side.

Different tasks (EA/EL/WSD/WSA) and tiers (string-similarity, classical ML,
KGE, deep learning, LLM-oriented) need very different fit/link/evaluate
wiring -- ranked vs. unranked evaluation, exhaustive vs. blocking-restricted
ranking, `.fit()` or not. Rather than have the harness guess at that, each
[BenchmarkRun][linkingtk.eval.harness.BenchmarkRun] carries a zero-argument
callable that already does exactly what a normal `examples/*_benchmark.py`
script does, and returns an [EvaluationReport][linkingtk.eval.report.EvaluationReport].
[run_benchmarks][linkingtk.eval.harness.run_benchmarks] only adds timing and,
for runs that make real LLM calls via a
[CountingLlmClient][linkingtk.eval.harness.CountingLlmClient], a call count --
see `examples/comparative_benchmark.py` for real usage.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from linkingtk.eval.report import EvaluationReport
from linkingtk.llm.client import LlmClient, LlmMessage


class CountingLlmClient(LlmClient):
    """Wraps a real [LlmClient][linkingtk.llm.client.LlmClient], counting its calls.

    Purely a benchmarking aid -- delegates every call unchanged and doesn't
    modify `linkingtk.llm.client` at all. Attach the instance to a
    [BenchmarkRun][linkingtk.eval.harness.BenchmarkRun]'s `client` field so
    [run_benchmarks][linkingtk.eval.harness.run_benchmarks] can report an
    LLM-call count alongside wall-clock time.

    Args:
        client: The real client to wrap.
    """

    def __init__(self, client: LlmClient) -> None:
        self._client = client
        self.call_count = 0

    def complete(
        self,
        messages: list[LlmMessage],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        self.call_count += 1
        return self._client.complete(messages, max_tokens=max_tokens, temperature=temperature)

    def complete_structured(
        self,
        messages: list[LlmMessage],
        *,
        schema: dict[str, Any],
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        self.call_count += 1
        return self._client.complete_structured(
            messages, schema=schema, max_tokens=max_tokens, temperature=temperature
        )


@dataclass
class BenchmarkRun:
    """One (task, tier, linker, dataset) comparison cell.

    Attributes:
        task: Display label, e.g. `"EA"`, `"EL"`, `"WSD"`. Results are
            grouped by this in [format_table][linkingtk.eval.harness.format_table].
        tier: Display label, e.g. `"MVP"`, `"Classical ML"`, `"KGE"`,
            `"Deep Learning"`, `"LLM-Oriented"`.
        linker: Display label, e.g. `"ReFinEDLinker"`.
        dataset: Display label, e.g. `"AIDA-CoNLL"`.
        run: Zero-argument callable that fits/links/evaluates and returns
            an [EvaluationReport][linkingtk.eval.report.EvaluationReport].
            Owns all task-specific wiring (blocking, ranked vs. unranked
            evaluation, training) -- the harness only times it.
        client: A [CountingLlmClient][linkingtk.eval.harness.CountingLlmClient]
            used inside `run`, if any -- set this so the resulting
            [BenchmarkResult][linkingtk.eval.harness.BenchmarkResult] reports
            an LLM-call count. `None` for runs that make no LLM calls.
    """

    task: str
    tier: str
    linker: str
    dataset: str
    run: Callable[[], EvaluationReport]
    client: CountingLlmClient | None = None


@dataclass
class BenchmarkResult:
    """The outcome of executing a single [BenchmarkRun][linkingtk.eval.harness.BenchmarkRun]."""

    run: BenchmarkRun
    metrics: dict[str, float | None]
    seconds: float
    llm_calls: int | None


def run_benchmarks(runs: list[BenchmarkRun]) -> list[BenchmarkResult]:
    """Executes each of `runs` in order, timing it and collecting its metrics.

    Args:
        runs: The comparison cells to execute, in order.

    Returns:
        One [BenchmarkResult][linkingtk.eval.harness.BenchmarkResult] per run.
    """
    results = []
    for benchmark_run in runs:
        start = time.monotonic()
        report = benchmark_run.run()
        elapsed = time.monotonic() - start
        llm_calls = benchmark_run.client.call_count if benchmark_run.client is not None else None
        results.append(
            BenchmarkResult(
                run=benchmark_run, metrics=report.metrics, seconds=elapsed, llm_calls=llm_calls
            )
        )
    return results


def _format_metric(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def format_table(results: list[BenchmarkResult]) -> str:
    """Renders `results` as a plain-text table, grouped by `run.task`.

    Each task gets its own section with its own metric columns (the union
    of metric keys across that task's results, in first-seen order), since
    different tasks report different metrics (e.g. EA's `Hits@k`/`MRR` vs.
    EL/WSD's `precision@1`/`recall`/`f1`).

    Args:
        results: Results to render, in the order they should appear within
            each task's section.

    Returns:
        The rendered table.
    """
    tasks: dict[str, list[BenchmarkResult]] = {}
    for result in results:
        tasks.setdefault(result.run.task, []).append(result)

    sections = []
    for task, task_results in tasks.items():
        metric_keys: list[str] = []
        for result in task_results:
            for key in result.metrics:
                if key not in metric_keys:
                    metric_keys.append(key)

        headers = ["Tier", "Linker", "Dataset", *metric_keys, "Seconds", "LLM calls"]
        rows = [
            [
                result.run.tier,
                result.run.linker,
                result.run.dataset,
                *[_format_metric(result.metrics.get(key)) for key in metric_keys],
                f"{result.seconds:.1f}",
                "-" if result.llm_calls is None else str(result.llm_calls),
            ]
            for result in task_results
        ]
        widths = [
            max(len(headers[i]), *(len(row[i]) for row in rows)) if rows else len(headers[i])
            for i in range(len(headers))
        ]
        lines = [
            f"=== {task} ===",
            "  ".join(header.ljust(width) for header, width in zip(headers, widths, strict=True)),
        ]
        lines.extend(
            "  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True))
            for row in rows
        )
        sections.append("\n".join(lines))
    return "\n\n".join(sections)

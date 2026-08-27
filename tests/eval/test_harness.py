from typing import Any

from linkingtk.eval.harness import (
    BenchmarkRun,
    CountingLlmClient,
    format_table,
    run_benchmarks,
)
from linkingtk.eval.report import EvaluationReport
from linkingtk.llm.client import LlmClient, LlmMessage


class _FakeLlmClient(LlmClient):
    def complete(self, messages: list[LlmMessage], **kwargs: Any) -> str:
        return "reply"

    def complete_structured(
        self, messages: list[LlmMessage], *, schema: dict[str, Any], **kwargs: Any
    ) -> dict[str, Any]:
        return {"ok": True}


def test_counting_llm_client_delegates_and_counts() -> None:
    fake = _FakeLlmClient()
    client = CountingLlmClient(fake)

    assert client.complete([LlmMessage(role="user", content="hi")]) == "reply"
    assert client.call_count == 1

    assert client.complete_structured(
        [LlmMessage(role="user", content="hi")], schema={"type": "object"}
    ) == {"ok": True}
    assert client.call_count == 2


def test_run_benchmarks_times_and_collects_metrics() -> None:
    run = BenchmarkRun(
        task="EA",
        tier="MVP",
        linker="FakeLinker",
        dataset="FakeDataset",
        run=lambda: EvaluationReport(metrics={"precision@1": 1.0}),
    )
    (result,) = run_benchmarks([run])

    assert result.metrics == {"precision@1": 1.0}
    assert result.seconds >= 0
    assert result.llm_calls is None


def test_run_benchmarks_reports_llm_call_count() -> None:
    client = CountingLlmClient(_FakeLlmClient())

    def _run() -> EvaluationReport:
        client.complete([LlmMessage(role="user", content="hi")])
        client.complete([LlmMessage(role="user", content="hi")])
        return EvaluationReport(metrics={"precision@1": 0.5})

    run = BenchmarkRun(
        task="EL",
        tier="LLM-Oriented",
        linker="LlmBaseLinker",
        dataset="Toy",
        run=_run,
        client=client,
    )
    (result,) = run_benchmarks([run])

    assert result.llm_calls == 2


def test_format_table_groups_by_task_with_unioned_metric_columns() -> None:
    results = run_benchmarks(
        [
            BenchmarkRun(
                task="EA",
                tier="MVP",
                linker="StringSimilarityLinker",
                dataset="Toy",
                run=lambda: EvaluationReport(metrics={"Hits@1": 0.5, "MRR": 0.6}),
            ),
            BenchmarkRun(
                task="EA",
                tier="KGE",
                linker="KGELinker",
                dataset="Toy",
                run=lambda: EvaluationReport(metrics={"Hits@1": 0.8}),
            ),
            BenchmarkRun(
                task="EL",
                tier="MVP",
                linker="StringSimilarityLinker",
                dataset="Toy",
                run=lambda: EvaluationReport(metrics={"precision@1": 0.9}),
            ),
        ]
    )
    table = format_table(results)

    assert "=== EA ===" in table
    assert "=== EL ===" in table
    ea_section, el_section = table.split("\n\n")
    assert "Hits@1" in ea_section
    assert "MRR" in ea_section
    assert "0.500" in ea_section
    # KGE's row has no MRR value at all -- rendered as a placeholder, not omitted.
    assert "-" in ea_section
    assert "precision@1" in el_section
    assert "MRR" not in el_section


def test_format_table_renders_llm_calls_placeholder_when_absent() -> None:
    results = run_benchmarks(
        [
            BenchmarkRun(
                task="WSD",
                tier="MVP",
                linker="LeskLinker",
                dataset="Toy",
                run=lambda: EvaluationReport(metrics={"precision@1": 1.0}),
            )
        ]
    )
    table = format_table(results)

    assert "LLM calls" in table
    lines = table.splitlines()
    data_row = lines[2]
    assert data_row.rstrip().endswith("-")

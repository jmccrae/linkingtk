# Comparative benchmark harness (all tiers)

Runs one linker per complexity tier -- MVP, Classical ML, KGE (Entity
Alignment only), Deep Learning and LLM-Oriented -- on the same real
dataset for each task (EA, EL, WSD) and prints a single table with both
accuracy and cost (wall-clock time, LLM-call count) side by side, so you
can see the tradeoff directly instead of comparing numbers across
separate pages.

The underlying harness
([`BenchmarkRun`][linkingtk.eval.harness.BenchmarkRun],
[`run_benchmarks`][linkingtk.eval.harness.run_benchmarks],
[`format_table`][linkingtk.eval.harness.format_table]) is reusable for
your own comparisons: build a list of `BenchmarkRun`s, each wrapping a
zero-argument callable that fits/links/evaluates a linker and returns an
[`EvaluationReport`][linkingtk.eval.report.EvaluationReport], and pass it
to `run_benchmarks` -- it times each run and `format_table` renders the
results grouped by task.

Datasets used in this example:

- **EA**: [ICEWS-WIKI](../datasets/real_world_ea.md#icews)
- **EL**: [AIDA-CoNLL](../datasets/index.md)
- **WSD**: UFSAC's SemEval-2007 split

Each task's LLM-Oriented row re-ranks that task's Deep Learning linker's
own top candidates with a local LLM rather than asking the LLM to decide
from scratch, keeping the number of real LLM calls small (`--sample`,
default 30).

Requires: the `kge` optional dependency group; a local UFSAC 2.1 checkout
at `~/data/ufsac-public-2.1/` and the checkpoint produced by
[the full-corpus training example](glossbert_full_training.md); a local
Ollama server with `ollama pull llama2:13b` (or pass `--model` for a
different model). AIDA-CoNLL and ICEWS-WIKI are fetched automatically.

```python
--8<-- "examples/comparative_benchmark.py"
```

Run with:

```bash
uv run python examples/comparative_benchmark.py
```

Or a single, fast task:

```bash
uv run python examples/comparative_benchmark.py --tasks el --sample 3
```

Example output (one task at a time, at a reduced `--sample` for speed --
a full run across all three tasks takes considerably longer):

```text
=== EA ===
Tier           Linker                  Dataset     precision@1  recall  f1     Hits@1  Hits@10  MRR    Seconds  LLM calls
MVP            StringSimilarityLinker  ICEWS-WIKI  0.935        0.935   0.935  -       -        -      0.4      -
Classical ML   EntMatcherLinker        ICEWS-WIKI  0.980        0.963   0.972  -       -        -      10.5     -
KGE            KGELinker (TransE)      ICEWS-WIKI  -            -       -      0.000   0.004    0.003  1288.1   -
Deep Learning  SimpleHHEALinker        ICEWS-WIKI  -            -       -      0.810   0.873    0.832  16.6     -
LLM-Oriented   ChatEALinker            ICEWS-WIKI  -            -       -      0.797   0.858    0.819  253.8    273

=== EL ===
Tier           Linker                   Dataset     precision@1  recall  f1     Seconds  LLM calls
MVP            StringSimilarityLinker   AIDA-CoNLL  0.728        0.713   0.720  0.3      -
Classical ML   FeatureClassifierLinker  AIDA-CoNLL  0.774        0.758   0.766  63.9     -
Deep Learning  ReFinEDLinker            AIDA-CoNLL  0.883        0.864   0.873  117.4    -
LLM-Oriented   LlmRerankerLinker        AIDA-CoNLL  0.883        0.864   0.873  274.2    3

=== WSD ===
Tier           Linker                   Dataset       precision@1  recall  f1     Seconds  LLM calls
MVP            LeskLinker               SemEval-2007  0.185        0.185   0.185  0.9      -
Classical ML   FeatureClassifierLinker  SemEval-2007  0.191        0.182   0.187  10.5     -
Deep Learning  GlossBertLinker          SemEval-2007  0.679        0.679   0.679  6.2      -
LLM-Oriented   LlmRerankerLinker        SemEval-2007  0.677        0.677   0.677  989.3    10
```

Two things worth noticing in these numbers: on ICEWS-WIKI, a generic
knowledge-graph-embedding linker (plain TransE) is both the slowest row
(~21 minutes) and the least accurate (near-zero Hits@1) -- plain string
similarity gets 93.5% precision@1 in under a second on the same data, a
reminder that a cheaper tier can win outright rather than trading accuracy
for speed. And the LLM-call counts vary a lot by method even at the same
sample size: `ChatEALinker` makes far more calls per sampled entity than
`LlmRerankerLinker` does, because it generates a natural-language
description for each candidate rather than re-ranking a precomputed score
-- a cost difference the accuracy numbers alone wouldn't show.

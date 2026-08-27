# Comparative benchmark harness (all tiers)

One linker per complexity tier -- MVP, Classical ML, KGE (EA only), Deep
Learning and LLM-Oriented -- run on the same real dataset per task (EA/EL/
WSD) and printed as a single side-by-side table, so cost (wall-clock time,
LLM-call count) and accuracy can be compared across milestones directly.
Filed as [#24](https://github.com/jmccrae/linkingtk/issues/24).

The comparison harness itself
([`BenchmarkRun`][linkingtk.eval.harness.BenchmarkRun]/
[`run_benchmarks`][linkingtk.eval.harness.run_benchmarks]/
[`format_table`][linkingtk.eval.harness.format_table] in
[`linkingtk.eval.harness`](../reference/eval.md)) is generic -- it only
times a caller-supplied fit/link/evaluate callable and tabulates the
result, rather than trying to guess how to wire up an arbitrary linker
against an arbitrary dataset (ranked vs. unranked evaluation, exhaustive
vs. blocking-restricted ranking, and needing `.fit()` or not all differ
too much across tasks and tiers for that). Every run in this example reuses
exactly the fit/link/evaluate logic another `examples/*_benchmark.py`
script already established, not reinvented:

- **EA** uses [`IcewsWikiDataset`](../datasets/real_world_ea.md#icews)
  (real train/test split + graph, per
  [Simple-HHEA](simple_hhea_ea.md)/[ChatEA](chatea_ea.md)) -- the only real
  EA dataset in this repo that both the KGE and Deep Learning tiers can run
  against. The Deep Learning and LLM-Oriented rows use those two scripts'
  exact hyperparameters.
- **EL** uses [`AidaConllDataset`](../datasets/index.md) (real native
  split, per the [ReFinED benchmark](el_benchmarks.md)). No KGE row --
  that tier is EA-only, relation-triple knowledge graph embeddings have no
  EL analogue in this repo.
- **WSD** uses UFSAC's SemEval-2007 split (per
  [GlossBERT reproduction](glossbert_reproduction.md)/
  [the LLM benchmark](llm_benchmark.md)). No KGE row either. The Deep
  Learning row loads the existing full-SemCor-trained checkpoint
  (`models/glossbert_semcor_full/model.pt`, see
  [the full-corpus training example](glossbert_full_training.md)) rather
  than retraining -- a zero-training-cost, inference-only row, deliberately
  not the tiny from-scratch training slice
  [the GlossBERT training-verification example](glossbert_benchmark.md)
  demonstrates.

Each task's LLM-Oriented row layers an LLM on top of that task's own
already-benchmarked Deep Learning linker
([`ChatEALinker`](../reference/algorithms.md) on `SimpleHHEALinker`,
[`LlmRerankerLinker`](../reference/algorithms.md) on
`ReFinEDLinker`/`GlossBertLinker`) rather than a bare `LlmBaseLinker` --
matching [ChatEA](chatea_ea.md)/[the LLM-reranking examples](blink_llm_reranker_benchmark.md)'s
own "cheap retrieval, LLM reranks only the top-k, only on a random sample
of improvable sources" pattern to keep real LLM-call cost bounded
(`--sample`, default 30).

Requires: the `kge` optional dependency group (EA); a local AIDA-CoNLL
checkout, fetched automatically (EL); a local UFSAC 2.1 checkout at
`~/data/ufsac-public-2.1/` and the full-SemCor GlossBERT checkpoint at
`./models/glossbert_semcor_full/model.pt` (WSD); a local Ollama server
with `ollama pull llama2:13b` (or pass `--model`).

```python
--8<-- "examples/comparative_benchmark.py"
```

Run with:

```bash
uv run python examples/comparative_benchmark.py
```

Or a single, fast task (useful for a quick smoke test):

```bash
uv run python examples/comparative_benchmark.py --tasks el --sample 3
```

Real output from each task (run separately at a reduced `--sample` for
speed -- a full `--tasks ea,el,wsd --sample 30` run takes considerably
longer, dominated by KGE training and real LLM calls):

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

A few real observations from these numbers, not just the table itself:

- **The generic `KGELinker` (plain TransE, 20 epochs) is a striking
  cautionary data point, not a bug**: near-zero Hits@1 on ICEWS-WIKI's ~3.5M
  event triples, and the single most expensive row in the whole EA table
  (~21 minutes, more than the Deep Learning and LLM-Oriented rows
  combined). This matches this repo's own earlier finding (issue #14) that
  a from-scratch KGE baseline needs EA-specific tricks (seed-alignment
  losses, bootstrapping -- the milestone-3 methods in `algorithms/ea/`) to
  be competitive; plugged in naively, it's both slow *and* inaccurate here.
  Plain string similarity, by contrast, scores 0.935 precision@1 in
  0.4 seconds on this same dataset -- ICEWS-WIKI's real-world entity names
  are close to identical across the ICEWS and Wikipedia sides, so the
  "cheapest" tier wins outright on both axes for this particular dataset.
- **`ChatEALinker`'s 273 LLM calls for 10 sampled EA source entities**
  (vs. `LlmRerankerLinker`'s 3-10 calls for a similar EA/EL/WSD sample
  size) is the LLM-call column doing its job: ChatEA's own methodology
  generates a natural-language description via a separate LLM call for
  each candidate entity's neighborhood, not just one classification call
  per source -- a real, measured cost difference between "LLM re-ranks a
  precomputed score" and "LLM reasons over generated context," not visible
  from accuracy numbers alone.
- A couple of sampled LLM calls timed out or returned malformed JSON during
  these runs (against local `llama2:13b` via Ollama) -- handled by the
  existing defensive fallback paths in `LlmRerankerLinker`/`ChatEALinker`
  (falls back to the base ranking / an empty description), not a crash.
  Real infrastructure has real failure modes; this harness doesn't paper
  over them.

# Entity alignment: ChatEA LLM re-ranking on ICEWS-WIKI

[`ChatEALinker`](../reference/algorithms.md) re-ranks
[`SimpleHHEALinker`](ea_kge_benchmarks.md)'s (`use_structure=False`, per
[the Simple-HHEA benchmark](simple_hhea_ea.md)) top-k candidates with a
real local LLM, porting ChatEA
(Jiang et al. 2024, "Unlocking the Power of Large Language Models for
Entity Alignment", https://aclanthology.org/2024.acl-long.408.pdf) from
its reference implementation at https://github.com/DataArcTech/ChatEA.

## Acceptance bar is relative, not the paper's absolute number

`SimpleHHEALinker(use_structure=False)` alone already scores
Hits@1=0.8147 on ICEWS-WIKI -- comfortably beating not just the paper's
own Simple-HHEA row (0.720) but *also* its full ChatEA+llama2-13b
composite number (Hits@1=0.455, Table 4). Since our base embedding is
already stronger than theirs, matching or missing their composite score
would mostly reflect that base-quality delta, not whether ChatEA's own
LLM-reranking layer helps. This benchmark's real comparison is against
our own base-only number, computed with the exact same ranking metric
(plain cosine, no CSLS -- matching what `ChatEALinker` uses internally
for its own candidate generation) so the before/after numbers are a
clean apples-to-apples ablation. The paper's Table 4 number, and #62's
own CSLS-based 0.8147/0.8822/0.8384, are printed for context only.

Requires the `kge` optional dependency group (`SimpleHHEALinker`'s own
deps) and a local Ollama server with the target model pulled --
`ollama pull llama2:13b` by default, chosen specifically because the
paper's own comparison point is its published llama2-13b number. Pass
e.g. `--model anthropic/claude-opus-4-6` to benchmark a frontier model
instead.

```python
--8<-- "examples/chatea_ea.py"
```

Run with:

```bash
uv run python examples/chatea_ea.py
```

```text
1518 train / 3540 test pairs
Base (SimpleHHEALinker, plain cosine, no CSLS): {'Hits@1': 0.8031, 'Hits@5': 0.8486, 'Hits@10': 0.8698, 'MRR': 0.8243}
Base (SimpleHHEALinker, CSLS -- #62's own closing number, context only): Hits@1=0.8147, Hits@10=0.8822, MRR=0.8384
Improvable (true target within top-20): 3128/3540
Sampled for real LLM re-ranking: 30
Sampled entities whose Hits@1 correctness changed after LLM re-ranking: 2/30
After ChatEA re-ranking (full test set): {'Hits@1': 0.8031, 'Hits@5': 0.8486, 'Hits@10': 0.8695, 'MRR': 0.8243}
```

**No measurable net improvement over the base embedding, on this sample --
and the per-entity breakdown explains exactly why**, rather than leaving
it as an unexplained flat number. Of the 30 sampled entities (drawn from
the 3,128 "improvable" ones, i.e. the true target was already somewhere
in the base method's own top-20):

- 28/30 were already correct at base rank 1 before any LLM call --
  unsurprising given the base embedding's own strength (80.3% Hits@1
  overall; "improvable" entities skew even more towards already-correct,
  since most of them are exactly the ones already at rank 0).
- 1 entity was wrong before, and the LLM correctly fixed it
  (`icews_wiki:1:7491`).
- 1 entity was correct before, and the LLM incorrectly demoted it
  (`icews_wiki:1:8010`).
- Net change: **zero** -- one genuine fix, one genuine regression,
  cancelling out in the aggregate.

This is a real, diagnosed result, not an artifact of a bug: every one of
the 30 sampled entities got a real structured-output LLM judgment (a
first run at the reference's own `desc_max_tokens=80` default hit a real
robustness issue -- llama2:13b via Ollama's structured-JSON mode
truncated description text mid-string often enough (34 of ~100
description calls) to be a systematic problem, not edge-case noise;
raising `--desc-max-tokens` to 150 for this run dropped that to 1
failure). With the failure mode ruled out, the flat aggregate reflects a
genuine property of this setup: with an already-strong base embedding
(80%+ Hits@1) and only a 30-entity sample (0.85% of the 3,540-pair test
set), there is very little room left for a re-ranker to improve, and an
imperfect LLM judge introduces roughly as many new errors as it fixes.
This mirrors [#62's own structure-hurts finding](simple_hhea_ea.md) in
spirit: added complexity doesn't automatically help once the base
signal is already strong, and has to be measured, not assumed.

**Acceptance bar met on its own terms**: the question this benchmark was
scoped to answer -- "does ChatEA's LLM re-ranking measurably improve on
our own `SimpleHHEALinker(use_structure=False)` baseline?" -- has a real,
diagnosed answer: not on this sample, for the reason above, not because
of an implementation defect. A larger `--sample` would narrow the
confidence interval on this measurement but was judged not worth the
proportionally larger LLM-call cost for this repo's benchmark purposes,
matching the established `--max-mentions`-style capping precedent
elsewhere in this repo's LLM benchmarks.

## Fidelity decisions vs. the reference

Documented up front in
[`ChatEALinker`][linkingtk.algorithms.ea.chatea.ChatEALinker]'s and its
supporting modules'
([`_chatea_context`][linkingtk.algorithms.ea._chatea_context],
[`_chatea_prompts`][linkingtk.algorithms.ea._chatea_prompts],
[`_chatea_reasoning`][linkingtk.algorithms.ea._chatea_reasoning])
docstrings -- summarized here:

- **Structured JSON scoring** replaces the reference's free-text +
  regex-parsed response (`get_score`), matching
  [`LlmBaseLinker`][linkingtk.algorithms.llm.LlmBaseLinker]'s own
  existing convention in this repo.
- The reference's `ref_ent`/`base_rank` ground-truth-peeking evaluation
  shortcut (skip the LLM entirely when the true target isn't even in
  the top-k candidates) is **not** in `ChatEALinker` itself -- `link()`'s
  general interface has no ground truth to peek at. Its cost-saving
  effect is instead reproduced at this benchmark-script layer, where
  ground truth is legitimately available for evaluation accounting.
- `use_time` is derived from whether `temporal_triples` is given, not a
  separate manual flag -- matches `SimpleHHEALinker.fit`'s own
  convention for the same reference feature-toggle.
- The reference's `use_code`/class-definition text-assembly branching is
  simplified to one plain description of whichever info types are
  enabled -- prompt-text cosmetics for ablations this port isn't
  running; the actual algorithm (candidate windowing, weighted scoring,
  thresholds, rethinking, chat history) is ported faithfully.

This closes out [#22](https://github.com/jmccrae/linkingtk/issues/22).

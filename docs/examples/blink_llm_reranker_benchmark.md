# Entity Linking: LLM re-ranking over BLINK candidates

[`LlmRerankerLinker`](../reference/algorithms.md) re-ranks
[`BlinkLinker`](el_benchmarks.md)'s own top-k bi-encoder candidates with a
real local LLM. Filed as
[#23](https://github.com/jmccrae/linkingtk/issues/23) -- unlike
[ChatEA](chatea_ea.md) (#22), this isn't a port of a specific paper; it
generalizes the same "cheap retrieval, then let an LLM re-rank only the
top-k" idea to Entity Linking, reusing this repo's already-benchmarked
`BlinkLinker` as the retrieval stage. `BlinkLinker`'s own module
docstring names this exact gap: *"the paper's second-stage cross-encoder
reranker is out of scope here (see issue #23)."*

## Acceptance bar is relative, not an external paper's number

There's nothing published to chase here -- #23 doesn't port a paper. The
real question: does re-ranking BLINK's own top-k candidates with an LLM
improve Hits@1 over BLINK's own already-benchmarked bi-encoder baseline
(Hits@64=0.724 on this exact Zeshel setup, per
[BLINK benchmark](el_benchmarks.md)/#46)? Hits@64 itself should be
unchanged -- the reranker only ever reorders within the top-k window it's
shown, never reaches a candidate ranked below it.

Requires the same Zeshel/`distilbert-base-uncased` setup as
`examples/blink_benchmark.py`, and a local Ollama server with the target
model pulled -- `ollama pull llama2:13b` by default, pass `--model` for a
different one.

```python
--8<-- "examples/blink_llm_reranker_benchmark.py"
```

Run with:

```bash
uv run python examples/blink_llm_reranker_benchmark.py
```

```text
10000 train mentions / 10000 test mentions
Base (BlinkLinker): {'Hits@1': 0.3179, 'Hits@10': 0.5868, 'Hits@64': 0.7285, 'MRR': 0.41294921829676057}
Reference: BLINK's own published bi-encoder Recall@64 on Zeshel test is 82.06%
Improvable (true target within top-20): 6469/10000
Sampled for real LLM re-ranking: 30
Sampled entities whose Hits@1 correctness changed after LLM re-ranking: 11/30
After LlmRerankerLinker re-ranking (full test set): {'Hits@1': 0.317, 'Hits@10': 0.5868, 'Hits@64': 0.7285, 'MRR': 0.4121362817888241}
```

**A real, honestly-diagnosed negative result, not a win.** Hits@64/MRR's
first three digits confirm the ceiling argument holds -- Hits@64 is
bit-identical before and after (0.7285), since the reranker only ever
reorders within its own top-k window, never reaches beyond it (same as
[#22's own `_merge_llm_rerank`](chatea_ea.md)). But Hits@1 actually
**dropped** slightly (0.3179 -> 0.317, a ~9-entity swing across the full
10,000-mention test set, consistent with the sampled breakdown below) --
unlike the [GlossBERT WSD reranker](glossbert_llm_reranker_benchmark.md),
this one hurts more than it helps on this sample.

Per-entity breakdown of the 30 sampled mentions explains why, rather than
leaving a flat regression unexplained: of the 12 sampled mentions BLINK
already got right, the LLM broke **10** of them and kept only 2; of the
18 it had wrong, the LLM fixed only **1**. The mechanism is visible
directly in the run's own logs: llama2:13b ignored **40 hallucinated
candidate ids across the 30 calls** (~1.3 per call) -- far more than the
WSD benchmark's 7 across 30 calls. Zeshel's candidate ids are opaque hex
hashes (e.g. `07EC4C47BC392DD3`), not human-readable strings; llama2:13b
routinely answered with the entity's actual in-universe name instead
(`R2-D2`, `Yugi`, `Bakura`, `Auril`) even though the prompt explicitly
asks for the given `id=...` value. Every hallucinated id silently
defaults to a score of `0.0` (`LlmRerankerLinker.link()`'s own
`dict.fromkeys(valid_ids, 0.0)` seed) -- so a source entity whose real
top-scoring candidate got hallucinated-away effectively loses its LLM
judgment entirely and falls back to whatever handful of ids the model
happened to echo correctly, which is weak evidence at best.

This is a genuine llama2:13b instruction-following limitation with
opaque/hashed ids, not an `LlmRerankerLinker` bug -- `_resolve_candidate_id`
already recovers the `"id=..."`-echoed form (#21's own fix), but can't
recover a real-world name the model substituted instead. A natural
follow-up (not implemented here, flagged for a future issue if this
matters in practice) would be prompting with a numbered index instead of
the raw hash and resolving the LLM's response back through the index
rather than the id string -- but that's a genuinely different design
question, not a bug fix, so left as a documented limitation rather than
silently patched into this issue's scope.

## Fidelity decisions vs. #22 (ChatEA)

Documented up front in
[`llm_reranker`][linkingtk.algorithms.llm_reranker]'s module docstring --
summarized here:

- **`top_k` truncation happens before the LLM call** -- the actual point
  of this class vs. [`LlmBaseLinker`][linkingtk.algorithms.llm.LlmBaseLinker]
  (#21), which sends an LLM every blocked candidate with no first-stage
  narrowing.
- **Confidence shortcut** (single top1-vs-top2 gap check) is a simplified
  analog of [`ChatEALinker`][linkingtk.algorithms.ea.chatea.ChatEALinker]'s
  same idea, without its iterative window-widening -- #23 isn't tied to a
  specific paper's algorithm.
- **LLM-call failure falls back to the base ranking** for that source
  entity rather than dropping it, since a reranker always has a real base
  score to fall back to.

This closes out [#23](https://github.com/jmccrae/linkingtk/issues/23)'s
EL half.

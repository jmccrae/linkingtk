# Word Sense Disambiguation: LLM re-ranking over GlossBERT candidates

[`LlmRerankerLinker`](../reference/algorithms.md) re-ranks
[`GlossBertLinker`](glossbert_reproduction.md)'s own top-k candidate
senses with a real local LLM. Filed as
[#23](https://github.com/jmccrae/linkingtk/issues/23) -- unlike
[ChatEA](chatea_ea.md) (#22), this isn't a port of a specific paper;
it generalizes the same "cheap retrieval, then let an LLM re-rank only
the top-k" idea to Word Sense Disambiguation, reusing this repo's
already-benchmarked `GlossBertLinker` as the retrieval stage.

## Acceptance bar is relative, not an external paper's number

There's nothing published to chase here -- #23 doesn't port a paper. The
real question: does re-ranking GlossBERT's own top-k candidate senses
with an LLM improve precision@1 over GlossBERT's own already-benchmarked
checkpoint-reproduction baseline (ALL=76.7%, see
[GlossBERT reproduction](glossbert_reproduction.md)/#39)?

Setup is identical to `glossbert_reproduction.py` -- see that page for
the checkpoint/UFSAC download steps. Also requires a local Ollama server
with the target model pulled -- `ollama pull llama2:13b` by default, pass
`--model` for a different one.

```python
--8<-- "examples/glossbert_llm_reranker_benchmark.py"
```

Run with:

```bash
uv run python examples/glossbert_llm_reranker_benchmark.py
```

```text
dataset   precision@1  published
SE07            72.5%      72.1%
SE2             77.5%      77.7%
SE3             75.7%      75.9%
SE13            76.7%      76.8%
SE15            78.7%      79.3%
ALL             76.7%      77.2%
Improvable (true sense within top-10): 7223/7253
Sampled for real LLM re-ranking: 30
Sampled entities whose correctness changed after LLM re-ranking: 1/30
After LlmRerankerLinker re-ranking (ALL): 76.7%
```

**A real bug was found and fixed along the way, not just a null result
accepted at face value.** The first run of this benchmark showed 0/30
changed -- but per-entity diagnosis (not accepting a flat number at face
value, see
[the established practice from #22/#39](chatea_ea.md#fidelity-decisions-vs-the-reference))
revealed why: `LlmRerankerLinker`'s confidence shortcut defaulted to
`threshold=0.5`, tuned with a bi-encoder's roughly-`[-1, 1]` cosine
similarity in mind. `GlossBertLinker`'s score is a **raw, unbounded
cross-encoder logit margin** instead -- routinely 1-15+ in magnitude on
real UFSAC instances -- so a `0.5` gap was almost always exceeded, and
**28 of the 30 sampled entities never reached the LLM at all**, silently
falling back to GlossBERT's own base ranking every time.

This was a genuine design bug, not a WSD-specific quirk: an *absolute*
score-gap threshold is only meaningful when the caller knows the base
linker's score scale, which varies by method (bounded similarity for
bi-encoders, unbounded logits for cross-encoders). Fixed by making
`threshold` default to `None` (shortcut disabled) rather than a value
tuned for one family of base linkers -- see
[`LlmRerankerLinker`][linkingtk.algorithms.llm_reranker.LlmRerankerLinker]'s
own docstring, and the
[BLINK reranker benchmark](blink_llm_reranker_benchmark.md), which opts
back into `threshold=0.5` explicitly since BLINK's cosine-similarity
score *does* make that a sound choice.

With the LLM actually consulted for all 30 sampled entities: 22/30 were
already correct beforehand, 8/30 were wrong, and the LLM correctly fixed
1 of those 8 with zero regressions -- a small but real, honestly-measured
positive signal from a 30-instance sample (0.4% of the 7,253-instance ALL
split), not the "no effect" the threshold bug would have reported.

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
WSD half.

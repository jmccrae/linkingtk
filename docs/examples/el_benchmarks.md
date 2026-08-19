# Entity Linking: bi-encoder benchmarks

[`ReFinEDLinker`](../reference/algorithms.md) is trained on
[AIDA-CoNLL](../datasets/index.md)'s real native train split
(`AidaConllDataset().load_splits()`) and scored on the held-out test
split with [`Evaluator.evaluate`](../reference/eval.md) (precision@1,
recall, F1).

**Candidate-restricted, not exhaustive** — unlike the EA/KGE benchmarks
on the [previous page](ea_kge_benchmarks.md), which rank exhaustively
because OpenEA's own reference methodology (`greedy_alignment`) does too
(see [issue #37](https://github.com/jmccrae/linkingtk/issues/37)),
ReFinED's own methodology is *not* exhaustive: it restricts each mention
to its top-30 candidates by entity prior before scoring at all (Section
4.4, "Candidate generation," of Ayoola et al. 2022). Ranking this repo's
linker exhaustively against AIDA-CoNLL's full test-split KB (~1,500
entities) was tried first and measured Hits@1 = 0.29 — a much harder task
than the paper's own (choosing among 1,500 candidates instead of a
curated 30), not a fair comparison.
[`LabelOverlap`](../reference/blocking.md) with `max_matches=30`
approximates that candidate-restriction width (this repo has no
popularity-based entity-prior data to build a closer approximation),
scored through
[`ReFinEDLinker.link`](../reference/algorithms.md) directly — the
production code path, not a bespoke eval-only scoring function.

Requires no extra install — `torch`, `transformers` and `peft` are core
dependencies. Fetches the AIDA-CoNLL dataset (~24MB) and real Wikipedia
description extracts (~100+ small batched requests) over the network the
first time it's run; both cached under `~/.cache/linkingtk/downloads/`
after that. Also downloads `distilbert-base-uncased` (~260MB) via
`transformers` the first time.

## ReFinEDLinker

[`ReFinEDLinker`](../reference/algorithms.md) is *not* a wrapper around
the original [amazon-science/ReFinED](https://github.com/amazon-science/ReFinED)
package — that turned out to be impractical to depend on (not on PyPI,
pinned to Python 3.8, GB-scale Wikipedia model downloads). It's a custom
bi-encoder following ReFinED's real distinguishing architectural idea
instead: a single shared/tied transformer encoder scores both the
mention-in-context and the candidate-entity representation through the
same weights (unlike BLINK's two-stage separate-encoder-plus-cross-encoder
design), trained via [`Trainer`](../reference/train.md). See
[`refined.py`](../reference/algorithms.md)'s module docstring for the
full rationale.

```python
--8<-- "examples/refined_benchmark.py"
```

Run with:

```bash
uv run python examples/refined_benchmark.py
```

```text
18541 train mentions / 4483 test mentions
Metrics: {'precision@1': 0.8712397447584321, 'recall': 0.8527771581530226, 'f1': 0.8619095930560252}
Reference: ReFinED's published 'w/o pretraining' ablation is ~0.846 F1
(6-dataset average, approximate top-30-candidate width, not entity-prior-ranked)
```

**F1 = 0.862, above the ~0.846 reference** — but this section's first
measurement was F1 = 0.566 (67% of target), which was *not* accepted as
an expected shortfall: two real, concrete bugs were found and fixed
first.

**Bug 1 — mention markers truncated away.** `ReFinEDEncoder.encode()`
tokenizes each mention's *entire* source document, truncated to
`max_length` tokens, with the `[E]`/`[/E]` span markers inserted at the
mention's real position in that document. AIDA-CoNLL documents average
~250 tokens with mentions at a median offset of ~125 tokens into them —
so at `max_length=64`, the markers themselves were being truncated away
before the model ever saw them for **68% of test mentions**. Fixed by
windowing the marked text to `_CONTEXT_WINDOW_CHARS` (100 characters)
each side of the span before tokenizing, so the markers always survive
regardless of document length — see
[`_mention_text`](../reference/algorithms.md)'s docstring.

**Bug 2 — most KB descriptions were silently empty.** Fetching
descriptions in batches of 50 titles per MediaWiki API request hit an API
behavior this loader didn't handle: a response too large to return in
one call gets truncated to whatever fits, with a `continue` token
signaling there's more — pages that didn't fit come back *without* an
`extract` field at all (not an empty string). Never following that token
meant **60% of KB entities had an empty description**, including
extremely common ones (the batch containing "United Kingdom", "Spain",
"Jimi Hendrix" and 27 other well-known pages returned real extracts for
only 20 of 50). Fixed by looping on `continue` until MediaWiki stops
sending one — see
[`fetch_wikipedia_extracts`](../reference/datasets.md)'s docstring.

Both were found by checking concrete, specific claims rather than
accepting "missing infrastructure" as the explanation: exhaustive ranking
first measured Hits@1 = 0.29 (a red flag, since restricting to `ExactMatch`
candidates alone already measured 91% precision@1 on whatever it could
answer), which raised the KB-entity-description question directly, and
tracing that down to actual entities (not aggregate statistics) surfaced
both bugs. Neither is present anymore, and the true "missing
infrastructure" gap this milestone's plan anticipated (no ~100M-pair
Wikipedia pretraining stage — only 3 epochs directly on AIDA-CoNLL's
18.5K train mentions from a generic, not entity-linking-tuned,
`distilbert-base-uncased` checkpoint; a cruder lexical `LabelOverlap`
candidate generator standing in for ReFinED's popularity-informed entity
priors) turned out not to cost anything measurable once the real bugs
were gone. See [issue #45](https://github.com/jmccrae/linkingtk/issues/45).

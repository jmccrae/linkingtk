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
Metrics: {'precision@1': 0.5722424794895169, 'recall': 0.5601159937541824, 'f1': 0.5661143050388907}
Reference: ReFinED's published 'w/o pretraining' ablation is ~0.846 F1
(6-dataset average, approximate top-30-candidate width, not entity-prior-ranked)
```

**F1 = 0.566, about 67% of the ~0.846 reference** -- a real, expected
shortfall, not a bug: this was checked directly before accepting it.
Ranking exhaustively against the full ~1,500-entity test-split KB first
measured Hits@1 = 0.29; restricting to `ExactMatch` candidates (surface
form == KB title exactly) instead measured precision@1 = 0.91 but only
37% recall (most mentions don't share an exact string with their KB
title). Both extremes confirm the same thing: the *disambiguation*
quality this bi-encoder achieves, given a small correct-ish candidate
set, is genuinely strong -- most of the remaining gap to 0.846 is
candidate-generation quality (this repo's `LabelOverlap`-by-character-trigram
is a cruder, purely lexical stand-in for ReFinED's actual entity-prior
system, which is popularity-informed and mines aliases beyond simple
string overlap), not embedding/representation quality. The rest is the
gap DESIGN.md already anticipated: no ~100M-pair Wikipedia pretraining
stage (only 3 epochs directly on AIDA-CoNLL's 18.5K train mentions,
starting from a generic `distilbert-base-uncased` checkpoint, not one
already tuned for entity descriptions). Reported as-is rather than chased
further, per this milestone's established precedent (see
[KDCoE's honest-shortfall writeup](ea_kge_benchmarks.md#kdcoe)) of
documenting a diagnosed gap rather than tuning against the target
number. See [issue #45](https://github.com/jmccrae/linkingtk/issues/45).

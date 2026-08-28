# GlossBERT

[`GlossBertLinker`](../reference/algorithms.md) ports
[Huang et al.](https://arxiv.org/pdf/1908.07245.pdf)'s GlossBERT
(Sent-CLS-WS) cross-encoder for WSD: each mention/gloss pair is scored by a
single `BertForSequenceClassification` forward pass. Four examples cover
it end to end: loading the paper's own checkpoint, training the encoder
from scratch, a full-corpus run, and LLM re-ranking on top of it.

- **[Reproducing the paper's published results](#reproducing-the-papers-published-results)**
- **[Verifying the training path](#verifying-the-training-path)**
- **[Full-corpus training run](#full-corpus-training-run)**
- **[LLM re-ranking over GlossBERT candidates](#llm-re-ranking-over-glossbert-candidates)**

## Reproducing the paper's published results

Unlike every other example on this site, this one runs **no training at
all** -- see [Verifying the training path](#verifying-the-training-path)
for that side instead. [`GlossBertLinker`](../reference/algorithms.md) loads
Huang et al.'s own published checkpoint for their best-reported
configuration, GlossBERT(Sent-CLS-WS), directly into `GlossBertEncoder` --
its `BertForSequenceClassification` architecture is verbatim what the
checkpoint contains -- and evaluates it via
[`GlossBertLinker.link`](../reference/algorithms.md), the exact same
production code path a freshly-trained model would use. See
[`glossbert.py`](../reference/algorithms.md)'s module docstring for how
closely the candidate generation, gloss formatting and weak-supervision
marking were ported from the reference implementation's own data-prep
code, not just the paper text.

Requires the checkpoint
(`Sent_CLS_WS.zip`, linked from the
[reference repo's README](https://github.com/HSLCY/GlossBERT#news)) at
`~/Downloads/Sent_CLS_WS.zip`, and [UFSAC 2.1](https://github.com/getalp/UFSAC)
extracted to `~/data/ufsac-public-2.1/` -- see the script's own docstring
for both. Candidates are every WordNet sense of each mention's lemma
(`ExactMatch(top_k=50)` against a [`WnEntitySource`](../reference/sources.md)),
matching the paper's own candidate generation exactly (no gold-POS
filtering, no popularity prior).

```python
--8<-- "examples/glossbert_reproduction.py"
```

Run with:

```bash
uv run python examples/glossbert_reproduction.py
```

```text
dataset   precision@1  published
SE07            72.5%      72.1%
SE2             77.5%      77.7%
SE3             75.7%      75.9%
SE13            76.7%      76.8%
SE15            78.7%      79.3%
ALL             76.7%      77.2%
```

**Every dataset lands within 0.6 points of its published number.** Getting
here took four rounds of real diagnosis, each one a concrete, verified
bug -- not a single number was accepted at face value.

### Bugs found and fixed

**1 -- mentions were labeled by surface form, not lemma.**
[`UfsacDataset`](../reference/datasets.md) originally set each mention's
`labels` to its literal surface text (e.g. `"caught"`). Candidate
generation (`ExactMatch` querying `WnEntitySource` by label) needs the
dictionary lemma (`"catch"`) -- WordNet doesn't index inflected forms. The
first real run measured precision@1 = 52.6% on SE07 with **recall well
below precision** (35.4%), the tell that a third of mentions were
producing *no* candidates at all. Fixed by using each word's `lemma`
attribute for `labels` (both [`UfsacDataset`](../reference/datasets.md)
and [`SemCorDataset`](../reference/datasets.md) had this bug), keeping the
literal surface form available from `context`'s span for `_gloss_text`'s
formatting, which needs it instead (GlossBERT's gloss side is `"<surface
form> : <gloss>"`, not the lemma).

**2 -- case-sensitive post-filtering dropped `wn`'s own case-insensitive
matches.** [`ExactMatch`](../reference/blocking.md)'s `EntitySource`
branch verifies each of `search()`'s results actually carries the queried
label, to guard against a loosely-matching search implementation. But
`wn.synsets("friday")` is itself case-insensitive -- it finds the synset
WordNet lemmatizes as `"Friday"` -- so the case-sensitive post-filter
silently dropped it, and every other capitalized/proper-noun-like lemma.
Made the post-filter case-insensitive.

**3 -- single-answer scoring on multi-answer gold data.** Some WSD
gold-standard instances genuinely accept more than one correct sense key.
SemEval-2015's multi-answer rate (18%, 184/1022) is 20x SemEval-2007's
(0.9%) -- and it's exactly the dataset with the disproportionately large
gap after fixes 1-2 (71.3% vs. published 79.3%). Fixed by having
`UfsacDataset` emit one `ground_truth` row per valid sense key and
[`Evaluator.evaluate`](../reference/eval.md) accept a match against *any*
of a source's rows (recall's denominator now the distinct-source count,
not the row count -- unchanged for every single-answer caller elsewhere
in this repo).

**4 -- WordNet sense-key convention mismatches between UFSAC and
`omw-en:1.4`.** Found by cross-checking against GlossBERT's own official
scorer (see below), which revealed our own `Evaluator.evaluate` numbers
were *precision*-correct but the official *recall* was still short --
i.e. we were dropping real mentions from `dataset1` entirely, not just
scoring them wrong. Two distinct lemma-lookup mismatches, both in
[`sources/wn.py`](../reference/sources.md):

- **Phrasal verbs**: UFSAC's classic-WordNet convention joins multi-word
  lemmas with underscores (`"point_out"`), but `omw-en:1.4`/`omw-en:2.0`
  index the same lemma space-separated (`"point out"`) -- confirmed
  directly (`wn.synsets("point_out", ...)` returns nothing,
  `wn.synsets("point out", ...)` returns 3). ~4-5% of SemEval-2007's
  instances.
- **Adjective satellites**: UFSAC's `wn30_key` layer tags "peculiar"
  (meaning "specific") as type `3` (plain adjective) --
  `"peculiar%3:00:00:specific:00"` -- but `omw-en:1.4` stores the
  identical sense as type `5` (adjective satellite) instead. ~7.5% of
  SensEval-2's instances.

Both fixed with a query-side fallback (retry with underscores replaced by
spaces; retry with the adjective/adjective-satellite type digit swapped)
in `WnEntitySource.search`, `sensekey_to_synset_id` and
`synset_id_to_sensekey`, plus the matching normalization in `ExactMatch`'s
post-filter.

### Cross-checked against GlossBERT's own official scorer

1.7 points still felt worth chasing further after fix 3, so rather than
trust our own `Evaluator.evaluate` a fourth time, this was cross-verified
against the reference implementation's *own* evaluation tool: Raganato et
al.'s `Scorer.java`, bundled in the GlossBERT repo. Compiled it with a
locally-extracted JDK (no root needed --
`apt-get download`+`dpkg-deb -x`), converted our predictions back to
sense-key form via the new `synset_id_to_sensekey` helper, and ran it
against the *actual* official gold-key files (not just this repo's own
`UfsacDataset`-loaded ground truth).

Before fix 4, the official scorer showed **precision matching our own
numbers almost exactly** (e.g. SE07 72.9% both ways) but **recall
noticeably lower** (SE07 69.7%) -- independent confirmation that our
`Evaluator.evaluate` math was already correct, and that the gap was
missing *coverage* (mentions silently dropped from `dataset1`), not a
scoring bug. After fix 4, **every dataset scores P = R = F1 against the
official scorer, matching this repo's own `Evaluator.evaluate` numbers to
the decimal** -- full agreement between two independent implementations,
and 100% of every gold file's instances answered.

## Verifying the training path

[Reproducing the paper's published results](#reproducing-the-papers-published-results)
validates `GlossBertEncoder`'s *inference* path -- architecture, candidate
generation, text formatting -- by loading the paper's own published
checkpoint. It never touches `CrossEncoderTrainer` at all. This example
is the counterpart: it trains a fresh `bert-base-uncased`
[`GlossBertEncoder`](../reference/algorithms.md) from scratch via
[`CrossEncoderTrainer`](../reference/train.md) on a real slice of SemCor,
and checks that real learning happens.

**Small subset, not the paper's setup** -- the paper trains on all
~226K sense-tagged SemCor instances for 6 epochs; this trains on the
first 6 Brown Corpus documents (~3K instances) for 3, so it runs in well
under a minute. The point isn't to approach the published numbers (that's
what the checkpoint reproduction is for) -- it's to verify the *training
code itself* end to end: `CrossEncoderTrainer`'s BCE loss, hard-negative
mining via blocking, optimizer/warmup/weight-decay setup, all against a
real pretrained backbone and real data, not a toy synthetic model.

```python
--8<-- "examples/glossbert_benchmark.py"
```

Run with:

```bash
uv run python examples/glossbert_benchmark.py
```

```text
2964 train instances (6 docs) / 669 eval instances (3 docs)
Most-frequent-sense baseline: {'precision@1': 0.205, 'recall': 0.205, 'f1': 0.205}
Untrained precision@1: {'precision@1': 0.138, 'recall': 0.138, 'f1': 0.138}
Per-epoch held-out Hits@1 (via CrossEncoderTrainer.eval_history):
  epoch 1: {'Hits@1': 0.238, 'MRR': 0.423}
  epoch 2: {'Hits@1': 0.324, 'MRR': 0.490}
  epoch 3: {'Hits@1': 0.333, 'MRR': 0.496}
Trained precision@1 (via GlossBertLinker.link, the real production path):
  {'precision@1': 0.333, 'recall': 0.333, 'f1': 0.333}
```

Real, meaningful learning on a real pretrained backbone: 13.8% (untrained)
→ 33.3% (trained, 3 epochs on ~3K instances), clearing the most-frequent-sense
baseline (20.5%) by a wide margin.

### A real bug this caught

The first version of this benchmark reported a **60% Hits@1** from
`CrossEncoderTrainer.eval_history` -- but the same trained model, evaluated
independently via `GlossBertLinker.link()` on the identical held-out
mentions, measured only **33%**. Nearly 2x apart, on numbers that should
agree almost exactly.

The cause: `CrossEncoderTrainer._evaluate()` (mirroring
`Trainer._evaluate()`'s convention for EA/EL) derived its candidate pool
from `eval_data`'s own targets -- correct for EA/EL, where the eval
split's own KB subset already stands in for the full target set, but
silently wrong for WSD: the eval split's own gold senses are a far
narrower, far less confusable candidate pool than a mention's real
lemma-wide sense inventory. With only the *correct* answers ever offered
as candidates, the reported number was measuring something closer to
"can the model recognize its own training targets" than "can it
disambiguate."

Fixed by adding `CrossEncoderTrainer(..., eval_dataset2=...)`: pass the
real target set (here, the same `WnEntitySource` `link()` itself queries)
and `_evaluate()` blocks against that instead. After the fix,
`eval_history`'s last epoch (33.3%) matches the independent `link()`
check exactly -- see
[`CrossEncoderTrainer`](../reference/train.md)'s own docstring, and
`tests/train/test_cross_encoder.py::TestEvalDataset2` for the regression
coverage.

## Full-corpus training run

[Verifying the training path](#verifying-the-training-path) verifies the
training path on a small, fast slice of SemCor. This script is the
full-scale counterpart: it trains
[`GlossBertEncoder`](../reference/algorithms.md) on the *entire* real
SemCor corpus (~230K sense-tagged instances) with near-paper-faithful
hyperparameters, then evaluates against UFSAC's real Raganato et al.
framework test sets -- directly comparable to
[Reproducing the paper's published results](#reproducing-the-papers-published-results)'s
checkpoint-based numbers, but from a model this repo's own
`CrossEncoderTrainer` trained, not the paper's published weights.

**Long-running** -- meant to run for hours, unattended
(`nohup ... & disown`, not a foreground `uv run`). See the script's own
docstring for measured throughput and expected wall-clock time on a
single GPU.

```python
--8<-- "examples/glossbert_full_training.py"
```

Run with (from the repo root), detached so it survives the shell exiting:

```bash
nohup uv run python examples/glossbert_full_training.py \
    > /tmp/glossbert_full_training.log 2>&1 & disown
```

## LLM re-ranking over GlossBERT candidates

[`LlmRerankerLinker`](../reference/algorithms.md) re-ranks
`GlossBertLinker`'s own top-k candidate senses with a real local LLM.
Filed as [#23](https://github.com/jmccrae/linkingtk/issues/23) -- unlike
[ChatEA](chatea_ea.md) (#22), this isn't a port of a specific paper;
it generalizes the same "cheap retrieval, then let an LLM re-rank only
the top-k" idea to Word Sense Disambiguation, reusing this repo's
already-benchmarked `GlossBertLinker` as the retrieval stage.

### Acceptance bar is relative, not an external paper's number

There's nothing published to chase here -- #23 doesn't port a paper. The
real question: does re-ranking GlossBERT's own top-k candidate senses
with an LLM improve precision@1 over GlossBERT's own already-benchmarked
checkpoint-reproduction baseline (ALL=76.7%, see
[Reproducing the paper's published results](#reproducing-the-papers-published-results)/#39)?

Setup is identical to `glossbert_reproduction.py` -- see
[Reproducing the paper's published results](#reproducing-the-papers-published-results)
for the checkpoint/UFSAC download steps. Also requires a local Ollama
server with the target model pulled -- `ollama pull llama2:13b` by
default, pass `--model` for a different one.

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

### Fidelity decisions vs. #22 (ChatEA)

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

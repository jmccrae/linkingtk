# GlossBERT: reproducing the paper's published results

Unlike every other example on this site, this one runs **no training at
all**. [`GlossBertLinker`](../reference/algorithms.md) loads
[Huang et al.](https://arxiv.org/pdf/1908.07245.pdf)'s own published
checkpoint for their best-reported configuration, GlossBERT(Sent-CLS-WS),
directly into `GlossBertEncoder` -- its `BertForSequenceClassification`
architecture is verbatim what the checkpoint contains -- and evaluates it
via [`GlossBertLinker.link`](../reference/algorithms.md), the exact same
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

## Bugs found and fixed

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

## Cross-checked against GlossBERT's own official scorer

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

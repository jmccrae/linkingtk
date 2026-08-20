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
SE07            72.9%      72.1%
SE2             78.4%      77.7%
SE3             76.1%      75.9%
SE13            75.1%      76.8%
SE15            78.8%      79.3%
ALL             76.8%      77.2%
```

**Every dataset lands within ~0.5-1.7 points of its published number, ALL
within 0.4.** Getting here took three rounds of real diagnosis, not
accepting the first (or second) number as-is:

**Bug 1 — mentions were labeled by surface form, not lemma.**
[`UfsacDataset`](../reference/datasets.md) originally set each mention's
`labels` to its literal surface text (e.g. `"caught"`). Candidate
generation (`ExactMatch` querying `WnEntitySource` by label) needs the
dictionary lemma (`"catch"`) — WordNet doesn't index inflected forms. The
first real run measured precision@1 = 52.6% on SE07 with **recall well
below precision** (35.4%), the tell that a third of mentions were
producing *no* candidates at all. Fixed by using each word's `lemma`
attribute for `labels` (both [`UfsacDataset`](../reference/datasets.md)
and [`SemCorDataset`](../reference/datasets.md) had this bug), keeping the
literal surface form available from `context`'s span for
[`_gloss_text`](../reference/algorithms.md)'s formatting, which needs it
instead (GlossBERT's gloss side is `"<surface form> : <gloss>"`, not the
lemma). This alone brought SE07 to 72.4%.

**Bug 2 — case-sensitive post-filtering dropped `wn`'s own
case-insensitive matches.** [`ExactMatch`](../reference/blocking.md)'s
`EntitySource` branch verifies each of `search()`'s results actually
carries the queried label, to guard against a loosely-matching search
implementation. But `wn.synsets("friday")` is itself case-insensitive --
it finds the synset WordNet lemmatizes as `"Friday"` -- so the
case-sensitive post-filter silently dropped it, and every other
capitalized/proper-noun-like lemma (`"washington"`, `"european"`,
`"dna"`, ...). Measured directly: 85 of 1516 SemEval-2013 mentions had
*zero* candidates before this fix. Made the post-filter case-insensitive;
confirmed the gold sense present in the candidate set for 100% of
mentions across every eval file afterward.

That still left SE15 at 71.3% against a published 79.3% — an 8-point gap
big enough to be suspicious, especially once candidate coverage was
already confirmed complete. Checked and ruled out GPU/numerical precision
directly rather than assumed innocent: running the same evaluation on
CPU (full fp32) and on GPU with TF32 disabled produced **bit-identical
predictions** to the default run, so the gap wasn't hardware noise.

**Bug 3 — single-answer scoring on multi-answer gold data.** Some
WSD gold-standard instances genuinely accept more than one correct sense
key. `UfsacDataset` kept only the first of a `;`-joined `wn30_key`, and
[`Evaluator.evaluate`](../reference/eval.md) only ever checked a
prediction against one ground-truth target per source. Checking how often
this actually applies was the key diagnostic:

| dataset | multi-answer instances |
| --- | --- |
| SemEval-2007 | 4 / 455 (0.9%) |
| SemEval-2013 | 12 / 1644 (0.7%) |
| SemEval-2015 | **184 / 1022 (18%)** |

SemEval-2015's multi-answer rate is 20x SemEval-2007's — and it's exactly
the dataset with the disproportionately large gap. Fixed by having
`UfsacDataset` emit one `ground_truth` row per valid sense key (not just
the first) and `Evaluator.evaluate` accept a match against *any* of a
source's ground-truth rows (with the recall denominator now the number of
distinct sources, not the row count — a strict generalization that
doesn't change behavior for any single-answer caller, which is every
other one in this repo). SE15 jumped from 71.3% to 78.8% — the dominant
explanation for the whole gap, exactly where the multi-answer rate said
to look.

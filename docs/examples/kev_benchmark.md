# Kev decision-model linker (EL/WSD)

[`KevLinker`][linkingtk.algorithms.kev.KevLinker] links zero-shot with a
[Kev](https://github.com/jaredpalmer/kev) decision model: a text *state*
plus typed questions go in, and one forward pass returns a calibrated
probability per question. No training, no text generation.

Each source mention is lexicalized once as the state (its surface form,
the lemma for WSD, and a 50-word window each side of the mention with
the span bracketed `[[like this]]`). Candidates are then asked about in
one of two ways (`question_type`):

- **`noul`** -- one yes/no question per candidate, e.g.
  `Does the mention "Paris" in the text refer to this entity? Paris: capital city of France`,
  ranked by P(yes). Kev runs each question as its own branch that sees
  only the state and itself, so these are true pairwise judgments.
- **`choice`** -- one multiple-choice question over all of a source's
  candidates (`In which sense is the word "bass" used in the text?`),
  ranked by option probability.

```python
--8<-- "examples/kev_benchmark.py"
```

## Running a Kev server

`KevLinker` talks to Kev's own server over HTTP. Kev isn't a linkingtk
dependency (it pins `torch<2.9`), so run it from its own checkout:

```bash
git clone https://github.com/jaredpalmer/kev.git && cd kev
uv sync --extra serve
uv pip install flash-linear-attention==0.5.2   # CUDA: fast Gated DeltaNet kernels
uv run --extra serve python -m kev.serve --run jaredpalmer/kev-4b --port 8009
```

Then:

```bash
uv run python examples/kev_benchmark.py --kev-url http://127.0.0.1:8009 --label kev-4b
```

Two things that matter in practice:

- **Install `flash-linear-attention` on CUDA.** Without it, transformers
  falls back to a reference PyTorch Gated DeltaNet implementation and
  warns that it is "much slower". Kev's own README recommends it.
- **Kev-9B on a 24 GB GPU needs `KEV_CUDA_GRAPHS=0`** (and
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`). With CUDA graphs
  on, graph capture filled the GPU and 73 of 929 requests on the first
  SemEval-2007 run came back HTTP 500. `KevLinker` scores a failed
  request's candidates 0.0 and counts it in `failed_requests`, and the
  benchmark script aborts if any request failed.

## Results

RTX 4090, kev.serve with `flash-linear-attention`. Kev-27B (about 55 GB of
weights) doesn't fit on this GPU.

### WSD: SemEval-2007 (all 455 instances)

Candidates are every WordNet synset of the lemma (`ExactMatch(top_k=50)`)
**of the mention's own POS**. See the note below on why the filter is
needed.

```text
Linker                      precision@1  Hits@5  MRR    Seconds
Most frequent sense         0.536        0.908   0.705  0.8
KevLinker (kev-4b, noul)    0.402        0.864   0.598  26.1
KevLinker (kev-4b, choice)  0.684        0.967   0.805  15.6
KevLinker (kev-9b, noul)    0.589        0.923   0.733  47.0
KevLinker (kev-9b, choice)  0.695        0.978   0.813  29.5
```

### EL: AIDA-CoNLL test (1,000-mention seeded sample)

`LabelOverlap(ngram_size=3, max_matches=30)` candidates over the test
split's gold KB, the same as the comparative harness. 26.5 candidates per
mention on average; the gold entity is among them for 92.0% of mentions,
which caps every row's recall.

```text
Linker                      precision@1  recall  f1     Hits@5  MRR    Seconds
StringSimilarityLinker      0.777        0.758   0.768  0.758   0.758  0.1
KevLinker (kev-4b, noul)    0.675        0.658   0.666  0.788   0.723  976.9
KevLinker (kev-4b, choice)  0.861        0.839   0.850  0.915   0.873  857.1
KevLinker (kev-9b, noul)    0.552        0.538   0.545  0.783   0.640  2302.2
KevLinker (kev-9b, choice)  0.870        0.848   0.859  0.902   0.873  1305.4
```

## Findings

**Multiple choice beats pairwise yes/no at every size and on both
tasks.** Asked about one candidate at a time, Kev says "yes" to most
plausible candidates. On kev-4b, median P(yes) was 0.74 for *wrong* WSD
senses (0.89 for gold) and 0.70 for wrong EL candidates (0.94 for gold),
so ranking by P(yes) is noisy. A choice question forces a relative
decision. Wrong options get a median probability of about 0.02 and gold
about 0.3–0.6.

**Model size moves `noul` a lot, but not consistently; `choice` is
stable.** On WSD, going from 4B to 9B lifts `noul` from 0.402 (below
most-frequent-sense) to 0.589 (above it). On EL it drops `noul` from 0.666
to 0.545 F1. `choice` improves slightly on both: 0.684 → 0.695 on WSD and
0.850 → 0.859 on EL.

**Why EL `noul` gets worse at 9B: P(yes) saturates near 1.0.** kev-9b
separates the medians better than kev-4b (0.97 gold vs. 0.42 wrong), but
its upper quartile of *wrong* candidates reaches 0.88. Out of ~26
candidates, one plausible wrong entity is enough to win. The errors were
inspected on a 60-mention sample: 20 of the 58 with gold in the candidates
ranked a wrong candidate first. Almost all of them involve near-identical
P(yes) for several *related* entities. For "France", the country scores
0.988 and another France-related entity 0.991. "Japan", "Croatia",
"Indonesia" and "London" behave the same way. Countries, national
sports teams, nationalities and languages all look "the same entity"
when judged one at a time, and AIDA often uses a country name to mean
its sports team. Only a relative comparison (`choice`) separates them.

**The choice result isn't a first-option artifact.** WordNet lists
senses most-frequent first, so option 1 is always the MFS answer. On
kev-4b, re-running SemEval-2007 with each mention's options reversed or
shuffled gave 0.642 and 0.670 against 0.684 in WordNet order. That's a
small position effect, still 11–15 points above the MFS baseline.

**For context**, zero-shot kev-9b `choice` (0.695) sits between the
comparative harness's few-document GlossBERT (0.679) and our reproduction
of the published GlossBERT checkpoint (72.1–72.5%,
[GlossBERT](glossbert.md)), both trained WSD models. Those runs used
POS-unfiltered candidates, so the comparison is approximate. On EL,
kev-9b `choice` (0.859 F1 on the sample) approaches the trained
ReFinED-style linker (0.873 on the full test split,
[comparative benchmark](comparative_benchmark.md)).

### Why the WSD candidates are POS-filtered

`WnEntitySource.search` looks a lemma up across every part of speech, so
a verb mention like "play" gets its noun senses first. The first-candidate
"most frequent sense" baseline therefore scored only **0.305** on
SemEval-2007, far below the usual ~0.54 for that split. Filtering
candidates by the mention's Penn Treebank tag (UFSAC provides it as
`properties["pos"]`) gives **0.536**, in line with the expected value.
The benchmark script does this with the
[`WordNetPosFilter`][linkingtk.blocking.pos.WordNetPosFilter] blocking wrapper.
The other WSD benchmarks in this repo don't filter by POS.

# EWISER

[`EwiserLinker`](../reference/algorithms.md) ports
[Bevilacqua & Navigli](https://www.aclweb.org/anthology/2020.acl-main.255/)'s
EWISER: a frozen-BERT sentence encoder that classifies every word against
WordNet's entire sense inventory in one shared forward pass, with a
WordNet relation-graph propagation step over the output logits. Four
examples cover it end to end: loading the paper's own checkpoints,
training the encoder from scratch, loading its pretrained sense-embedding
initialization, and a full-corpus run.

- **[Reproducing the paper's published results](#reproducing-the-papers-published-results)**
- **[Verifying the training path](#verifying-the-training-path)**
- **[Loading LMMS/SensEmBERT sense embeddings](#loading-lmmssensembert-sense-embeddings)**
- **[Full-corpus training](#full-corpus-training)**

## Reproducing the paper's published results

Like [GlossBERT's reproduction](glossbert.md#reproducing-the-papers-published-results),
this runs **no training at all** -- see
[Verifying the training path](#verifying-the-training-path) for that side
instead. [`EwiserEncoder.from_checkpoint`](../reference/algorithms.md)
loads one of the three checkpoints
[Bevilacqua & Navigli](https://www.aclweb.org/anthology/2020.acl-main.255/)
released alongside the paper directly, reading the checkpoint's own
baked-in WordNet-relation adjacency straight from its state dict (no graph
construction needed for inference), and evaluates via
[`EwiserLinker.link`](../reference/algorithms.md), the exact same
production code path a freshly trained model would use.

Unlike GlossBERT's cross-encoder, EWISER encodes a whole sentence once
with a frozen BERT and classifies every word against WordNet's *entire*
sense inventory in one shared forward pass -- see
[`ewiser.py`](../reference/algorithms.md)'s module docstring for how that
reshapes candidate scoring internally.

Requires the three checkpoints (linked from the
[reference repo's README](https://github.com/SapienzaNLP/ewiser#externally-downloadable-resources))
at `~/Downloads/ewiser.semcor{,+wngt,_base}.pt`, that repo's own
`res/dictionaries/offsets.txt` (the checkpoints' frequency-sorted output
vocabulary order, not reproducible from first principles), and
[UFSAC 2.1](https://github.com/getalp/UFSAC) extracted to
`~/data/ufsac-public-2.1/` -- see the script's own docstring for all
three. Candidates are restricted to each mention's own tagged part of
speech (unlike GlossBERT's deliberate no-POS-filtering), matching the
reference's own candidate generation exactly.

```python
--8<-- "examples/ewiser_reproduction.py"
```

Run with:

```bash
uv run python examples/ewiser_reproduction.py
```

```text
=== SemCor (ewiser.semcor_base.pt) ===
dataset   precision@1  published
SE07            68.8%      71.0%
SE2             77.3%      77.5%
SE3             77.8%      77.9%
SE13            76.5%      76.4%
SE15            78.1%      77.8%
ALL             76.9%      77.0%

=== SemCor + untagged glosses (ewiser.semcor.pt) ===
dataset   precision@1  published
SE07            70.8%      71.0%
SE2             78.9%      78.9%
SE3             77.9%      78.4%
SE13            79.0%      78.9%
SE15            79.5%      79.3%
ALL             78.2%      78.3%

=== SemCor + tagged glosses + WordNet Examples (ewiser.semcor+wngt.pt) ===
dataset   precision@1  published
SE07            74.5%      75.2%
SE2             80.7%      80.8%
SE3             77.6%      79.0%
SE13            80.5%      80.7%
SE15            82.5%      81.8%
ALL             79.7%      80.1%
```

**The `ALL` column -- the paper's own headline metric -- lands within
0.1-0.4 points on all three checkpoints.** Getting here took real
diagnosis against a live oracle, not code review alone: the reference's
own Python 3.10 (`qbert`/fairseq) environment was stood up locally, and a
single instance's score was traced layer by layer against it until the
first point of divergence was found -- twice.

### Bugs found and fixed

**1 -- tokenization must not pre-split punctuation-attached words.**
`tokenizer(words, is_split_into_words=True)` still runs BERT's own
punctuation-splitting pre-tokenizer on each provided word, so a UFSAC
token like `"Oct."` became two pre-tokens (`"Oct"`, `"."`) before
wordpiece ever ran, producing a *standalone* `"."` token where the
reference (wordpiece applied directly to the literal string, no
pre-splitting) produces a `"##."` continuation piece instead -- a
different vocabulary entry, unrelated embedding. First measured impact:
61.3% vs. a 71.0% published target on SemEval-2007. Traced by comparing a
single sentence's mean-pooled word vector for "reported" against the
reference's own real BERT forward pass -- the vectors matched almost
perfectly for words *without* internal punctuation, and diverged sharply
(different vocab id entirely) exactly at `"Oct."`/`"Sept."` positions.
Fixed with a from-scratch WordPiece implementation
([`wordpiece_tokenize_words`](../reference/algorithms.md)) applied to
each word in isolation, matching the reference's own literal per-word
call.

**2 -- the encoder input must sum the last 4 BERT layers, not just the
final one.** Every released checkpoint's own recorded
`context_embeddings_use_all_hidden=False` reads, in isolation, as "use
only the final layer" -- a reasonable interpretation of that one config
field, and wrong. The reference's own `TaggerModel.build_model` computes
its actual internal `use_all_hidden` flag from
`len(args.context_embeddings_type) > 1` -- a variable-name mix-up
(`context_embeddings_type` is the string `"bert"`, not the real layer
list) that happens to make `use_all_hidden` unconditionally `True` for
every BERT-backed checkpoint, silently overriding the correctly-named,
correctly-read flag. Found by extracting the reference's own real
internal representation via `model.forward_encode_words(...)` directly
and comparing its norm (~82) against a single-layer extraction's norm
(~18) -- an unmistakable, not-subtle divergence once actually measured,
not something a second reading of the config field would have surfaced.

**A separate, real memory bug**, found while testing on the full `ALL`
split (7253 instances): `EwiserEncoder.score()` was holding every
distinct sentence's full `[num_words, 117664]` logits matrix in memory
simultaneously for the whole call. Fixed by consuming and releasing each
sentence-batch's logits immediately after extracting the scores it
contributes, before encoding the next batch.

### The residual gap

The small remaining per-dataset gaps (mostly on the smallest eval sets,
e.g. SemEval-2007's 455 instances) are a separately diagnosed and
verified residual, not a further bug: this port's Hugging Face
`transformers` encoder and the reference's original
`pytorch_pretrained_bert` encoder produce cosine-~0.99 (not 1.0)
mean-pooled word vectors for bit-identical input and bit-identical
embedding weights -- confirmed directly (matching `layer_norm_eps`,
matching `hidden_act`, word-embedding weights diffing by exactly `0.0`)
to be accumulated cross-library BERT forward-pass numerics, not a port
bug. A checkpoint's `BatchNorm` was calibrated tightly enough to the
original library's exact numerics that this residual can flip a close
call on individual instances, with a proportionally bigger effect on
small eval sets than on `ALL`.

## Verifying the training path

[Reproducing the paper's published results](#reproducing-the-papers-published-results)
validates `EwiserEncoder`'s *inference* path by loading the paper's own
published checkpoints. It never touches
[`EwiserTrainer`](../reference/train.md) at all. This example is the
counterpart: it trains a fresh `bert-base-cased`
[`EwiserEncoder`](../reference/algorithms.md) from scratch via
`EwiserTrainer` on a real slice of SemCor, and checks that real learning
happens.

**Small subset, and not expected to approach published numbers** -- unlike
[Reproducing the paper's published results](#reproducing-the-papers-published-results)
(a hard acceptance gate for the checkpoint path), this script has no such
bar. EWISER's own published results critically depend on initializing
the output layer (`decoder.logits.weight`) from externally pretrained
LMMS/SensEmBERT sense embeddings; loaders for those vectors were added in
[issue #57](https://github.com/jmccrae/linkingtk/issues/57) (see
[Loading LMMS/SensEmBERT sense embeddings](#loading-lmmssensembert-sense-embeddings)),
but this script doesn't use them -- from-scratch training here starts
that layer randomly. The point is to verify the *training code itself* --
`EwiserTrainer`'s cross-entropy loss, per-sentence batching, and
freeze-then-thaw schedule -- end to end against a real pretrained encoder
and real data, not to reproduce the paper's numbers.

This script does wire up EWISER's own distinguishing idea, the WordNet
relation-graph propagation step (`build_relation_adjacency`) -- an
earlier version trained a plain frozen-BERT + FFN classifier with no
graph at all, which meant it wasn't really exercising "EWISER," just a
baseline sense classifier. The graph's practical benefit here is still
capped by construction: `SenseVocabulary.from_wn` restricts the
vocabulary to gold senses actually observed in this slice, so every
neighbor the graph can propagate from is already a labeled entry -- there
is no truly-unseen synset for the mechanism to generalize to, unlike
training against the full ~117k-synset inventory.

This is also the only place the freeze-then-thaw schedule
(`freeze_output_epochs`) gets exercised against a real training run, not
just synthetic-data unit tests
(`tests/algorithms/wsd/test_ewiser_trainer.py`).

```python
--8<-- "examples/ewiser_benchmark.py"
```

Run with:

```bash
uv run python examples/ewiser_benchmark.py
```

```text
2964 train instances (6 docs) / 669 eval instances (3 docs)
Most-frequent-sense baseline: {'precision@1': 0.205, 'recall': 0.205, 'f1': 0.205}
Vocabulary size: 1521
Relation graph edges: 911
Untrained precision@1: {'precision@1': 0.395, 'recall': 0.395, 'f1': 0.395}
Per-epoch held-out Hits@1 (via EwiserTrainer.eval_history):
  epoch 1: {'Hits@1': 0.631, 'MRR': 0.767}
  epoch 2: {'Hits@1': 0.622, 'MRR': 0.761}
  epoch 3: {'Hits@1': 0.641, 'MRR': 0.769}
Trained precision@1 (via EwiserLinker.link, the real production path):
  {'precision@1': 0.641, 'recall': 0.641, 'f1': 0.641}
```

Real, meaningful learning on a real pretrained backbone: 39.5% (untrained)
→ 64.1% (trained, 3 epochs on ~3K instances), clearing the most-frequent-
sense baseline (20.5%) by a wide margin. `eval_history`'s per-epoch
numbers match the independent `EwiserLinker.link()` check exactly,
confirming the training path and the production inference path agree --
the same regression this repo's own
[`CrossEncoderTrainer`](../reference/train.md) history caught a real bug
on for GlossBERT (see
[GlossBERT's training verification](glossbert.md#verifying-the-training-path)).

Hits@1 dips slightly after the first epoch (63.1% → 62.2%) before
recovering -- expected noise on a slice this small (~3K instances, no
pretrained sense-embedding init), not a sign of a training-path bug; the
freeze-then-thaw transition at epoch 1 (`decoder.logits.weight` unfreezes,
learning rate drops to `output_unfreeze_lr`) briefly perturbs a model
that had already found a good frozen-output-layer optimum on this small
slice.

## Loading LMMS/SensEmBERT sense embeddings

[Verifying the training path](#verifying-the-training-path) trains
`EwiserEncoder` from scratch with a randomly initialized output layer,
and notes that the paper's own published numbers depend on initializing
that layer from externally pretrained LMMS/SensEmBERT sense embeddings
instead (Section 4.2, "Output Embeddings", of the paper) -- this example
exercises the [issue #57](https://github.com/jmccrae/linkingtk/issues/57)
loaders that make that possible:
[`load_synset_centroid_vectors`](../reference/algorithms.md) and
[`build_synset_centroid_vectors_from_lmms`](../reference/algorithms.md).

No training happens here. Two real, independent checks:

1. `load_synset_centroid_vectors` against the reference's own
   ``res/embeddings/sensembert+lmms.svd512.synset-centroid.vec`` -- the
   literal file `bin/train-ewiser.sh` (in a checkout of
   [SapienzaNLP/ewiser](https://github.com/SapienzaNLP/ewiser)) trains
   against, already combining SensEmBERT (nouns) with LMMS (other POS)
   and already SVD-reduced to 512d: the paper's own best ("LMMS +
   SensEmBERT") configuration from Table 2, baked in. Since this file
   *is* `SenseVocabulary.from_offsets_file`'s own token space (both come
   from the same reference release), coverage should land near 100% --
   a real acceptance check, not a fuzzy one.
2. `build_synset_centroid_vectors_from_lmms` against a raw LMMS release
   (sensekey-keyed, from [danlou/LMMS](https://github.com/danlou/LMMS)),
   building the same shape of tensor from scratch: resolving each
   sensekey to a synset, centroiding, and reducing with truncated SVD.
   There's no bit-exact reference to check this against (no raw
   SensEmBERT file or spec is available -- see the loader module's own
   docstring for why), so this only confirms the pipeline runs correctly
   end to end on real data.

Requires `res/dictionaries/offsets.txt` and
`res/embeddings/sensembert+lmms.svd512.synset-centroid.vec` from a
checkout of the reference repo, and a raw LMMS vectors file from
[danlou/LMMS](https://github.com/danlou/LMMS)'s "Download Sense
Embeddings" section -- see the script's own docstring for exact paths.

```python
--8<-- "examples/ewiser_pretrained_output_embedding.py"
```

Run with:

```bash
uv run python examples/ewiser_pretrained_output_embedding.py
```

```text
SenseVocabulary: 117664 entries (from ~/external/ewiser/res/dictionaries/offsets.txt)
load_synset_centroid_vectors: 117659/117664 matched (100.0%) from ~/Downloads/sensembert+lmms.svd512.synset-centroid.vec
build_synset_centroid_vectors_from_lmms: 117659/117664 matched (100.0%) from ~/Downloads/lmms-sp-wsd.albert-xxlarge-v2.vectors.txt
  reduced to 512d, sample row norm: 1.0000
```

Both loaders match 117,659 of 117,664 vocabulary entries (99.996%) --
as close to total coverage as this vocabulary gets (the handful of
unmatched slots are `SenseVocabulary`'s own reserved special tokens,
which no vector file could ever cover). The raw-LMMS path reaching the
same coverage as the pre-built reference file is notable on its own: this
particular LMMS release (the newer "Reloaded" AIJ 2022 profile, not the
paper's original 2019 one) propagates a vector to essentially every
WordNet sense via gloss embeddings, not just senses observed in annotated
data. The reduced vectors' unit-length norm (1.0000) confirms the
paper's stated final-state invariant actually holds after truncated SVD,
not just before it.

This does **not** mean a fresh `EwiserEncoder` built this way reproduces
a released checkpoint's own `decoder.logits.weight` bit-for-bit: the
reference's own two-stage training freezes this initialization for stage
1 only, then unfreezes and trains it for 20 more epochs in stage 2 --
real, expected drift from this raw initialization, not a bug to chase.

## Full-corpus training

[Verifying the training path](#verifying-the-training-path) only
verifies `EwiserTrainer`'s training *code path* on a tiny slice (6 docs,
~3K instances, a vocabulary restricted to gold senses observed in that
slice). This is the full-scale counterpart: the real 117,664-entry
WordNet vocabulary, a real from-scratch WordNet relation graph over that
whole vocabulary, and `decoder.logits.weight` initialized from the real
paper-best "LMMS + SensEmBERT" pretrained sense embeddings (issue #57)
rather than a random init -- trained on all 37,176 sentences of real
SemCor, evaluated on UFSAC's real Raganato et al. framework test sets.

**A real bug was fixed to make this possible** (issue #58): at the full
117,664-entry vocabulary, `StructuredLogits`'s graph-propagation step
with `structured_logits_trainable=True` (the default, and what the
reference's own `bin/train-ewiser.sh` uses in both training stages) used
to OOM -- `torch.sparse.mm`'s default backward materializes a dense
`[117664, 117664]` gradient (~51.6GB) instead of an `O(nnz)` one.
Confirmed directly: the OOM's own reported allocation size
(55,379,492,864 bytes) matches `117664**2 * 4 bytes` almost exactly, and
was invisible until now (`ewiser_benchmark.py`'s ~1,500-entry vocab
keeps that dense gradient to ~9MB). Fixed with `_SparsePropagate`, a
custom `torch.autograd.Function` computing the same gradient via an
`O(nnz)` indexed gather-multiply-sum -- see
[`_ewiser_structured_logits.py`](../reference/algorithms.md)'s own
module docstring, and its test file for a `gradcheck`-verified
correctness test. Measured directly on real hardware: 53.7ms per
8-sentence batch, 2.9GB peak GPU memory at the *full* vocabulary with a
trainable graph -- indistinguishable in cost from a non-trainable one.

**Acceptance threshold**: the closest real published number for this
exact training-data recipe (SemCor only, hypernymy-only relation graph)
is `ewiser.semcor_base.pt`'s own published ALL F1 of 77.0% -- the same
row [Reproducing the paper's published results](#reproducing-the-papers-published-results)
reproduced to within 0.1 points against the released checkpoint. Not a
hard bit-exact gate here (this script trains its own relation graph from
`wn` rather than loading the checkpoint's baked-in one -- real, expected
structural differences, documented in `_ewiser_graph.py`'s own
docstring). Two checks, against measured ALL F1: a **floor** (must clear
the most-frequent-sense baseline by 10 points, or training is actually
broken) and a **target** (within 5 points of 77.0%, i.e. >= 72.0%,
reported but not asserted -- a multi-hour run's output shouldn't be
discarded on a near miss).

Requires UFSAC 2.1, EWISER's own `res/dictionaries/offsets.txt` and
`res/embeddings/sensembert+lmms.svd512.synset-centroid.vec` -- see the
script's own docstring for exact paths.

```python
--8<-- "examples/ewiser_full_training.py"
```

Run with (detached, since this takes hours):

```bash
nohup uv run python -u examples/ewiser_full_training.py \
    > /tmp/ewiser_full_training.log 2>&1 & disown
```

**Real run, this sandbox's single RTX 4090**: 70 epochs completed in
18,341s (~5.1 hours), matching the ~5-hour planning estimate closely.
Loss dropped from 4.99 (epoch 1) to 0.049 (epoch 70), with a visible
drop right at the freeze-to-thaw transition (epoch 51, lr 1e-4 -> 1e-5)
as the previously-frozen output layer started fine-tuning. Dev (SE07)
Hits@1 climbed from 0.669 (epoch 1) to a stable ~0.70-0.71 band by the
last ~15 epochs.

Final evaluation on the real held-out UFSAC test sets:

```text
dataset   precision@1   published (SemCor)
SE2             76.6%                77.5%
SE3             76.2%                77.9%
SE13            75.3%                76.4%
SE15            75.8%                77.8%
ALL             75.8%                77.0%
ALL F1 = 0.758
target met: ALL F1 within 0.05 of 0.77 (published ewiser.semcor_base.pt)
```

**Both acceptance checks passed.** The floor check (ALL F1 must clear
the most-frequent-sense baseline by 10 points) passed without needing to
print anything (an `assert`, not a verdict line). The target check
passed with real margin: 75.8% is 1.2 points short of the published
77.0%, well inside the 5-point (72.0%) bar -- 3.8 points of margin to
spare. The gap is consistent across every test set (0.9-2.0 points, no
outliers), matching what's expected from the already-documented
structural differences from the reference's own recipe (a from-scratch
WordNet relation graph via `wn`, not the checkpoint's baked-in
BabelNet-id-based one -- see `_ewiser_graph.py`'s own docstring) rather
than pointing at a bug to chase.

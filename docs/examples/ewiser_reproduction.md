# EWISER: reproducing the paper's published results

Like [`glossbert_reproduction.md`](glossbert_reproduction.md), this runs
**no training at all** -- see
[the training verification example](ewiser_benchmark.md) for that side
instead.
[`EwiserEncoder.from_checkpoint`](../reference/algorithms.md) loads one of
the three checkpoints [Bevilacqua & Navigli](https://www.aclweb.org/anthology/2020.acl-main.255/)
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

## Bugs found and fixed

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

## The residual gap

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

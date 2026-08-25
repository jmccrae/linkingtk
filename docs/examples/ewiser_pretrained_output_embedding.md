# EWISER: loading LMMS/SensEmBERT sense embeddings

[`ewiser_benchmark.md`](ewiser_benchmark.md) trains `EwiserEncoder` from
scratch with a randomly initialized output layer, and notes that the
paper's own published numbers depend on initializing that layer from
externally pretrained LMMS/SensEmBERT sense embeddings instead (Section
4.2, "Output Embeddings", of the paper) -- this example exercises the
[issue #57](https://github.com/jmccrae/linkingtk/issues/57) loaders that
make that possible:
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

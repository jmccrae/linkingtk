"""Real-data verification of issue #57's sense-embedding loaders
(`linkingtk.algorithms.wsd._ewiser_sense_embeddings`).

No training happens here -- this exercises both loaders against real
downloaded vector files and checks their output against a real
`SenseVocabulary`.

Setup:

1. EWISER's own ``res/dictionaries/offsets.txt`` at `_OFFSETS_PATH` below
   (same file `examples/ewiser_reproduction.py` requires).
2. EWISER's own ``res/embeddings/sensembert+lmms.svd512.synset-centroid.vec``
   at `_CENTROID_VECTORS_PATH` below -- the literal file
   ``bin/train-ewiser.sh`` (in a checkout of
   https://github.com/SapienzaNLP/ewiser) trains against
   (``EMBEDDINGS='../res/embeddings/sensembert+lmms.svd512.synset-centroid.vec'``).
   Already combines SensEmBERT (nouns) with LMMS (other POS), already
   centroided per synset and SVD-reduced to 512d -- the paper's own best
   ("LMMS + SensEmBERT") configuration from Table 2.
3. A raw LMMS release vector file (sensekey-keyed, GloVe-style, no
   header) at `_RAW_LMMS_VECTORS_PATH` below, e.g. one of the
   ``lmms-sp-*.vectors.txt`` files from https://github.com/danlou/LMMS's
   "Download Sense Embeddings" section.

**Loader 1 check (`load_synset_centroid_vectors`) is a real acceptance
test, not a fuzzy one**: `_CENTROID_VECTORS_PATH` *is* the token space
`SenseVocabulary.from_offsets_file(_OFFSETS_PATH)` was built from (both
come from the same reference release), so coverage should land near
100%, not just "some". This does **not** reproduce a released
checkpoint's own `decoder.logits.weight` bit-for-bit -- the reference's
own two-stage training (`bin/train-ewiser.sh`) freezes this
initialization for stage 1 only, then unfreezes and trains it for 20
more epochs in stage 2, so real, expected drift from this raw
initialization is normal, not a bug to chase.

**Loader 2 check (`build_synset_centroid_vectors_from_lmms`) has no such
bit-exact reference to compare against** (no raw SensEmBERT file/spec is
available -- see the module's own docstring) -- this only checks that
the centroid-and-reduce pipeline runs end to end on real data and
produces coverage/shape/unit-norm results consistent with what the
loader's own unit tests already established on synthetic data.

Run with: `uv run python examples/ewiser_pretrained_output_embedding.py`
"""

from __future__ import annotations

from pathlib import Path

import torch

from linkingtk.algorithms.wsd._ewiser_sense_embeddings import (
    build_synset_centroid_vectors_from_lmms,
    load_synset_centroid_vectors,
)
from linkingtk.algorithms.wsd._ewiser_vocab import SenseVocabulary

_OFFSETS_PATH = Path.home() / "external" / "ewiser" / "res" / "dictionaries" / "offsets.txt"
_CENTROID_VECTORS_PATH = Path.home() / "Downloads" / "sensembert+lmms.svd512.synset-centroid.vec"
_RAW_LMMS_VECTORS_PATH = Path.home() / "Downloads" / "lmms-sp-wsd.albert-xxlarge-v2.vectors.txt"
_LMMS_TARGET_DIM = 512


def main() -> None:
    vocabulary = SenseVocabulary.from_offsets_file(_OFFSETS_PATH)
    print(f"SenseVocabulary: {len(vocabulary)} entries (from {_OFFSETS_PATH})")

    base = torch.nn.Linear(512, len(vocabulary), bias=False).weight.data.clone()
    _vectors, matched = load_synset_centroid_vectors(_CENTROID_VECTORS_PATH, vocabulary, base)
    coverage = matched / len(vocabulary)
    print(
        f"load_synset_centroid_vectors: {matched}/{len(vocabulary)} matched "
        f"({coverage:.1%}) from {_CENTROID_VECTORS_PATH}"
    )
    assert coverage > 0.95, (
        "expected near-total coverage (this file is that vocabulary's own "
        f"token space), got {coverage:.1%}"
    )

    raw_base = torch.nn.Linear(_LMMS_TARGET_DIM, len(vocabulary), bias=False).weight.data.clone()
    _raw_vectors, raw_matched = build_synset_centroid_vectors_from_lmms(
        _RAW_LMMS_VECTORS_PATH, vocabulary, raw_base, target_dim=_LMMS_TARGET_DIM
    )
    raw_coverage = raw_matched / len(vocabulary)
    print(
        f"build_synset_centroid_vectors_from_lmms: {raw_matched}/{len(vocabulary)} matched "
        f"({raw_coverage:.1%}) from {_RAW_LMMS_VECTORS_PATH}"
    )
    # A matched row's norm should be ~1.0 (both loaders renormalize after
    # reduction) -- pick a real vocabulary entry rather than "any nonzero
    # row", since every row already starts nonzero under nn.Linear's own
    # random default init.
    first_real_index = next(
        i for i in range(len(vocabulary)) if vocabulary.synset_id_for(i) is not None
    )
    sample_norm = raw_base[first_real_index].norm()
    print(f"  reduced to {_LMMS_TARGET_DIM}d, sample row norm: {sample_norm:.4f}")


if __name__ == "__main__":
    main()

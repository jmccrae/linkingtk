"""Pretrained sense-embedding loaders for `EwiserEncoder`'s output layer.

EWISER's own strongest results (Section 4.2, "Output Embeddings", of the
paper) come from initializing `decoder.logits.weight` -- the sense output
embedding layer -- from off-the-shelf sense vectors rather than a random
init, then optionally freezing it for a few epochs
([EwiserTrainer.freeze_output_epochs][linkingtk.train.ewiser_trainer.EwiserTrainer]).
Two off-the-shelf vector sets are named in the paper: LMMS 2048d
(Loureiro and Jorge, 2019) and SensEmBERT+LMMS (Scarlini et al., 2020,
enhancing LMMS with BabelNet+Wikipedia for nouns, backed off to LMMS for
verbs/adjectives/adverbs -- SensEmBERT's own vectors are "in the same
space as LMMS", per the paper). The paper's own recorded pipeline for
both (confirmed directly from its Section 4.2 text, not from the
reference's CC-BY-NC-SA-licensed scripts): normalize each raw sense
vector to unit length, take the centroid of every sense in a synset,
reduce to 512d with truncated SVD, normalize again.

Two loaders, for two different starting points:

- `load_synset_centroid_vectors` loads an already-built, already-reduced
  synset-vector file -- e.g. the literal
  ``res/embeddings/sensembert+lmms.svd512.synset-centroid.vec`` the
  reference's own training script uses (confirmed against
  ``bin/train-ewiser.sh``'s `$EMBEDDINGS` variable), a GloVe-style file
  keyed by WordNet 3.0 offset tokens (``"wn:02084071n"``, same convention
  `SenseVocabulary.from_offsets_file` already parses). This is the
  fastest path to the paper's own best configuration when such a file is
  available -- no SVD or centroiding needed at load time.
- `build_synset_centroid_vectors_from_lmms` builds the same shape of
  tensor from LMMS's own raw, sensekey-keyed release files (also
  GloVe-style, no header -- confirmed directly against a real downloaded
  LMMS release) -- for users who only have the raw per-sensekey vectors,
  not a pre-built synset file.

Neither loader ingests SensEmBERT's own raw (pre-combination) vector
files -- SensEmBERT was never open-sourced and its own site
(sensembert.org) is no longer reachable, so there is no real file or
confirmed format to build and verify a loader against. Use
`load_synset_centroid_vectors` with a pre-built SensEmBERT+LMMS file (as
the reference's own release provides) to get the paper's full best
configuration; `build_synset_centroid_vectors_from_lmms` alone only
covers the LMMS-only configuration (still a real, evaluated row in the
paper's own Table 2, just not its best one).
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from linkingtk.algorithms.wsd._ewiser_vocab import SenseVocabulary
from linkingtk.sources.wn import sensekey_to_synset_id, wn30_offset_to_synset_id

logger = logging.getLogger("linkingtk")


def load_synset_centroid_vectors(
    path: str | Path,
    vocabulary: SenseVocabulary,
    base: torch.Tensor,
    lexicon: str = "omw-en:1.4",
    progress: bool = True,
) -> tuple[torch.Tensor, int]:
    """Load a pre-built, offset-keyed synset-vector file into `base`, in place.

    Args:
        path: A GloVe-style plaintext file, no header, one line per
            synset: ``"<token> <float> <float> ...\\n"``, where `token`
            is a WordNet 3.0 offset (``"wn:02084071n"``) -- the same
            format and token convention as `offsets.txt`
            (`SenseVocabulary.from_offsets_file`), and the format of the
            reference's own ``res/embeddings/*.svd512.synset-centroid.vec``
            resource files. A line whose token isn't in that shape (e.g.
            a dummy padding entry) is skipped, not an error.
        vocabulary: The `SenseVocabulary` to align rows against.
        base: A ``[len(vocabulary), dim]`` tensor to fill in place --
            typically a freshly-constructed decoder's own
            ``decoder.logits.weight.data`` (or a clone of it), so that
            vocabulary entries this file has no vector for keep whatever
            init `base` already had rather than being zeroed.
        lexicon: Passed to `wn30_offset_to_synset_id`.
        progress: Show a `tqdm` progress bar (this file is typically
            ~100K+ lines).

    Returns:
        ``(base, matched)`` -- `base` (mutated in place, also returned
        for chaining) and the number of `vocabulary` entries a vector was
        found for.
    """
    from tqdm import tqdm

    vectors_by_synset: dict[str, torch.Tensor] = {}
    with Path(path).open() as handle:
        lines = tqdm(handle, desc=str(path), unit=" lines", disable=not progress)
        for line in lines:
            line = line.rstrip("\n")
            if not line:
                continue
            token, *floats = line.split(" ")
            if not token.startswith("wn:"):
                continue
            synset_id = wn30_offset_to_synset_id(token, lexicon=lexicon)
            if synset_id is None:
                continue
            vectors_by_synset[synset_id] = torch.tensor([float(x) for x in floats])

    matched = _fill_from_synset_vectors(base, vocabulary, vectors_by_synset)
    logger.info(
        "load_synset_centroid_vectors: %d/%d vocabulary entries matched (%s)",
        matched,
        len(vocabulary),
        path,
    )
    return base, matched


def build_synset_centroid_vectors_from_lmms(
    path: str | Path,
    vocabulary: SenseVocabulary,
    base: torch.Tensor,
    target_dim: int | None = None,
    lexicon: str = "omw-en:1.4",
    progress: bool = True,
) -> tuple[torch.Tensor, int]:
    """Build a synset-centroid init tensor from LMMS's own raw sensekey vectors.

    Reimplements the paper's own recorded pipeline (Section 4.2, "Output
    Embeddings" -- not ported from the reference's CC-BY-NC-SA-licensed
    ``bin/get_centroids.py``/``bin/reduce_dims.py``, though independently
    confirmed to agree with them): each raw sense vector is L2-normalized,
    every sense sharing a synset (via `sensekey_to_synset_id`) is averaged
    into that synset's centroid, then (if `target_dim` differs from the
    raw dimensionality) reduced with `sklearn.decomposition.TruncatedSVD`
    and L2-normalized again -- the paper states the *final* vectors are
    unit-length, and SVD projection doesn't preserve norms, so the second
    normalization is necessary to actually reach that state, not optional
    polish.

    This resolves every sensekey in `path` via `sensekey_to_synset_id`
    (one `wn` lookup each) -- a real LMMS release has ~100K-200K
    sensekeys, so this is a slow, one-time vocabulary-init pass, not
    something to call per training step.

    Args:
        path: A GloVe-style plaintext file, no header, one line per
            sensekey: ``"<sensekey> <float> <float> ...\\n"`` (LMMS's own
            release format -- confirmed directly against a real
            downloaded release).
        vocabulary: The `SenseVocabulary` to align rows against.
        base: A ``[len(vocabulary), dim]`` tensor to fill in place, same
            contract as `load_synset_centroid_vectors`'s `base`.
        target_dim: Reduce to this many dimensions via truncated SVD if
            given and different from the file's own raw dimensionality
            (the paper reduces LMMS's raw 2048d to 512d). `None` (the
            default) keeps the raw dimensionality as-is.
        lexicon: Passed to `sensekey_to_synset_id`.
        progress: Show a `tqdm` progress bar.

    Returns:
        ``(base, matched)``, same shape of result as
        `load_synset_centroid_vectors`.
    """
    from tqdm import tqdm

    vectors_by_synset: dict[str, list[torch.Tensor]] = {}
    with Path(path).open() as handle:
        lines = tqdm(handle, desc=str(path), unit=" lines", disable=not progress)
        for line in lines:
            line = line.rstrip("\n")
            if not line:
                continue
            sense_key, *floats = line.split(" ")
            synset_id = sensekey_to_synset_id(sense_key, lexicon=lexicon)
            if synset_id is None:
                continue
            vector = torch.tensor([float(x) for x in floats])
            vectors_by_synset.setdefault(synset_id, []).append(_l2_normalize(vector))

    centroids = {
        synset_id: _l2_normalize(torch.stack(vectors).mean(0))
        for synset_id, vectors in vectors_by_synset.items()
    }

    if centroids and target_dim is not None:
        raw_dim = next(iter(centroids.values())).shape[0]
        if target_dim != raw_dim:
            centroids = _reduce_dims(centroids, target_dim)

    matched = _fill_from_synset_vectors(base, vocabulary, centroids)
    logger.info(
        "build_synset_centroid_vectors_from_lmms: %d/%d vocabulary entries matched (%s)",
        matched,
        len(vocabulary),
        path,
    )
    return base, matched


def _l2_normalize(vector: torch.Tensor) -> torch.Tensor:
    normalized: torch.Tensor = vector / vector.norm()
    return normalized


def _reduce_dims(centroids: dict[str, torch.Tensor], target_dim: int) -> dict[str, torch.Tensor]:
    import numpy as np
    from sklearn.decomposition import TruncatedSVD

    keys = list(centroids.keys())
    matrix = torch.stack([centroids[key] for key in keys]).numpy()
    reduced = TruncatedSVD(n_components=target_dim, random_state=42).fit_transform(matrix)
    reduced = reduced / np.linalg.norm(reduced, axis=1, keepdims=True)
    return {
        key: torch.tensor(reduced[index], dtype=torch.float32) for index, key in enumerate(keys)
    }


def _fill_from_synset_vectors(
    base: torch.Tensor, vocabulary: SenseVocabulary, vectors_by_synset: dict[str, torch.Tensor]
) -> int:
    matched = 0
    for index in range(len(vocabulary)):
        synset_id = vocabulary.synset_id_for(index)
        if synset_id is None:
            continue
        vector = vectors_by_synset.get(synset_id)
        if vector is None:
            continue
        base[index] = vector
        matched += 1
    return matched

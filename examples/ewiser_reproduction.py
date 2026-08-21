"""Reproduces EWISER's published results using the paper's own checkpoints,
through this repo's `EwiserLinker`/`UfsacDataset` pipeline.

No training happens here -- this loads Bevilacqua & Navigli's own three
released checkpoints (https://github.com/SapienzaNLP/ewiser, ACL 2020)
directly into `EwiserEncoder.from_checkpoint` and evaluates each via
`EwiserLinker.link()`, the exact same production code path a freshly
trained model would use. **This is a hard acceptance gate for issue #40,
not a best-effort verification pass**: measured numbers are expected to
match the published ones (within a stated, verified methodology
difference, not an assumed one).

Setup:

1. The three checkpoints (from the Google Drive links in the EWISER
   README's "EWISER English checkpoints" section) at
   ``~/Downloads/ewiser.semcor.pt``, ``ewiser.semcor+wngt.pt``,
   ``ewiser.semcor_base.pt`` (or point `_CHECKPOINTS` below elsewhere).
2. EWISER's own ``res/dictionaries/offsets.txt`` (from a checkout of
   https://github.com/SapienzaNLP/ewiser) at `_OFFSETS_PATH` below --
   the checkpoints' frequency-sorted output-vocabulary order, not
   reproducible from first principles (see `SenseVocabulary`).
3. UFSAC 2.1 (https://github.com/getalp/UFSAC#content-of-the-repository)
   extracted to `_UFSAC_DIR` below -- this script reads its
   ``raganato_*.xml`` files, the same Raganato et al. (2017) WSD
   evaluation framework data EWISER's own paper reports numbers against.

Checkpoint -> published-row mapping: each checkpoint's own recorded
training args (its ``data`` path and ``save_dir`` name, both readable
straight from the checkpoint file) were cross-referenced against the
paper's Table 3 (`paper/ewiser.pdf`, extracted via ``pdftotext -layout``
for exact digits, not read visually) to identify precisely which trained
configuration each file is:

- ``ewiser.semcor_base.pt``: ``data=.../semcor``, ``save_dir`` names
  ``hyper`` (no ``hypo``) -- SemCor only, EWISER_hyper, Table 3's
  ``S-only, hyper`` row.
- ``ewiser.semcor.pt``: ``data=.../semcor+glosses_main.untagged``,
  ``save_dir`` names ``hyper+hypo`` -- SemCor + untagged glosses,
  EWISER_hyper+hypo, Table 3's ``S+G, hyper+hypo`` row.
- ``ewiser.semcor+wngt.pt``: ``data=.../semcor+glosses_main+examples``,
  ``save_dir`` names ``hyper`` -- SemCor + tagged glosses + WordNet
  examples, EWISER_hyper, Table 3's ``S+G+G+/E, hyper`` row.

This orders exactly the same way as the README's own "EWISER English
checkpoints" bullet list (SemCor / SemCor + untagged glosses / SemCor +
tagged glosses + WordNet Examples), which is the confirming cross-check
that this mapping is right, not a guess from filenames alone.

Candidates are restricted to each mention's own tagged part of speech
(``EwiserLinker``'s candidates, filtered by `_PosRestrictedMatch` below) --
unlike `glossbert_reproduction.py`'s deliberate no-POS-filtering (matching
GlossBERT's own paper convention), EWISER's reference candidate generation
*does* restrict to ``(lemma, POS)`` (confirmed directly against
``ewiser/fairseq_ext/data/wsd_dataset.py``'s
``lemma_pos_to_possible_senses``), so matching that is required to
reproduce its published numbers, not optional fidelity.

Measured results (ALL column, the paper's own headline metric) match
published numbers to within 0.1-0.4 points on all three checkpoints:
SemCor 76.9% vs. 77.0%, SemCor+untagged-glosses 78.2% vs. 78.3%,
SemCor+tagged-glosses+examples 79.7% vs. 80.1%. Getting here required
finding and fixing two real bugs, chased down by tracing a single
instance's score, layer by layer, against the reference's own real
forward pass running in its own (Python 3.10, `qbert`/fairseq) environment
(see `~/external/ewiser/.venv`) -- not accepted as a plausible shortfall:

1. Subword tokenization must NOT use
   ``tokenizer(words, is_split_into_words=True)``: that high-level call
   still runs BERT's punctuation-splitting pre-tokenizer on each provided
   word, so a word with internal punctuation (e.g. UFSAC's own
   ``"Oct."``) gets pre-split into ``"Oct"``/``"."`` before wordpiece,
   producing a standalone ``"."`` token where the reference (wordpiece
   applied directly to the literal string, no pre-splitting) produces a
   ``"##."`` continuation piece -- a different vocabulary entry, unrelated
   embedding. Fixed by
   [wordpiece_tokenize_words][linkingtk.algorithms.wsd._ewiser_text.wordpiece_tokenize_words],
   a from-scratch WordPiece implementation applied to each word in
   isolation.
2. The encoder input must be the **sum of the last 4 BERT hidden-state
   layers**, not just the final layer, **despite** every released
   checkpoint's own recorded `context_embeddings_use_all_hidden=False`
   (which reads as "single layer only"). That flag is dead: the
   reference's own `TaggerModel.build_model` computes `use_all_hidden`
   from `len(args.context_embeddings_type) > 1` -- a variable-name
   mix-up (`context_embeddings_type` is the string `"bert"`, not the
   actual layer list) that happens to make `use_all_hidden` always `True`
   for any BERT-backed checkpoint, silently overriding the
   correctly-named flag. See
   [EwiserEncoder][linkingtk.algorithms.wsd.ewiser.EwiserEncoder]'s
   `num_summed_layers` docstring.

The remaining small per-dataset gaps (mostly on the smallest eval sets,
e.g. SemEval-2007's 455 instances) are consistent with a real, separately
diagnosed and verified residual: this port's HF `transformers` encoder and
the reference's original `pytorch_pretrained_bert` encoder produce
cosine-~0.99 (not 1.0) mean-pooled word vectors for bit-identical input
and bit-identical weights -- confirmed directly, layer by layer, to be
accumulated cross-library BERT forward-pass numerics (matching
`layer_norm_eps`, matching `hidden_act`, bit-identical embedding weights),
not a further port bug. A checkpoint's `BatchNorm` was calibrated tightly
enough to the original library's exact numerics that this residual
difference can flip a close call on individual instances, with a bigger
relative effect on small eval sets than on `ALL`.

Run with: `uv run python examples/ewiser_reproduction.py`
"""

from __future__ import annotations

from pathlib import Path

import torch

from linkingtk.algorithms.wsd.ewiser import EwiserEncoder, EwiserLinker
from linkingtk.blocking.base import BlockingStrategy
from linkingtk.blocking.exact import ExactMatch
from linkingtk.core.entity import Entity
from linkingtk.core.source import EntitySource
from linkingtk.datasets.ufsac import UfsacDataset
from linkingtk.eval import Evaluator

_CHECKPOINT_DIR = Path.home() / "Downloads"
_OFFSETS_PATH = Path.home() / "external" / "ewiser" / "res" / "dictionaries" / "offsets.txt"
_UFSAC_DIR = Path.home() / "data" / "ufsac-public-2.1"

# (UFSAC filename stem, README column label)
_EVAL_SETS = [
    ("raganato_semeval2007", "SE07"),
    ("raganato_senseval2", "SE2"),
    ("raganato_senseval3", "SE3"),
    ("raganato_semeval2013", "SE13"),
    ("raganato_semeval2015", "SE15"),
    ("raganato_ALL", "ALL"),
]

# (checkpoint filename, README label, published F1 per _EVAL_SETS label --
# see module docstring for how these were matched to Table 3's rows)
_CHECKPOINTS = [
    (
        "ewiser.semcor_base.pt",
        "SemCor",
        {"SE07": 71.0, "SE2": 77.5, "SE3": 77.9, "SE13": 76.4, "SE15": 77.8, "ALL": 77.0},
    ),
    (
        "ewiser.semcor.pt",
        "SemCor + untagged glosses",
        {"SE07": 71.0, "SE2": 78.9, "SE3": 78.4, "SE13": 78.9, "SE15": 79.3, "ALL": 78.3},
    ),
    (
        "ewiser.semcor+wngt.pt",
        "SemCor + tagged glosses + WordNet Examples",
        {"SE07": 75.2, "SE2": 80.8, "SE3": 79.0, "SE13": 80.7, "SE15": 81.8, "ALL": 80.1},
    ),
]

_PENN_TO_WORDNET_POS = {"N": "n", "V": "v", "J": "a", "R": "r"}


def _wordnet_pos_of_synset_id(synset_id: str) -> str | None:
    pos = synset_id.rsplit("-", 1)[-1]
    if pos == "s":  # adjective satellite counts as a plain adjective for POS filtering
        return "a"
    return pos if pos in "nvar" else None


def _mention_wordnet_pos(mention: Entity) -> str | None:
    tag = mention.properties.get("pos")
    return _PENN_TO_WORDNET_POS.get(tag[0]) if tag else None


class _PosRestrictedMatch(BlockingStrategy):
    """Wraps `ExactMatch`, keeping only candidates whose WordNet POS matches
    the mention's own tagged POS (see module docstring)."""

    def __init__(self, top_k: int = 50) -> None:
        self._inner = ExactMatch(top_k=top_k)

    def candidate_pairs(
        self, dataset1: list[Entity], dataset2: list[Entity] | EntitySource
    ) -> list[tuple[Entity, Entity]]:
        pairs = self._inner.candidate_pairs(dataset1, dataset2)
        filtered = []
        for entity1, entity2 in pairs:
            mention_pos = _mention_wordnet_pos(entity1)
            if mention_pos is None or _wordnet_pos_of_synset_id(entity2.id) == mention_pos:
                filtered.append((entity1, entity2))
        return filtered


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    blocking = _PosRestrictedMatch(top_k=50)

    for filename, label, published in _CHECKPOINTS:
        print(f"\n=== {label} ({filename}) ===")
        encoder = EwiserEncoder.from_checkpoint(
            _CHECKPOINT_DIR / filename, _OFFSETS_PATH, forward_batch_size=32
        )
        encoder.to(device)
        linker = EwiserLinker(encoder)

        print(f"{'dataset':<8} {'precision@1':>12} {'published':>10}")
        for stem, eval_label in _EVAL_SETS:
            mentions, senses, ground_truth = UfsacDataset(
                source=str(_UFSAC_DIR / f"{stem}.xml")
            ).load()

            results = linker.link(mentions, senses, blocking=blocking)
            predictions = [(result.source_id, result.target_id) for result in results]
            report = Evaluator.evaluate(predictions=predictions, ground_truth=ground_truth)
            score = report.metrics["precision@1"] * 100

            print(f"{eval_label:<8} {score:>11.1f}% {published[eval_label]:>9.1f}%")


if __name__ == "__main__":
    main()

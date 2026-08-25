"""Trains `EwiserEncoder` on the *full* real SemCor corpus (37,176
sentences) with the paper's own architecture and best-known
initialization, then evaluates against UFSAC's real Raganato et al.
framework test sets -- the full-scale counterpart to
`examples/ewiser_benchmark.py`'s small, fast training-path smoke test.

Follow-up to #40 and #57 (LMMS/SensEmBERT output-embedding init): unlike
`ewiser_benchmark.py`, which restricts the vocabulary to gold senses
observed in its own small training slice, this trains a genuinely
**full-inventory** classifier -- the real 117,664-entry WordNet
vocabulary (`SenseVocabulary.from_offsets_file`), a real from-scratch
WordNet relation graph over that whole vocabulary
(`build_relation_adjacency`), and `decoder.logits.weight` initialized
from the real, paper-best "LMMS + SensEmBERT" pretrained sense
embeddings (`load_synset_centroid_vectors`,
`sensembert+lmms.svd512.synset-centroid.vec`) rather than a random init
-- so the graph-propagation step has real unseen synsets to generalize
to, and `EwiserTrainer.freeze_output_epochs` is finally protecting a
real pretrained init, not a no-op.

**Long-running.** Like `glossbert_full_training.py`, meant to run for
hours, unattended (`nohup ... & disown`, not a foreground `uv run`).

**A real bug was fixed to make this possible** (issue #58): at this
vocabulary size, `StructuredLogits`'s graph-propagation step with
`structured_logits_trainable=True` (the default, and what the
reference's own `bin/train-ewiser.sh` uses in both training stages) used
to OOM -- `torch.sparse.mm`'s default backward materializes a dense
``[117664, 117664]`` gradient (~51.6GB) instead of an ``O(nnz)`` one. See
`_ewiser_structured_logits.py`'s module docstring for the fix
(`_SparsePropagate`, a custom `torch.autograd.Function`). With the fix,
measured directly on this sandbox's single RTX 4090: 53.7ms per
8-sentence batch, 2.9GB peak GPU memory, at the *full* vocabulary +
trainable graph -- indistinguishable in cost from a non-trainable graph.
37,176 sentences / batch_size=8 -> ~4,647 batches/epoch -> ~4.2
min/epoch. `bin/train-ewiser.sh`'s own recipe (50 frozen epochs at
lr=1e-4, then 20 thawed epochs at lr=1e-5) is therefore ~5 hours total --
comfortably an overnight run, with margin.

**Hyperparameters, matching the reference's own `bin/train-ewiser.sh`
where GPU-safe**: `decoder_hidden_dim=512` (SVD-reduction target of the
pretrained output embeddings too), `model_name_or_path="bert-large-cased"`
(the paper's own encoder, unlike `ewiser_benchmark.py`'s deliberately
smaller/faster `bert-base-cased` smoke-test choice -- confirmed
comfortably fast enough at this scale by the measurement above),
`freeze_output_epochs=50, output_freeze_lr=1e-4, output_unfreeze_lr=1e-5`.
Candidates are restricted to each mention's own tagged part of speech
(`_PosRestrictedMatch` below, duplicated from `ewiser_reproduction.py` --
each example script here is self-contained), matching the reference's
own candidate generation exactly, for both dev-time monitoring and final
test-time evaluation.

**Acceptance threshold**: the closest real published number for this
exact training-data recipe (SemCor only, hypernymy-only relation graph)
is `ewiser.semcor_base.pt`'s own published ALL F1 of 77.0% (the same row
`ewiser_reproduction.py` reproduced to within 0.1 points against the
released checkpoint). This is **not** a hard bit-exact gate the way
`ewiser_reproduction.py`'s checkpoint reproduction is -- unlike that
script, this one trains its own relation graph from `wn` rather than
loading the checkpoint's baked-in BabelNet-id-based one, and
`_ewiser_graph.py`'s own docstring already documents that a from-scratch
graph "cannot be verified bit-exactly against EWISER's own -- only
structurally". Two checks are reported at the end, against measured ALL
F1:

- **Floor** (asserted): must clear the most-frequent-sense baseline by
  at least 10 points -- failing this means training is actually broken,
  not just under-tuned.
- **Target** (printed, not asserted -- a multi-hour run's real output
  shouldn't be discarded on a near miss): within 5 points of 77.0%
  (i.e. >= 72.0%) is reported "met"; a bigger gap is printed plainly, to
  be diagnosed rather than silently accepted.

Requires UFSAC 2.1 extracted to `~/data/ufsac-public-2.1/` (same as
`ewiser_reproduction.py`), EWISER's own `res/dictionaries/offsets.txt`
and `res/embeddings/sensembert+lmms.svd512.synset-centroid.vec` (from a
checkout of https://github.com/SapienzaNLP/ewiser) at `_OFFSETS_PATH`/
`_CENTROID_VECTORS_PATH` below.

Run with (from the repo root), detached so it survives the shell exiting:
```
nohup uv run python examples/ewiser_full_training.py \
    > /tmp/ewiser_full_training.log 2>&1 & disown
```
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import torch

from linkingtk.algorithms.wsd._ewiser_graph import build_relation_adjacency
from linkingtk.algorithms.wsd._ewiser_sense_embeddings import load_synset_centroid_vectors
from linkingtk.algorithms.wsd._ewiser_vocab import SenseVocabulary
from linkingtk.algorithms.wsd.ewiser import EwiserEncoder, EwiserLinker
from linkingtk.blocking.base import BlockingStrategy
from linkingtk.blocking.exact import ExactMatch
from linkingtk.core.entity import Entity, label_texts
from linkingtk.core.source import EntitySource
from linkingtk.datasets.ufsac import UfsacDataset
from linkingtk.eval import Evaluator
from linkingtk.train.arguments import TrainingArguments
from linkingtk.train.ewiser_trainer import EwiserTrainer

_UFSAC_DIR = Path.home() / "data" / "ufsac-public-2.1"
_OFFSETS_PATH = Path.home() / "external" / "ewiser" / "res" / "dictionaries" / "offsets.txt"
_CENTROID_VECTORS_PATH = Path.home() / "Downloads" / "sensembert+lmms.svd512.synset-centroid.vec"
_OUTPUT_DIR = Path("./models/ewiser_semcor_full")
_DECODER_HIDDEN_DIM = 512
_MODEL_NAME = "bert-large-cased"
_TARGET_ALL_F1 = 0.770
_TARGET_MARGIN = 0.05
_MFS_FLOOR_MARGIN = 0.10

# (UFSAC filename stem, README column label, ewiser.semcor_base.pt's own published F1)
_EVAL_SETS = [
    ("raganato_senseval2", "SE2", 77.5),
    ("raganato_senseval3", "SE3", 77.9),
    ("raganato_semeval2013", "SE13", 76.4),
    ("raganato_semeval2015", "SE15", 77.8),
    ("raganato_ALL", "ALL", 77.0),
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
    the mention's own tagged POS -- see `ewiser_reproduction.py`'s own copy
    of this class for the full rationale."""

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
    logging.basicConfig(
        level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s", force=True
    )
    logging.getLogger("linkingtk").setLevel(logging.INFO)
    t0 = time.time()

    vocabulary = SenseVocabulary.from_offsets_file(_OFFSETS_PATH)
    print(f"[{time.time() - t0:.0f}s] SenseVocabulary: {len(vocabulary)} entries")

    adjacency = build_relation_adjacency(vocabulary)
    print(f"[{time.time() - t0:.0f}s] relation graph: {adjacency.values().numel()} edges")

    base = torch.nn.Linear(_DECODER_HIDDEN_DIM, len(vocabulary), bias=False).weight.data.clone()
    _vectors, matched = load_synset_centroid_vectors(_CENTROID_VECTORS_PATH, vocabulary, base)
    print(
        f"[{time.time() - t0:.0f}s] output-embedding init: {matched}/{len(vocabulary)} "
        f"matched ({matched / len(vocabulary):.1%})"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = EwiserEncoder(
        model_name_or_path=_MODEL_NAME,
        vocabulary=vocabulary,
        adjacency=adjacency,
        decoder_hidden_dim=_DECODER_HIDDEN_DIM,
        structured_logits_trainable=True,
        output_embedding_init=base,
        max_length=128,
    )
    encoder.to(device)
    linker = EwiserLinker(encoder)
    blocking = _PosRestrictedMatch(top_k=50)

    train_mentions, senses, train_gt = UfsacDataset(source=str(_UFSAC_DIR / "semcor.xml")).load()
    print(f"[{time.time() - t0:.0f}s] loaded {len(train_gt)} SemCor training instances")

    # Most-frequent-sense baseline: the same sanity floor ewiser_benchmark.py checks.
    mfs_predictions = [
        (mention.id, candidates[0].id)
        for mention in train_mentions
        for candidates in [senses.search(label_texts(mention)[0], top_k=1)]
        if candidates
    ]
    mfs_report = Evaluator.evaluate(predictions=mfs_predictions, ground_truth=train_gt)
    mfs_f1 = mfs_report.metrics["f1"] or 0.0
    print(f"[{time.time() - t0:.0f}s] most-frequent-sense baseline (train set): {mfs_f1:.3f}")

    dev_mentions, _senses, dev_gt = UfsacDataset(
        source=str(_UFSAC_DIR / "raganato_semeval2007.xml")
    ).load()
    print(f"[{time.time() - t0:.0f}s] loaded {len(dev_gt)} SemEval-2007 dev instances")

    args = TrainingArguments(
        output_dir=str(_OUTPUT_DIR),
        learning_rate=1e-4,
        num_epochs=70,
        batch_size=8,
        device=device,
    )
    trainer = EwiserTrainer(
        model=encoder,
        args=args,
        train_data=(train_mentions, train_gt),
        eval_data=(dev_mentions, dev_gt),
        eval_dataset2=senses,
        blocking=blocking,
        freeze_output_epochs=50,
        output_freeze_lr=1e-4,
        output_unfreeze_lr=1e-5,
    )

    print(f"[{time.time() - t0:.0f}s] starting training ({args.num_epochs} epochs)")
    trainer.train()
    for i, epoch_report in enumerate(trainer.eval_history):
        print(f"[{time.time() - t0:.0f}s] epoch {i + 1} dev (SE07): {epoch_report.metrics}")
    print(f"[{time.time() - t0:.0f}s] training complete, checkpoint at {_OUTPUT_DIR}/model.pt")

    print(f"[{time.time() - t0:.0f}s] evaluating on real held-out test sets")
    print(f"{'dataset':<8} {'precision@1':>12} {'published (SemCor)':>20}")
    all_f1 = None
    for stem, label, published in _EVAL_SETS:
        mentions, _senses, ground_truth = UfsacDataset(
            source=str(_UFSAC_DIR / f"{stem}.xml")
        ).load()
        results = linker.link(mentions, senses, blocking=blocking)
        predictions = [(r.source_id, r.target_id) for r in results]
        report = Evaluator.evaluate(predictions=predictions, ground_truth=ground_truth)
        score = (report.metrics["precision@1"] or 0.0) * 100
        print(f"{label:<8} {score:>11.1f}% {published:>19.1f}%")
        if label == "ALL":
            all_f1 = report.metrics["f1"] or 0.0

    assert all_f1 is not None  # "ALL" is always in _EVAL_SETS
    print(f"[{time.time() - t0:.0f}s] ALL F1 = {all_f1:.3f}")
    assert all_f1 > mfs_f1 + _MFS_FLOOR_MARGIN, (
        f"floor check failed: ALL F1 ({all_f1:.3f}) doesn't clear the MFS baseline "
        f"({mfs_f1:.3f}) by {_MFS_FLOOR_MARGIN} -- training is likely broken, not just under-tuned"
    )
    if all_f1 >= _TARGET_ALL_F1 - _TARGET_MARGIN:
        print(
            f"[{time.time() - t0:.0f}s] target met: ALL F1 within {_TARGET_MARGIN} of "
            f"{_TARGET_ALL_F1} (published ewiser.semcor_base.pt)"
        )
    else:
        gap = _TARGET_ALL_F1 - all_f1
        print(
            f"[{time.time() - t0:.0f}s] below target: ALL F1 {gap:.3f} short of "
            f"{_TARGET_ALL_F1} (published ewiser.semcor_base.pt) -- worth diagnosing"
        )
    print(f"[{time.time() - t0:.0f}s] done")


if __name__ == "__main__":
    main()

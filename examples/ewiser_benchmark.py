"""Trains EwiserEncoder from scratch (real BERT, real gradient descent) on a
real slice of SemCor, and evaluates on held-out documents -- the
training-path counterpart to `examples/ewiser_reproduction.py`, which only
exercises inference against the paper's own published checkpoints.

**Small subset, not the paper's setup, and not expected to approach
published numbers.** Unlike `ewiser_reproduction.py` (a hard acceptance
gate for issue #40), this script has no such bar: EWISER's own published
results critically depend on initializing the output layer
(`decoder.logits.weight`) from externally pretrained LMMS/SensEmBERT
sense embeddings (Section 3.4 of the paper). Issue #57 added loaders for
those vectors (`load_synset_centroid_vectors`/
`build_synset_centroid_vectors_from_lmms` in
`linkingtk.algorithms.wsd._ewiser_sense_embeddings`, exercised against
real downloaded vector files in
`examples/ewiser_pretrained_output_embedding.py`), but this script
doesn't use them -- from-scratch training here starts that layer
randomly, and trains on a small `_TRAIN_DOCS`-document slice of SemCor
for a handful of epochs, not the full corpus. This exists to verify the
*training path itself* --
`EwiserTrainer`'s cross-entropy loss, per-sentence batching, and
freeze/unfreeze schedule all work correctly on real data and a real
pretrained encoder -- not to reproduce the paper's numbers.

This *does* wire up EWISER's own distinguishing idea -- the WordNet
relation-graph propagation step (`build_relation_adjacency`), unlike an
earlier version of this script which trained a plain frozen-BERT + FFN
classifier with no graph at all. Its practical benefit here is still
capped by the restricted vocabulary (see below): structured logits exist
to let the model score synsets it never saw labeled in training, by
propagating from labeled *neighbors* -- but every neighbor this graph can
reach is, by construction, already a labeled vocabulary entry, so there's
no truly-unseen synset for it to generalize to in this setup.

Also exercises the freeze-then-thaw schedule end to end on real data
(`freeze_output_epochs=1`): `decoder.logits.weight` starts frozen for the
first epoch, then unfreezes -- verified here via a real training run, not
just synthetic-data unit tests (see `tests/algorithms/wsd/test_ewiser_trainer.py`
for the synthetic-data version of this same check).

Requires UFSAC 2.1 extracted to `~/data/ufsac-public-2.1/` -- see
`ewiser_reproduction.py`'s docstring.

Run with: `uv run python examples/ewiser_benchmark.py`
"""

from __future__ import annotations

from pathlib import Path

import torch

from linkingtk.algorithms.wsd._ewiser_graph import build_relation_adjacency
from linkingtk.algorithms.wsd._ewiser_vocab import SenseVocabulary
from linkingtk.algorithms.wsd.ewiser import EwiserEncoder, EwiserLinker
from linkingtk.blocking.exact import ExactMatch
from linkingtk.core.entity import label_texts
from linkingtk.datasets.ufsac import UfsacDataset
from linkingtk.eval import Evaluator
from linkingtk.train.arguments import TrainingArguments
from linkingtk.train.ewiser_trainer import EwiserTrainer

_UFSAC_DIR = Path.home() / "data" / "ufsac-public-2.1"
_TRAIN_DOCS = 6
_EVAL_DOCS = 3
_MODEL_NAME = "bert-base-cased"  # smaller/faster than the paper's bert-large-cased


def _document_id(mention_id: str) -> str:
    # UfsacDataset's positional mention ids (semcor.xml has no native
    # per-word id, unlike the raganato_*.xml files) -- "ufsac:{doc}:{sent}:{index}".
    return mention_id.split(":")[1]


def main() -> None:
    mentions, senses, ground_truth = UfsacDataset(source=str(_UFSAC_DIR / "semcor.xml")).load()
    mentions_by_id = {m.id: m for m in mentions}

    doc_ids = sorted({_document_id(m.id) for m in mentions})
    train_docs = set(doc_ids[:_TRAIN_DOCS])
    eval_docs = set(doc_ids[_TRAIN_DOCS : _TRAIN_DOCS + _EVAL_DOCS])
    train_gt = [(mid, tid) for mid, tid in ground_truth if _document_id(mid) in train_docs]
    eval_gt = [(mid, tid) for mid, tid in ground_truth if _document_id(mid) in eval_docs]
    train_mentions = [mentions_by_id[mid] for mid, _tid in train_gt]
    eval_mentions = [mentions_by_id[mid] for mid, _tid in eval_gt]
    print(
        f"{len(train_gt)} train instances ({_TRAIN_DOCS} docs) / "
        f"{len(eval_gt)} eval instances ({_EVAL_DOCS} docs)"
    )

    blocking = ExactMatch(top_k=30)

    # Most-frequent-sense baseline: WordNet's own sense order (most-common
    # first) with no model at all -- the standard WSD sanity-check floor.
    mfs_predictions = [
        (mention.id, candidates[0].id)
        for mention in eval_mentions
        for candidates in [senses.search(label_texts(mention)[0], top_k=1)]
        if candidates
    ]
    mfs_report = Evaluator.evaluate(predictions=mfs_predictions, ground_truth=eval_gt)
    print(f"Most-frequent-sense baseline: {mfs_report.metrics}")

    # From-scratch vocabulary: every gold sense observed in *this slice*
    # (train+eval docs only, not all of SemCor's ~33K distinct senses) --
    # not the full ~117k WordNet inventory either, since from-scratch
    # training has no compatibility need to match it (see
    # SenseVocabulary.from_wn).
    vocabulary = SenseVocabulary.from_wn([synset_id for _mid, synset_id in train_gt + eval_gt])
    print(f"Vocabulary size: {len(vocabulary)}")

    # Wires up EWISER's own distinguishing idea -- the WordNet
    # graph-propagation step -- so this benchmark actually exercises it,
    # not just a plain frozen-BERT + FFN classifier. Built from this
    # restricted vocabulary (see the comment above), so its practical
    # benefit here is capped: structured logits exist to let the model
    # score synsets it never saw labeled by propagating from labeled
    # neighbors, but every neighbor this graph can reach is, by
    # construction, already a labeled vocabulary entry.
    adjacency = build_relation_adjacency(vocabulary)
    print(f"Relation graph edges: {adjacency.values().numel()}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = EwiserEncoder(
        model_name_or_path=_MODEL_NAME,
        vocabulary=vocabulary,
        adjacency=adjacency,
        max_length=128,
    )
    encoder.to(device)
    linker = EwiserLinker(encoder)

    before = Evaluator.evaluate(
        predictions=[
            (r.source_id, r.target_id)
            for r in linker.link(eval_mentions, senses, blocking=blocking)
        ],
        ground_truth=eval_gt,
    )
    print(f"Untrained precision@1: {before.metrics}")

    args = TrainingArguments(
        output_dir="./models/ewiser_semcor_subset",
        learning_rate=1e-3,
        num_epochs=3,
        batch_size=8,
        device=device,
    )
    trainer = EwiserTrainer(
        model=encoder,
        args=args,
        train_data=(train_mentions, train_gt),
        eval_data=(eval_mentions, eval_gt),
        eval_dataset2=senses,
        blocking=blocking,
        freeze_output_epochs=1,
        output_freeze_lr=1e-3,
        output_unfreeze_lr=1e-4,
    )
    trainer.train()
    print("Per-epoch held-out Hits@1 (via EwiserTrainer.eval_history):")
    for i, epoch_report in enumerate(trainer.eval_history):
        print(f"  epoch {i + 1}: {epoch_report.metrics}")

    after = Evaluator.evaluate(
        predictions=[
            (r.source_id, r.target_id)
            for r in linker.link(eval_mentions, senses, blocking=blocking)
        ],
        ground_truth=eval_gt,
    )
    print("Trained precision@1 (via EwiserLinker.link, the real production path):")
    print(f"  {after.metrics}")


if __name__ == "__main__":
    main()

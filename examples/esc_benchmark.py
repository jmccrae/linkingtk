"""Trains `EscEncoder` from scratch (real BART, real gradient descent) on a
real slice of SemCor, and evaluates on held-out documents -- the
training-path counterpart to `examples/esc_reproduction.py`, which only
exercises inference against the paper's own published checkpoint.

**Small subset and a smaller backbone, not the paper's setup, and not
expected to approach published numbers.** Unlike `esc_reproduction.py` (a
hard acceptance gate for issue #41), this script has no such bar: this
exists to verify the *training path itself* -- `EscTrainer`'s extractive
-QA span-position loss, per-mention candidate resolution/shuffling, and
batching all work correctly on real data and a real pretrained encoder --
not to reproduce the paper's numbers. The reference's own full recipe also
does Poisson-sampled distractor-gloss noise during training
(`add_glosses_noise`) and trains ``facebook/bart-large`` for far longer
than a handful of epochs on a 9-document slice; this uses
``facebook/bart-base`` instead (same "smaller/faster backbone for the
training-path check" convention `ewiser_benchmark.py` already uses for
EWISER's `bert-base-cased` vs. the paper's `bert-large-cased`).

Requires UFSAC 2.1 extracted to `~/data/ufsac-public-2.1/` -- see
`esc_reproduction.py`'s docstring.

Run with: `uv run python examples/esc_benchmark.py`
"""

from __future__ import annotations

from pathlib import Path

import torch

from linkingtk.algorithms.wsd.esc import EscEncoder, EscLinker
from linkingtk.blocking.exact import ExactMatch
from linkingtk.core.entity import label_texts
from linkingtk.datasets.ufsac import UfsacDataset
from linkingtk.eval import Evaluator
from linkingtk.train.arguments import TrainingArguments
from linkingtk.train.esc_trainer import EscTrainer

_UFSAC_DIR = Path.home() / "data" / "ufsac-public-2.1"
_TRAIN_DOCS = 6
_EVAL_DOCS = 3
_MODEL_NAME = "facebook/bart-base"  # smaller/faster than the paper's facebook/bart-large


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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = EscEncoder(model_name_or_path=_MODEL_NAME, max_length=512, forward_batch_size=4)
    encoder.to(device)
    linker = EscLinker(encoder)

    before = Evaluator.evaluate(
        predictions=[
            (r.source_id, r.target_id)
            for r in linker.link(eval_mentions, senses, blocking=blocking)
        ],
        ground_truth=eval_gt,
    )
    print(f"Untrained precision@1: {before.metrics}")

    args = TrainingArguments(
        output_dir="./models/esc_semcor_subset",
        learning_rate=3e-5,
        num_epochs=5,
        batch_size=4,
        device=device,
    )
    trainer = EscTrainer(
        model=encoder,
        args=args,
        train_data=(train_mentions, train_gt),
        senses=senses,
        eval_data=(eval_mentions, eval_gt),
        blocking=blocking,
    )
    trainer.train()
    print("Per-epoch training loss (via EscTrainer.loss_history):")
    for i, loss in enumerate(trainer.loss_history):
        print(f"  epoch {i + 1}: {loss:.4f}")
    print("Per-epoch held-out Hits@1 (via EscTrainer.eval_history):")
    for i, epoch_report in enumerate(trainer.eval_history):
        print(f"  epoch {i + 1}: {epoch_report.metrics}")

    after = Evaluator.evaluate(
        predictions=[
            (r.source_id, r.target_id)
            for r in linker.link(eval_mentions, senses, blocking=blocking)
        ],
        ground_truth=eval_gt,
    )
    print("Trained precision@1 (via EscLinker.link, the real production path):")
    print(f"  {after.metrics}")


if __name__ == "__main__":
    main()

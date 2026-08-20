"""Trains GlossBertEncoder from scratch (real `bert-base-uncased`, real
gradient descent) on a real slice of SemCor, and evaluates on held-out
documents -- the training-path counterpart to
`examples/glossbert_reproduction.py`, which only exercises inference
against the paper's published checkpoint.

**Small subset, not the paper's setup.** The paper trains on all of
SemCor (~226K sense-tagged instances) for 6 epochs at `batch_size=64`,
`max_seq_length=512` -- multiple GPU-hours even on this sandbox's single
RTX 4090. This script instead trains on the first `_TRAIN_DOCS` Brown
Corpus documents (~3K instances) for 3 epochs at `max_length=128`, runs in
under a minute, and exists to verify the *training path itself* --
`CrossEncoderTrainer`'s loss, hard-negative mining, and optimizer setup
end to end on real data and a real pretrained backbone -- not to
reproduce the paper's published numbers (that's what the checkpoint-based
`glossbert_reproduction.py` is for). Don't expect these numbers to
approach the published ones; a fraction of the data and epochs will not
reach that quality, and that's not the point here.

**Regression context**: this script's own `eval_dataset2=senses` wiring
matters -- `CrossEncoderTrainer._evaluate()` defaults to deriving its
candidate pool from `eval_data`'s own (already-correct) targets alone,
which for WSD silently omits every real decoy sense and overstates
Hits@1. Measured directly during this benchmark's development: 60%
self-reported Hits@1 vs. 33% via the real `GlossBertLinker.link()` path
on the same held-out mentions -- see `CrossEncoderTrainer`'s own
docstring for the fix.

Requires UFSAC 2.1 extracted to `~/data/ufsac-public-2.1/` -- see
`glossbert_reproduction.py`'s docstring.

Run with: `uv run python examples/glossbert_benchmark.py`
"""

from __future__ import annotations

from pathlib import Path

import torch

from linkingtk.algorithms.wsd.glossbert import GlossBertLinker
from linkingtk.blocking.exact import ExactMatch
from linkingtk.core.entity import Entity, label_texts
from linkingtk.datasets.ufsac import UfsacDataset
from linkingtk.eval import Evaluator
from linkingtk.sources.wn import WnEntitySource
from linkingtk.train.arguments import TrainingArguments
from linkingtk.train.cross_encoder import CrossEncoderTrainer

_UFSAC_DIR = Path.home() / "data" / "ufsac-public-2.1"
_TRAIN_DOCS = 6
_EVAL_DOCS = 3


def _document_id(mention_id: str) -> str:
    # UfsacDataset's positional mention ids (semcor.xml has no native
    # per-word id, unlike the raganato_*.xml files) -- "ufsac:{doc}:{sent}:{index}".
    return mention_id.split(":")[1]


def _resolve_pairs(
    ground_truth: list[tuple[str, str]],
    mentions_by_id: dict[str, Entity],
    senses: WnEntitySource,
) -> list[tuple[Entity, Entity]]:
    cache: dict[str, Entity | None] = {}
    pairs = []
    for mention_id, synset_id in ground_truth:
        if synset_id not in cache:
            cache[synset_id] = senses.get(synset_id)
        sense = cache[synset_id]
        if sense is not None:
            pairs.append((mentions_by_id[mention_id], sense))
    return pairs


def main() -> None:
    mentions, senses, ground_truth = UfsacDataset(source=str(_UFSAC_DIR / "semcor.xml")).load()
    mentions_by_id = {m.id: m for m in mentions}

    doc_ids = sorted({_document_id(m.id) for m in mentions})
    train_docs = set(doc_ids[:_TRAIN_DOCS])
    eval_docs = set(doc_ids[_TRAIN_DOCS : _TRAIN_DOCS + _EVAL_DOCS])
    train_gt = [(mid, tid) for mid, tid in ground_truth if _document_id(mid) in train_docs]
    eval_gt = [(mid, tid) for mid, tid in ground_truth if _document_id(mid) in eval_docs]
    train_pairs = _resolve_pairs(train_gt, mentions_by_id, senses)
    eval_pairs = _resolve_pairs(eval_gt, mentions_by_id, senses)
    eval_mentions = list({m.id: m for m, _s in eval_pairs}.values())
    print(
        f"{len(train_pairs)} train instances ({_TRAIN_DOCS} docs) / "
        f"{len(eval_pairs)} eval instances ({_EVAL_DOCS} docs)"
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

    linker = GlossBertLinker(model_name_or_path="bert-base-uncased", max_length=128)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    linker.model.to(device)

    before = Evaluator.evaluate(
        predictions=[
            (r.source_id, r.target_id)
            for r in linker.link(eval_mentions, senses, blocking=blocking)
        ],
        ground_truth=eval_gt,
    )
    print(f"Untrained precision@1: {before.metrics}")

    args = TrainingArguments(
        output_dir="./models/glossbert_semcor_subset",
        learning_rate=2e-5,
        negative_samples_ratio=3,
        num_epochs=3,
        batch_size=16,
        weight_decay=0.01,
        warmup_ratio=0.1,
        device=device,
    )
    trainer = CrossEncoderTrainer(
        model=linker.model,
        args=args,
        train_data=train_pairs,
        eval_data=eval_pairs,
        eval_dataset2=senses,
        blocking=blocking,
    )
    trainer.train()
    print("Per-epoch held-out Hits@1 (via CrossEncoderTrainer.eval_history):")
    for i, epoch_report in enumerate(trainer.eval_history):
        print(f"  epoch {i + 1}: {epoch_report.metrics}")

    after = Evaluator.evaluate(
        predictions=[
            (r.source_id, r.target_id)
            for r in linker.link(eval_mentions, senses, blocking=blocking)
        ],
        ground_truth=eval_gt,
    )
    print("Trained precision@1 (via GlossBertLinker.link, the real production path):")
    print(f"  {after.metrics}")


if __name__ == "__main__":
    main()

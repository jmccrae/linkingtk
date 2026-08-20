"""Trains GlossBertEncoder on the *full* real SemCor corpus (~230K
sense-tagged instances) with near-paper-faithful hyperparameters, then
evaluates the result against UFSAC's real Raganato et al. framework test
sets -- the full-scale counterpart to `examples/glossbert_benchmark.py`'s
small, fast verification slice.

**Long-running.** Unlike every other example in this repo, this is meant
to run for hours, unattended (`nohup ... & disown`, not a foreground
`uv run`). Measured directly on this sandbox's single RTX 4090 before
launching: ~545 rows/sec (positives + mined negatives combined) at
`batch_size=64`, `max_length=512`, so ~230K positives (expanding to
~700K-1M rows with negatives) is roughly 20-30 minutes per epoch, low
single-digit GB of peak GPU memory (WSD sentences are short despite the
paper's `max_length=512`) -- comfortably an overnight run at the paper's
own 6 epochs, with margin.

**Hyperparameters, matching the paper (`README.md`/`run_classifier_WSD_sent.py`
in the reference implementation) where GPU-safe**: `learning_rate=2e-5`,
`weight_decay=0.01`, `warmup_ratio=0.1`, `max_length=512`,
`num_epochs=6`. `batch_size=64` matches the paper's own effective batch
(no gradient accumulation needed here -- measured peak memory leaves
comfortable headroom on a 24GB GPU at this scale). Candidates are every
WordNet sense of a lemma across all POS
(`ExactMatch(top_k=50)`/`negative_samples_ratio=49`), matching the
paper's own candidate generation, not an artificially-capped subset.

**Dev/test split matches the paper's own protocol**: train on all of
SemCor, monitor on SemEval-2007 (the paper's own dev set) via
`CrossEncoderTrainer.eval_history` every epoch, and only evaluate on the
real held-out test sets (SensEval-2/3, SemEval-2013/2015, ALL) once,
after training completes -- exactly like `glossbert_reproduction.py`'s
checkpoint-based evaluation, so the two are directly comparable.

Requires UFSAC 2.1 extracted to `~/data/ufsac-public-2.1/` -- see
`glossbert_reproduction.py`'s docstring.

Run with (from the repo root), detached so it survives the shell exiting:
```
nohup uv run python examples/glossbert_full_training.py \
    > /tmp/glossbert_full_training.log 2>&1 & disown
```
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import torch

from linkingtk.algorithms.wsd.glossbert import GlossBertLinker
from linkingtk.blocking.exact import ExactMatch
from linkingtk.core.entity import Entity
from linkingtk.datasets.ufsac import UfsacDataset
from linkingtk.eval import Evaluator
from linkingtk.sources.wn import WnEntitySource
from linkingtk.train.arguments import TrainingArguments
from linkingtk.train.cross_encoder import CrossEncoderTrainer

_UFSAC_DIR = Path.home() / "data" / "ufsac-public-2.1"
_OUTPUT_DIR = Path("./models/glossbert_semcor_full")

# (UFSAC filename stem, README column label, README's published "this checkpoint" score)
_TEST_SETS = [
    ("raganato_senseval2", "SE2", 77.7),
    ("raganato_senseval3", "SE3", 75.9),
    ("raganato_semeval2013", "SE13", 76.8),
    ("raganato_semeval2015", "SE15", 79.3),
    ("raganato_ALL", "ALL", 77.2),
]


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
    logging.basicConfig(
        level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s", force=True
    )
    logging.getLogger("linkingtk").setLevel(logging.INFO)
    t0 = time.time()
    train_mentions, senses, train_gt = UfsacDataset(source=str(_UFSAC_DIR / "semcor.xml")).load()
    train_mentions_by_id = {m.id: m for m in train_mentions}
    train_pairs = _resolve_pairs(train_gt, train_mentions_by_id, senses)
    print(f"[{time.time() - t0:.0f}s] loaded {len(train_pairs)} SemCor training instances")

    dev_mentions, _senses, dev_gt = UfsacDataset(
        source=str(_UFSAC_DIR / "raganato_semeval2007.xml")
    ).load()
    dev_mentions_by_id = {m.id: m for m in dev_mentions}
    dev_pairs = _resolve_pairs(dev_gt, dev_mentions_by_id, senses)
    print(f"[{time.time() - t0:.0f}s] loaded {len(dev_pairs)} SemEval-2007 dev instances")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    linker = GlossBertLinker(model_name_or_path="bert-base-uncased", max_length=512)
    linker.model.to(device)

    blocking = ExactMatch(top_k=50)
    args = TrainingArguments(
        output_dir=str(_OUTPUT_DIR),
        learning_rate=2e-5,
        negative_samples_ratio=49,
        num_epochs=6,
        batch_size=64,
        weight_decay=0.01,
        warmup_ratio=0.1,
        device=device,
    )
    trainer = CrossEncoderTrainer(
        model=linker.model,
        args=args,
        train_data=train_pairs,
        eval_data=dev_pairs,
        eval_dataset2=senses,
        blocking=blocking,
    )

    print(f"[{time.time() - t0:.0f}s] starting training ({args.num_epochs} epochs)")
    trainer.train()
    for i, epoch_report in enumerate(trainer.eval_history):
        print(f"[{time.time() - t0:.0f}s] epoch {i + 1} dev (SE07): {epoch_report.metrics}")
    print(f"[{time.time() - t0:.0f}s] training complete, checkpoint at {_OUTPUT_DIR}/model.pt")

    print(f"[{time.time() - t0:.0f}s] evaluating on real held-out test sets")
    print(f"{'dataset':<8} {'precision@1':>12} {'published':>10}")
    for stem, label, published in _TEST_SETS:
        mentions, _senses, ground_truth = UfsacDataset(
            source=str(_UFSAC_DIR / f"{stem}.xml")
        ).load()
        results = linker.link(mentions, senses, blocking=blocking)
        predictions = [(r.source_id, r.target_id) for r in results]
        report = Evaluator.evaluate(predictions=predictions, ground_truth=ground_truth)
        score = (report.metrics["precision@1"] or 0.0) * 100
        print(f"{label:<8} {score:>11.1f}% {published:>9.1f}%")
    print(f"[{time.time() - t0:.0f}s] done")


if __name__ == "__main__":
    main()

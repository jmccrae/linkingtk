"""Trains ReFinEDLinker's bi-encoder on AIDA-CoNLL's real train split and
reports precision@1 (== InKB micro-F1 in this gold-mention,
one-prediction-per-mention setting) on the held-out test split.

**Candidate-restricted, not exhaustive.** Unlike the EA/KGE benchmarks in
this repo (`examples/*_benchmark.py`), which rank exhaustively because
OpenEA's own reference methodology (`greedy_alignment`) does too (see
issue #37), ReFinED's own methodology is *not* exhaustive: Section 4.4
("Candidate generation") restricts each mention to its top-30 candidates
by entity prior before scoring at all. Ranking this repo's linker
exhaustively against AIDA-CoNLL's full test-split KB (~1,500 entities)
was tried first and measured Hits@1 = 0.29 -- a much harder task than the
paper's own (rank among 1,500 candidates vs. its curated 30), not a fair
comparison. [LabelOverlap][linkingtk.blocking.label_overlap.LabelOverlap]
with `max_matches=30` approximates that candidate-restriction width (this
repo has no popularity-based entity-prior data to build a closer
approximation), scored via [ReFinEDLinker.link][linkingtk.algorithms.el.refined.ReFinEDLinker.link]
directly -- the production code path, not a bespoke eval-only scoring
function.

Compared against ReFinED's own published "w/o pretraining (fine-tuned)"
ablation (Table 1 of Ayoola et al. 2022): ~84.6 F1 averaged across 6
datasets (no clean AIDA-only figure exists). That's the honest target
here, not the paper's 87.5-93.9 F1 headline numbers, which depend on
~100M-pair Wikipedia pretraining this doesn't reproduce. See
refined.py's module docstring for the full history of this design
decision (issues #45/#47).

Fetches the AIDA-CoNLL dataset (~24MB) and real Wikipedia description
extracts (~100+ small requests, batched) over the network the first time
it's run; both cached under ~/.cache/linkingtk/downloads after that.
Also downloads distilbert-base-uncased (~260MB) via `transformers` the
first time.

Run with: `uv run python examples/refined_benchmark.py`
"""

from __future__ import annotations

import torch

from linkingtk.algorithms.el.refined import ReFinEDLinker
from linkingtk.blocking.exact import ExactMatch
from linkingtk.blocking.label_overlap import LabelOverlap
from linkingtk.datasets.aida_conll import AidaConllDataset
from linkingtk.eval import Evaluator
from linkingtk.train.arguments import TrainingArguments
from linkingtk.train.trainer import Trainer


def main() -> None:
    dataset = AidaConllDataset()
    mentions, kb, _ground_truth = dataset.load()
    train_pairs, test_pairs, _val_pairs = dataset.load_splits()

    mentions_by_id = {entity.id: entity for entity in mentions}
    kb_by_id = {entity.id: entity for entity in kb}
    train_data = [(mentions_by_id[m], kb_by_id[e]) for m, e in train_pairs]

    print(f"{len(train_data)} train mentions / {len(test_pairs)} test mentions")

    linker = ReFinEDLinker(model_name="distilbert-base-uncased", embedding_dim=256, max_length=96)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = TrainingArguments(
        output_dir="./models/refined_aida_conll",
        learning_rate=2e-5,
        num_epochs=3,
        batch_size=32,
        negative_samples_ratio=4,
        loss="infonce",
        device=device,
    )
    Trainer(model=linker.encoder, args=args, train_data=train_data, blocking=ExactMatch()).train()

    test_source_ids = {m for m, _ in test_pairs}
    test_target_ids = {e for _, e in test_pairs}
    test_mentions = [entity for entity in mentions if entity.id in test_source_ids]
    test_kb = [kb_by_id[entity_id] for entity_id in test_target_ids]

    results = linker.link(
        test_mentions, test_kb, blocking=LabelOverlap(ngram_size=3, max_matches=30)
    )
    predictions = [(result.source_id, result.target_id) for result in results]
    report = Evaluator.evaluate(predictions=predictions, ground_truth=test_pairs)
    print(f"Metrics: {report.metrics}")
    print("Reference: ReFinED's published 'w/o pretraining' ablation is ~0.846 F1")
    print("(6-dataset average, approximate top-30-candidate width, not entity-prior-ranked)")


if __name__ == "__main__":
    main()

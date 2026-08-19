"""Trains BlinkLinker's bi-encoder on Zeshel's real train-domain split and
reports Hits@1/Hits@10/Hits@64 (== Recall@k) on the held-out test domains'
mentions, ranked against **all** of each domain's own KB entities.

**Exhaustive per-domain ranking, not candidate-restricted.** Unlike
ReFinED's benchmark (examples/refined_benchmark.py), which restricts each
mention to a curated top-30 candidate set (matching ReFinED's own
methodology), BLINK's own bi-encoder numbers (Wu et al. 2020, Table 1)
are a Recall@k over the *entire* per-domain entity dictionary -- that's
what a bi-encoder retrieval stage is actually measured on. So this script
encodes each test domain's mentions and its full KB, computes one dense
similarity matrix per domain, and reports Hits@64 (== Recall@64) via
Evaluator.evaluate_ranked -- the same exhaustive-ranking pattern
test_blink_benchmark.py's methodology test already exercises at toy scale.

Compared against BLINK's own published bi-encoder-only Recall@64 on its
"Zero-shot EL" (Zeshel) test set: **82.06%** (BERT-base, Table 1). This is
the one dataset BLINK's own paper reports a genuine bi-encoder-only number
for -- unlike AIDA-CoNLL (no BLINK numbers exist at all), TAC-KBP2010
(LDC-licensed, not freely downloadable) or WikilinksNED Unseen-Mentions
(no stable public host). See zeshel.py's and blink.py's module
docstrings for the full history of this design decision (issues #16/#46).

The `naist-nlp/zeshel` mirror's split sizes don't exactly match the
paper's original 49K/10K/10K mention counts (the original release is
Google-Drive-hosted, not automatable), so treat this comparison as
in-spirit, not a byte-identical reproduction.

Downloads ~2.4GB total over the network the first time it's run (the
mentions config plus the full entity dictionary, cached by the
`datasets` library after that). Also downloads `distilbert-base-uncased`
(~260MB, x2 -- one per tower) via `transformers` the first time.

Run with: `uv run python examples/blink_benchmark.py`
"""

from __future__ import annotations

from collections import defaultdict

import torch

from linkingtk.algorithms.el.blink import BlinkLinker
from linkingtk.blocking.exact import ExactMatch
from linkingtk.core.entity import Entity
from linkingtk.datasets.zeshel import ZeshelDataset
from linkingtk.eval import Evaluator
from linkingtk.train.arguments import TrainingArguments
from linkingtk.train.trainer import Trainer


def _group_by_domain(entities: list[Entity]) -> dict[str, list[Entity]]:
    grouped: dict[str, list[Entity]] = defaultdict(list)
    for entity in entities:
        grouped[entity.properties["domain"]].append(entity)
    return grouped


def main() -> None:
    dataset = ZeshelDataset()
    mentions, kb, _ground_truth = dataset.load()
    train_pairs, test_pairs, _val_pairs = dataset.load_splits()

    mentions_by_id = {entity.id: entity for entity in mentions}
    kb_by_id = {entity.id: entity for entity in kb}
    train_data = [(mentions_by_id[m], kb_by_id[e]) for m, e in train_pairs]

    print(f"{len(train_data)} train mentions / {len(test_pairs)} test mentions")

    # Hyperparameters match the paper's own bi-encoder-base config for this
    # exact dataset (Appendix A.2, Table 10/"Zero-shot Entity Linking
    # Dataset"): lr=2e-5, batch_size=128, max_length=128, 5 epochs -- not
    # ReFinED's defaults, which were never validated against BLINK's own
    # numbers. Swapping in bert-base-uncased (matching Table 11's 220M-param
    # "Bi-encoder (base)" -- two full BERT-base towers, not two smaller
    # DistilBERT ones) made no measurable difference (Hits@64 0.7403 vs.
    # 0.7416), so distilbert-base-uncased is kept for its lower compute cost.
    linker = BlinkLinker(
        mention_model_name="distilbert-base-uncased", embedding_dim=256, max_length=128
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = TrainingArguments(
        output_dir="./models/blink_zeshel",
        learning_rate=2e-5,
        num_epochs=5,
        batch_size=128,
        negative_samples_ratio=4,
        loss="infonce",
        device=device,
    )
    # Only ~4.6% of Zeshel train mentions' surface text exactly matches their
    # gold entity's title (natural referring expressions, not names-as-
    # mentions like AIDA-CoNLL, ExactMatch's usual home), so ExactMatch mines
    # essentially no hard negatives here. Swapping in LabelOverlap(ngram_size=3,
    # max_matches=10) to actually find some was tried -- and made results
    # *worse* (Hits@64 0.699 vs. 0.740, at a negative_samples_ratio halved to
    # 2 to fit the resulting larger per-step batch in GPU memory), not better,
    # so ExactMatch is kept. In-batch negatives (127 per step at batch_size=128)
    # are apparently already carrying the useful contrastive signal here.
    Trainer(model=linker.encoder, args=args, train_data=train_data, blocking=ExactMatch()).train()

    test_source_ids = {m for m, _ in test_pairs}
    test_mentions_by_domain = _group_by_domain([e for e in mentions if e.id in test_source_ids])
    test_kb_by_domain = _group_by_domain(kb)

    linker.encoder.to(device)
    linker.encoder.eval()
    all_ranked_predictions: list[tuple[str, list[str]]] = []
    with torch.no_grad():
        for domain, domain_mentions in test_mentions_by_domain.items():
            domain_kb = test_kb_by_domain.get(domain, [])
            if not domain_kb:
                continue
            mention_emb = linker.encoder.encode(domain_mentions)
            kb_emb = linker.encoder.encode(domain_kb)
            similarities = mention_emb @ kb_emb.T
            order = torch.argsort(similarities, dim=1, descending=True)
            all_ranked_predictions.extend(
                (mention.id, [domain_kb[j].id for j in row.tolist()])
                for mention, row in zip(domain_mentions, order, strict=True)
            )

    report = Evaluator.evaluate_ranked(
        all_ranked_predictions, ground_truth=test_pairs, top_k=[1, 10, 64]
    )
    print(f"Metrics: {report.metrics}")
    print("Reference: BLINK's own published bi-encoder Recall@64 on Zeshel test is 82.06%")
    print(
        "(naist-nlp/zeshel mirror's split isn't byte-identical to the paper's -- "
        "in-spirit comparison)"
    )


if __name__ == "__main__":
    main()

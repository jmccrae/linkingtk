"""Benchmark-methodology test: BlinkLinker's bi-encoder trained on a
Zeshel-shaped train/test domain split, evaluated with
linkingtk.eval.Evaluator.evaluate_ranked (Hits@1, Hits@10, MRR) via
exhaustive (not blocking-restricted) per-domain ranking -- matching how
BLINK's own paper measures bi-encoder Recall@k (Table 1): the gold entity
ranked among *all* of its domain's KB entities, not a candidate-restricted
subset.

Uses a tiny local fixture, not the real (33K+ document) Zeshel release --
this checks the *methodology* (native train/test domain split via
ZeshelDataset.load_splits(), exhaustive per-domain ranking, well-formed
metrics), not a literal comparison to BLINK's published numbers. See
examples/blink_benchmark.py for the same methodology at real dataset
scale, compared against the published 82.06% test Recall@64 reference.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from linkingtk.algorithms.el.blink import BlinkLinker  # noqa: E402
from linkingtk.blocking.exact import ExactMatch  # noqa: E402
from linkingtk.datasets.zeshel import ZeshelDataset  # noqa: E402
from linkingtk.eval import Evaluator  # noqa: E402
from linkingtk.train.arguments import TrainingArguments  # noqa: E402
from linkingtk.train.trainer import Trainer  # noqa: E402

_TINY_MODEL = "hf-internal-testing/tiny-random-DistilBertModel"

# Two domains, disjoint between train and test -- the zero-shot protocol
# Zeshel's own split boundaries encode. KB entities overlap in shape (same
# count per domain) but not in identity, same as the real dataset.
_MENTION_ROWS_BY_SPLIT = {
    "train": [
        {
            "subset": "trainworld",
            "id": "doc1",
            "text": "Paris is the capital of France. Berlin is the capital of Germany.",
            "entities": [
                {"start": 0, "end": 5, "label": ["Q90"]},
                {"start": 33, "end": 39, "label": ["Q3787"]},
            ],
        }
    ],
    "validation": [
        {
            "subset": "trainworld",
            "id": "doc2",
            "text": "Rome is the capital of Italy.",
            "entities": [{"start": 0, "end": 4, "label": ["Q220"]}],
        }
    ],
    "test": [
        {
            "subset": "testworld",
            "id": "doc3",
            "text": "I love Madrid in the springtime. Lisbon has great museums.",
            "entities": [
                {"start": 7, "end": 13, "label": ["Q2807"]},
                {"start": 34, "end": 40, "label": ["Q597"]},
            ],
        }
    ],
}

_KB_ROWS = [
    {"id": "Q90", "subset": "trainworld", "name": "Paris", "description": "capital of France"},
    {"id": "Q3787", "subset": "trainworld", "name": "Berlin", "description": "capital of Germany"},
    {"id": "Q220", "subset": "trainworld", "name": "Rome", "description": "capital of Italy"},
    {"id": "Q2807", "subset": "testworld", "name": "Madrid", "description": "capital of Spain"},
    {"id": "Q597", "subset": "testworld", "name": "Lisbon", "description": "capital of Portugal"},
]


@pytest.fixture
def dataset(tmp_path: Path) -> ZeshelDataset:
    mentions_dir = tmp_path / "mentions"
    mentions_dir.mkdir()
    for split, rows in _MENTION_ROWS_BY_SPLIT.items():
        pd.DataFrame(rows).to_parquet(mentions_dir / f"{split}.parquet")

    kb_dir = tmp_path / "kb"
    kb_dir.mkdir()
    pd.DataFrame(_KB_ROWS).to_parquet(kb_dir / "kb.parquet")

    return ZeshelDataset(mentions_source=str(mentions_dir), kb_source=str(kb_dir))


def test_blink_linker_reports_ranked_metrics_on_held_out_domain(
    dataset: ZeshelDataset, tmp_path: Path
) -> None:
    mentions, kb, _ground_truth = dataset.load()
    train_pairs, test_pairs, _val_pairs = dataset.load_splits()

    mentions_by_id = {entity.id: entity for entity in mentions}
    kb_by_id = {entity.id: entity for entity in kb}
    train_data = [(mentions_by_id[m], kb_by_id[e]) for m, e in train_pairs]

    linker = BlinkLinker(mention_model_name=_TINY_MODEL, embedding_dim=16, max_length=32)
    args = TrainingArguments(
        output_dir=str(tmp_path / "model"),
        learning_rate=5e-4,
        num_epochs=30,
        batch_size=8,
        negative_samples_ratio=1,
    )
    Trainer(model=linker.encoder, args=args, train_data=train_data, blocking=ExactMatch()).train()

    # Exhaustive per-domain ranking: the test mentions' own domain's KB
    # entities only -- mirrors BLINK's Recall@k methodology, not a
    # candidate-restricted eval.
    test_source_ids = {m for m, _ in test_pairs}
    test_mentions = [entity for entity in mentions if entity.id in test_source_ids]
    test_domains = {entity.properties["domain"] for entity in test_mentions}
    test_kb = [entity for entity in kb if entity.properties["domain"] in test_domains]

    linker.encoder.eval()
    with torch.no_grad():
        mention_emb = linker.encoder.encode(test_mentions)
        kb_emb = linker.encoder.encode(test_kb)
    similarities = mention_emb @ kb_emb.T
    ranked_predictions = [
        (mention.id, [test_kb[j].id for j in order.tolist()])
        for mention, order in zip(
            test_mentions, torch.argsort(similarities, dim=1, descending=True), strict=True
        )
    ]
    report = Evaluator.evaluate_ranked(ranked_predictions, ground_truth=test_pairs, top_k=[1, 2])

    assert set(report.metrics) == {"Hits@1", "Hits@2", "MRR"}
    for value in report.metrics.values():
        assert 0.0 <= value <= 1.0
    assert report.metrics["Hits@2"] >= report.metrics["Hits@1"]
    assert report.metrics["MRR"] >= report.metrics["Hits@1"]

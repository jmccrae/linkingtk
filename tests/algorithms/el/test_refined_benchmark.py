"""Benchmark-methodology test: ReFinEDLinker trained on an AIDA-CoNLL-shaped
train/test split, evaluated with linkingtk.eval.Evaluator.evaluate_ranked
(Hits@1, Hits@10, MRR) via exhaustive (not blocking-restricted) ranking --
matching the #37/#38 lesson that blocking-restricted eval upper-bounds by
blocking's own recall, not embedding quality.

Uses a tiny local fixture, not the real (946+216+231 document) AIDA-CoNLL
dataset -- this checks the *methodology* (native train/test split via
AidaConllDataset.load_splits(), exhaustive ranking, well-formed metrics),
not a literal comparison to ReFinED's published numbers. See
examples/refined_benchmark.py for the same methodology at real dataset
scale, compared against the published ~84.6 F1 reference (see that
script's and refined.py's module docstrings for why that's the honest
target).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from linkingtk.algorithms.el.refined import ReFinEDLinker  # noqa: E402
from linkingtk.blocking.exact import ExactMatch  # noqa: E402
from linkingtk.datasets.aida_conll import AidaConllDataset  # noqa: E402
from linkingtk.eval import Evaluator  # noqa: E402
from linkingtk.train.arguments import TrainingArguments  # noqa: E402
from linkingtk.train.trainer import Trainer  # noqa: E402

_TINY_MODEL = "hf-internal-testing/tiny-random-DistilBertModel"

# Two documents per split; train and test share no mentions but the KB
# (unique QIDs) overlaps across splits, same shape as the real dataset --
# generalization to a mention the model wasn't directly trained on, not
# just memorization of the training mention text.
_ROWS_BY_SPLIT = {
    "train": [
        {
            "document_id": 1,
            "text": "Paris is the capital of France. Berlin is the capital of Germany.",
            "entities": [
                {"start": 0, "end": 5, "tag": "LOC", "pageid": 1, "qid": 90, "title": "Paris"},
                {
                    "start": 33,
                    "end": 39,
                    "tag": "LOC",
                    "pageid": 2,
                    "qid": 3787,
                    "title": "Berlin",
                },
            ],
        }
    ],
    "validation": [
        {
            "document_id": 2,
            "text": "Rome is the capital of Italy.",
            "entities": [
                {"start": 0, "end": 4, "tag": "LOC", "pageid": 3, "qid": 220, "title": "Rome"},
            ],
        }
    ],
    "test": [
        {
            "document_id": 3,
            "text": "I love Paris in the springtime. Berlin has great museums.",
            "entities": [
                {"start": 7, "end": 12, "tag": "LOC", "pageid": 1, "qid": 90, "title": "Paris"},
                {
                    "start": 33,
                    "end": 39,
                    "tag": "LOC",
                    "pageid": 2,
                    "qid": 3787,
                    "title": "Berlin",
                },
            ],
        }
    ],
}


def _fake_fetcher(titles: list[str]) -> dict[str, str]:
    descriptions = {
        "Paris": "capital and most populous city of France",
        "Berlin": "capital and largest city of Germany",
        "Rome": "capital city of Italy",
    }
    return {title: descriptions.get(title, "") for title in titles}


@pytest.fixture
def fixture_source(tmp_path: Path) -> str:
    for split, rows in _ROWS_BY_SPLIT.items():
        pd.DataFrame(rows).to_parquet(tmp_path / f"{split}.parquet")
    return str(tmp_path)


def test_refined_linker_reports_ranked_metrics_on_held_out_split(
    fixture_source: str, tmp_path: Path
) -> None:
    dataset = AidaConllDataset(source=fixture_source, description_fetcher=_fake_fetcher)
    mentions, kb, _ground_truth = dataset.load()
    train_pairs, test_pairs, _val_pairs = dataset.load_splits()

    mentions_by_id = {entity.id: entity for entity in mentions}
    kb_by_id = {entity.id: entity for entity in kb}
    train_data = [(mentions_by_id[m], kb_by_id[e]) for m, e in train_pairs]

    linker = ReFinEDLinker(model_name=_TINY_MODEL, embedding_dim=16, max_length=32)
    args = TrainingArguments(
        output_dir=str(tmp_path / "model"),
        learning_rate=5e-4,
        num_epochs=30,
        batch_size=8,
        negative_samples_ratio=1,
    )
    Trainer(model=linker.encoder, args=args, train_data=train_data, blocking=ExactMatch()).train()

    test_source_ids = {m for m, _ in test_pairs}
    test_target_ids = {e for _, e in test_pairs}
    test_mentions = [entity for entity in mentions if entity.id in test_source_ids]
    test_kb = [kb_by_id[entity_id] for entity_id in test_target_ids]

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
    report = Evaluator.evaluate_ranked(ranked_predictions, ground_truth=test_pairs, top_k=[1, 10])

    assert set(report.metrics) == {"Hits@1", "Hits@10", "MRR"}
    for value in report.metrics.values():
        assert 0.0 <= value <= 1.0
    assert report.metrics["Hits@10"] >= report.metrics["Hits@1"]
    assert report.metrics["MRR"] >= report.metrics["Hits@1"]

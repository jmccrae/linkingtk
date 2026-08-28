"""Benchmarks LlmRerankerLinker (#23) layered on top of GlossBertLinker's own
cross-encoder scoring, using the paper's published checkpoint -- same setup
as `examples/glossbert_reproduction.py`.

**Acceptance bar is relative, not an external paper's number.** Unlike #22
(ChatEA), #23 doesn't port a specific paper -- there's nothing published to
chase. The question this script answers: does re-ranking GlossBERT's own
top-k candidate senses with an LLM improve precision@1 over GlossBERT's
own already-benchmarked checkpoint-reproduction baseline (ALL=76.7%, per
`glossbert_reproduction.py`/#39)?

First reproduces the full published table exactly like
`glossbert_reproduction.py` (no LLM calls, no cost). Then, on the "ALL"
split, samples a random subset of mentions whose true sense already falls
within GlossBertLinker's own top-k candidates (`--sample`, default 30 --
matching `examples/chatea_ea.py`'s established precedent for capping
real-LLM-call benchmarks) and re-ranks just that window with
`LlmRerankerLinker`, reporting before/after precision@1.

Setup: identical to `glossbert_reproduction.py` -- see that script's
docstring for the checkpoint/UFSAC download steps. Also requires a local
Ollama server with `ollama pull llama2:13b` (or pass `--model`).

Run with: `uv run python examples/glossbert_llm_reranker_benchmark.py`
"""

from __future__ import annotations

import argparse
import random
import zipfile
from pathlib import Path

import torch

from linkingtk.algorithms.llm_reranker import LlmRerankerLinker
from linkingtk.algorithms.wsd.glossbert import GlossBertEncoder, GlossBertLinker
from linkingtk.blocking.exact import ExactMatch
from linkingtk.datasets.ufsac import UfsacDataset
from linkingtk.eval import Evaluator
from linkingtk.llm.client import create_client

_CHECKPOINT_ZIP = Path.home() / "Downloads" / "Sent_CLS_WS.zip"
_CHECKPOINT_DIR = Path.home() / ".cache" / "linkingtk" / "glossbert_checkpoint"
_UFSAC_DIR = Path.home() / "data" / "ufsac-public-2.1"

# (UFSAC filename stem, README column label, README's "this checkpoint" score)
_EVAL_SETS = [
    ("raganato_semeval2007", "SE07", 72.1),
    ("raganato_senseval2", "SE2", 77.7),
    ("raganato_senseval3", "SE3", 75.9),
    ("raganato_semeval2013", "SE13", 76.8),
    ("raganato_semeval2015", "SE15", 79.3),
    ("raganato_ALL", "ALL", 77.2),
]


def _ensure_checkpoint_extracted() -> Path:
    if not (_CHECKPOINT_DIR / "pytorch_model.bin").exists():
        _CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(_CHECKPOINT_ZIP) as archive:
            archive.extractall(_CHECKPOINT_DIR)
    return _CHECKPOINT_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="ollama/llama2:13b")
    parser.add_argument("--sample", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--random-seed", type=int, default=20260827)
    args = parser.parse_args()

    checkpoint_dir = _ensure_checkpoint_extracted()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    encoder = GlossBertEncoder.from_checkpoint(checkpoint_dir, max_length=512)
    linker = GlossBertLinker(model=encoder)
    linker.model.to(device)
    blocking = ExactMatch(top_k=50)

    print(f"{'dataset':<8} {'precision@1':>12} {'published':>10}")
    all_mentions = all_senses = all_ground_truth = None
    for stem, label, published in _EVAL_SETS:
        mentions, senses, ground_truth = UfsacDataset(source=str(_UFSAC_DIR / f"{stem}.xml")).load()

        results = linker.link(mentions, senses, blocking=blocking)
        predictions = [(result.source_id, result.target_id) for result in results]
        report = Evaluator.evaluate(predictions=predictions, ground_truth=ground_truth)
        score = report.metrics["precision@1"] * 100

        print(f"{label:<8} {score:>11.1f}% {published:>9.1f}%")
        if label == "ALL":
            all_mentions, all_senses, all_ground_truth = mentions, senses, ground_truth

    assert all_mentions is not None and all_senses is not None and all_ground_truth is not None
    mentions_by_id = {entity.id: entity for entity in all_mentions}
    true_target_by_source = dict(all_ground_truth)

    base_results = linker.link(all_mentions, all_senses, blocking=blocking)
    base_prediction_by_source = {result.source_id: result.target_id for result in base_results}
    base_topk_by_source = {
        result.source_id: [result.target_id, *result.alternatives][: args.top_k]
        for result in base_results
    }

    improvable_ids = [
        source_id
        for source_id, top_k_ids in base_topk_by_source.items()
        if source_id in true_target_by_source and true_target_by_source[source_id] in top_k_ids
    ]
    print(
        f"Improvable (true sense within top-{args.top_k}): "
        f"{len(improvable_ids)}/{len(base_topk_by_source)}"
    )

    rng = random.Random(args.random_seed)
    sampled_ids = set(rng.sample(improvable_ids, min(args.sample, len(improvable_ids))))
    print(f"Sampled for real LLM re-ranking: {len(sampled_ids)}")

    client = create_client(args.model)
    reranker = LlmRerankerLinker(base_linker=linker, client=client, task="wsd", top_k=args.top_k)
    sampled_mentions = [mentions_by_id[source_id] for source_id in sampled_ids]
    llm_results = reranker.link(sampled_mentions, all_senses, blocking=blocking)

    merged_prediction_by_source = dict(base_prediction_by_source)
    changed_correctness = 0
    for result in llm_results:
        source_id = result.source_id
        true_target = true_target_by_source[source_id]
        was_correct = base_prediction_by_source.get(source_id) == true_target
        merged_prediction_by_source[source_id] = result.target_id
        now_correct = result.target_id == true_target
        if was_correct != now_correct:
            changed_correctness += 1
        print(f"  {source_id}: base_correct={was_correct} -> llm_correct={now_correct}")
    print(
        f"Sampled entities whose correctness changed after LLM "
        f"re-ranking: {changed_correctness}/{len(llm_results)}"
    )

    final_report = Evaluator.evaluate(
        predictions=list(merged_prediction_by_source.items()), ground_truth=all_ground_truth
    )
    final_score = final_report.metrics["precision@1"] * 100
    print(f"After LlmRerankerLinker re-ranking (ALL): {final_score:.1f}%")


if __name__ == "__main__":
    main()

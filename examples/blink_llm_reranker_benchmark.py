"""Benchmarks LlmRerankerLinker (#23) layered on top of BlinkLinker's own
bi-encoder retrieval, on the same Zeshel setup as `examples/blink_benchmark.py`.

**Acceptance bar is relative, not an external paper's number.** Unlike
#22 (ChatEA), #23 doesn't port a specific paper -- there's nothing
published to chase. The question this script answers is simply: does
re-ranking BLINK's own top-k candidates with an LLM improve Hits@1 over
BLINK's own already-benchmarked bi-encoder baseline (Hits@64=0.724 on
this exact Zeshel setup, per `blink_benchmark.py`/#46)? Hits@64 itself
should be unchanged -- the reranker only ever reorders within the top-k
window it's shown, never reaches a candidate ranked below it (same
ceiling argument as #22's `_merge_llm_rerank`).

Trains `BlinkLinker` identically to `blink_benchmark.py` (see that
script's docstring for the hyperparameter provenance), then for a random
sample of test mentions whose true entity already falls within the top-k
window (`--sample`, default 30 -- matching `examples/chatea_ea.py`'s own
established precedent for capping real-LLM-call benchmarks), re-ranks
that window with `LlmRerankerLinker` and reports before/after Hits@1/10/64.

Requires a local Ollama server with `ollama pull llama2:13b` (or pass
`--model` for a different one). Downloads/compute cost matches
`blink_benchmark.py`'s own (~2.4GB Zeshel + 2x distilbert-base-uncased).

Run with: `uv run python examples/blink_llm_reranker_benchmark.py`
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from collections.abc import Iterable

import torch

from linkingtk.algorithms.base import DEFAULT_BLOCKING
from linkingtk.algorithms.el.blink import BlinkLinker
from linkingtk.algorithms.llm_reranker import LlmRerankerLinker
from linkingtk.blocking.base import BlockingStrategy
from linkingtk.blocking.exact import ExactMatch
from linkingtk.core.entity import Entity
from linkingtk.core.source import EntitySource
from linkingtk.datasets.zeshel import ZeshelDataset
from linkingtk.eval import Evaluator
from linkingtk.llm.client import create_client
from linkingtk.train.arguments import TrainingArguments
from linkingtk.train.trainer import Trainer

_RERANK_TOP_K = 20


def _group_by_domain(entities: list[Entity]) -> dict[str, list[Entity]]:
    grouped: dict[str, list[Entity]] = defaultdict(list)
    for entity in entities:
        grouped[entity.properties["domain"]].append(entity)
    return grouped


def _merge_llm_rerank(base_ranking: list[str], top_k: int, llm_order: list[str]) -> list[str]:
    """Replaces `base_ranking`'s top-`top_k` prefix with `llm_order`, leaving
    everything below `top_k` untouched -- the reranker only ever reorders
    within its own candidate window, never reaches beyond it."""
    prefix_ids = set(base_ranking[:top_k])
    reordered = [target_id for target_id in llm_order if target_id in prefix_ids]
    reordered_set = set(reordered)
    missing = [target_id for target_id in base_ranking[:top_k] if target_id not in reordered_set]
    return reordered + missing + base_ranking[top_k:]


class _PrecomputedScorer:
    """A `CandidateScorer` returning BLINK's own real top-k scores, computed
    once up front from the same dense per-domain similarity matrix
    `blink_benchmark.py` itself evaluates on -- avoids re-encoding through
    `BlinkEncoder` a second time just to satisfy `LlmRerankerLinker`'s
    interface."""

    def __init__(self, scores_by_source: dict[str, list[tuple[str, float]]]) -> None:
        self._scores_by_source = scores_by_source

    def score_candidates(
        self,
        dataset1: list[Entity],
        dataset2: list[Entity] | EntitySource,
        blocking: BlockingStrategy = DEFAULT_BLOCKING,
    ) -> dict[str, list[tuple[str, float]]]:
        return {
            entity.id: self._scores_by_source[entity.id]
            for entity in dataset1
            if entity.id in self._scores_by_source
        }


class _PrecomputedBlocking(BlockingStrategy):
    """Resolves the same precomputed top-k ids back to real `Entity` objects
    -- what `LlmRerankerLinker.link()` needs to render candidates in its
    LLM prompt, without re-running blocking/retrieval from scratch."""

    def __init__(
        self, ranked_ids_by_source: dict[str, list[str]], entities_by_id: dict[str, Entity]
    ) -> None:
        self._ranked_ids_by_source = ranked_ids_by_source
        self._entities_by_id = entities_by_id

    def candidate_pairs(
        self, dataset1: list[Entity], dataset2: list[Entity] | EntitySource
    ) -> Iterable[tuple[Entity, Entity]]:
        return [
            (entity, self._entities_by_id[target_id])
            for entity in dataset1
            for target_id in self._ranked_ids_by_source.get(entity.id, [])
            if target_id in self._entities_by_id
        ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="ollama/llama2:13b")
    parser.add_argument("--sample", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--random-seed", type=int, default=20260827)
    args = parser.parse_args()

    dataset = ZeshelDataset()
    mentions, kb, _ground_truth = dataset.load()
    train_pairs, test_pairs, _val_pairs = dataset.load_splits()

    mentions_by_id = {entity.id: entity for entity in mentions}
    kb_by_id = {entity.id: entity for entity in kb}
    train_data = [(mentions_by_id[m], kb_by_id[e]) for m, e in train_pairs]

    print(f"{len(train_data)} train mentions / {len(test_pairs)} test mentions")

    # Same hyperparameters as blink_benchmark.py -- see that script's own
    # docstring for their provenance against BLINK's paper.
    linker = BlinkLinker(
        mention_model_name="distilbert-base-uncased", embedding_dim=256, max_length=128
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_args = TrainingArguments(
        output_dir="./models/blink_zeshel_reranker",
        learning_rate=2e-5,
        num_epochs=5,
        batch_size=128,
        negative_samples_ratio=4,
        loss="infonce",
        device=device,
    )
    Trainer(
        model=linker.encoder, args=train_args, train_data=train_data, blocking=ExactMatch()
    ).train()

    test_source_ids = {m for m, _ in test_pairs}
    test_mentions_by_domain = _group_by_domain([e for e in mentions if e.id in test_source_ids])
    test_kb_by_domain = _group_by_domain(kb)

    linker.encoder.to(device)
    linker.encoder.eval()
    all_ranked_predictions: list[tuple[str, list[str]]] = []
    top_k_scores_by_source: dict[str, list[tuple[str, float]]] = {}
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
            for row_index, (mention, row) in enumerate(zip(domain_mentions, order, strict=True)):
                top_rows = row[:_RERANK_TOP_K].tolist()
                top_k_scores_by_source[mention.id] = [
                    (domain_kb[j].id, float(similarities[row_index, j])) for j in top_rows
                ]

    base_report = Evaluator.evaluate_ranked(
        all_ranked_predictions, ground_truth=test_pairs, top_k=[1, 10, 64]
    )
    print(f"Base (BlinkLinker): {base_report.metrics}")
    print("Reference: BLINK's own published bi-encoder Recall@64 on Zeshel test is 82.06%")

    base_ranking = dict(all_ranked_predictions)
    true_target_by_source = dict(test_pairs)
    improvable_ids = [
        source_id
        for source_id, ranked in base_ranking.items()
        if source_id in true_target_by_source
        and true_target_by_source[source_id] in ranked[:_RERANK_TOP_K]
    ]
    print(
        f"Improvable (true target within top-{_RERANK_TOP_K}): "
        f"{len(improvable_ids)}/{len(test_pairs)}"
    )

    rng = random.Random(args.random_seed)
    sampled_ids = set(rng.sample(improvable_ids, min(args.sample, len(improvable_ids))))
    print(f"Sampled for real LLM re-ranking: {len(sampled_ids)}")

    ranked_ids_by_source = {
        source_id: [target_id for target_id, _ in scored]
        for source_id, scored in top_k_scores_by_source.items()
    }
    entities_by_id = {
        target_id: kb_by_id[target_id]
        for scored in top_k_scores_by_source.values()
        for target_id, _ in scored
    }

    client = create_client(args.model)
    reranker = LlmRerankerLinker(
        base_linker=_PrecomputedScorer(top_k_scores_by_source),
        client=client,
        task="el",
        top_k=args.top_k,
        # BLINK's score is a cosine similarity of L2-normalized vectors,
        # roughly in [-1, 1] -- a 0.5 gap is a meaningful, safe confidence
        # threshold here. Not a default on LlmRerankerLinker itself, since
        # this doesn't hold for every base linker (see its own docstring).
        threshold=0.5,
    )
    sampled_mentions = [mentions_by_id[source_id] for source_id in sampled_ids]
    llm_results = reranker.link(
        sampled_mentions, kb, blocking=_PrecomputedBlocking(ranked_ids_by_source, entities_by_id)
    )

    merged_ranking: dict[str, list[str]] = dict(base_ranking)
    changed_correctness = 0
    for result in llm_results:
        source_id = result.source_id
        true_target = true_target_by_source[source_id]
        llm_order = [result.target_id, *result.alternatives]
        merged_ranking[source_id] = _merge_llm_rerank(
            base_ranking[source_id], args.top_k, llm_order
        )
        was_correct_at_1 = base_ranking[source_id][0] == true_target
        now_correct_at_1 = merged_ranking[source_id][0] == true_target
        if was_correct_at_1 != now_correct_at_1:
            changed_correctness += 1
        print(
            f"  {source_id}: base_rank@1_correct={was_correct_at_1} "
            f"-> llm_rank@1_correct={now_correct_at_1}"
        )
    print(
        f"Sampled entities whose Hits@1 correctness changed after LLM "
        f"re-ranking: {changed_correctness}/{len(llm_results)}"
    )

    final_report = Evaluator.evaluate_ranked(
        list(merged_ranking.items()), ground_truth=test_pairs, top_k=[1, 10, 64]
    )
    print("After LlmRerankerLinker re-ranking (full test set):", final_report.metrics)


if __name__ == "__main__":
    main()

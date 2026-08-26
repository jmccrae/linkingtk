"""Benchmarks ChatEALinker (#22) on ICEWS-WIKI, layered on top of the
finalized SimpleHHEALinker(use_structure=False) base embedding from #62.

**Acceptance bar is relative, not the paper's absolute published number.**
SimpleHHEALinker(use_structure=False) alone already scores Hits@1=0.8147
on ICEWS-WIKI (see docs/examples/simple_hhea_ea.md) -- comfortably
beating not just the paper's own Simple-HHEA row (0.720) but *also* its
full ChatEA+llama2-13b composite number (Hits@1=0.455, Table 4 of
https://aclanthology.org/2024.acl-long.408.pdf). Since our base embedding
is already stronger than theirs, matching or missing their composite
number would mostly reflect that base-quality delta, not whether ChatEA's
LLM-reranking layer itself helps. This script's real comparison is
against our own base-only number, computed here with the exact same
ranking metric (plain cosine, no CSLS -- matching what ChatEALinker
itself uses internally) so the "before"/"after" numbers are a clean
apples-to-apples ablation. The paper's Table 4 number, and #62's own
CSLS-based 0.8147/0.8822/0.8384, are also printed for context only.

Method: for every test entity, first rank all test targets by
`SimpleHHEALinker`'s own plain-cosine similarity (identical formula
`ChatEALinker` uses for its own top-k candidate generation, via
`rank_exhaustive(metric="cosine", csls_k=0)`). An entity whose true
target already falls outside the top-`top_k` window can never be fixed
by re-ranking *within* that window -- ChatEA only ever reorders its own
top-k candidates, never reaches beyond them (see
`linkingtk.algorithms.ea._chatea_reasoning`'s module docstring) -- so
only "improvable" entities (true target inside the top-k window) are
worth spending real LLM calls on; this mirrors the reference's own
`base_rank >= 20` evaluation-cost shortcut (`eval_alignment_for_evaluate`),
reproduced here at the benchmark-script layer specifically because this
script legitimately has ground truth to make that call, unlike
`ChatEALinker.link()`'s own general-purpose interface.

A configurable random sample of the improvable entities (`--sample`,
default 30) is then actually sent through `ChatEALinker.link()` for real
LLM re-ranking -- matching this repo's own established precedent
(`examples/llm_benchmark.py`'s `--max-mentions`) of capping expensive
real-LLM-call benchmarks. Each sampled entity's top-`top_k` window in the
full ranking is replaced by ChatEALinker's own re-ranked order; everything
else (non-sampled entities, and every entity's ranking below top_k) is
left untouched, so Hits@1/5/10 + MRR are computed the same way,
before and after, over the *same* held-out test split.

Requires the `kge` optional dependency group (SimpleHHEALinker's own
deps) and a local Ollama server with the target model pulled (default
`ollama pull llama2:13b` -- chosen specifically because the paper's own
comparison point is its published llama2-13b number). Pass e.g.
`--model anthropic/claude-opus-4-6` to benchmark a frontier model
instead.

Run with: `uv run python examples/chatea_ea.py`
"""

from __future__ import annotations

import argparse
import random

from linkingtk.algorithms.ea import ChatEALinker, SimpleHHEALinker
from linkingtk.core.entity import Entity
from linkingtk.datasets.icews import IcewsWikiDataset
from linkingtk.eval import Evaluator, rank_exhaustive
from linkingtk.llm.client import create_client
from linkingtk.utils.graph import to_triples


def _merge_llm_rerank(base_ranking: list[str], top_k: int, llm_order: list[str]) -> list[str]:
    """Replaces `base_ranking`'s top-`top_k` prefix with `llm_order`, leaving
    everything below `top_k` untouched -- ChatEA only ever reorders within
    its own candidate window, never reaches beyond it."""
    prefix_ids = set(base_ranking[:top_k])
    reordered = [target_id for target_id in llm_order if target_id in prefix_ids]
    reordered_set = set(reordered)
    missing = [target_id for target_id in base_ranking[:top_k] if target_id not in reordered_set]
    return reordered + missing + base_ranking[top_k:]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="ollama/llama2:13b")
    parser.add_argument("--sample", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--random-seed", type=int, default=20260826)
    parser.add_argument(
        "--desc-max-tokens",
        type=int,
        default=150,
        help=(
            "Forwarded to ChatEALinker's description-generation calls. Higher "
            "than the reference's own 80 -- measured directly against "
            "llama2:13b via Ollama: 80 truncated the model's structured JSON "
            "output mid-string often enough (34/~100 description calls in an "
            "initial run) to be a real, not edge-case, failure rate."
        ),
    )
    args = parser.parse_args()

    dataset = IcewsWikiDataset()
    entities1, entities2, _ = dataset.load()
    train_pairs, test_pairs, _val_pairs = dataset.load_splits()
    graph1, graph2 = dataset.load_graphs()
    graph = to_triples(graph1) + to_triples(graph2)
    temporal1, temporal2 = dataset.load_temporal_graphs()
    relation_names = dataset.load_relation_labels()

    base_linker = SimpleHHEALinker(device="cuda", use_structure=False)
    base_linker.fit(
        entities1,
        entities2,
        ground_truth=train_pairs,
        graph=graph,
        temporal_triples=temporal1 + temporal2,
        random_state=0,
    )

    test_source_ids = {s for s, _ in test_pairs}
    test_target_ids = {t for _, t in test_pairs}
    test_entities1: list[Entity] = [e for e in entities1 if e.id in test_source_ids]
    test_entities2: list[Entity] = [e for e in entities2 if e.id in test_target_ids]

    base_ranking = dict(
        rank_exhaustive(base_linker, test_entities1, test_entities2, metric="cosine", csls_k=0)
    )
    base_report = Evaluator.evaluate_ranked(
        list(base_ranking.items()), ground_truth=test_pairs, top_k=[1, 5, 10]
    )
    print(f"{len(train_pairs)} train / {len(test_pairs)} test pairs")
    print("Base (SimpleHHEALinker, plain cosine, no CSLS):", base_report.metrics)
    print(
        "Base (SimpleHHEALinker, CSLS -- #62's own closing number, context only): "
        "Hits@1=0.8147, Hits@10=0.8822, MRR=0.8384"
    )
    print(
        "Paper's published ChatEA+llama2-13b on ICEWS-WIKI (Table 4, context only): "
        "Hits@1=0.455, MRR=0.553"
    )

    true_target_by_source = dict(test_pairs)
    improvable_ids = [
        source_id
        for source_id, ranked in base_ranking.items()
        if true_target_by_source[source_id] in ranked[: args.top_k]
    ]
    print(
        f"Improvable (true target within top-{args.top_k}): {len(improvable_ids)}/{len(test_pairs)}"
    )

    rng = random.Random(args.random_seed)
    sampled_ids = set(rng.sample(improvable_ids, min(args.sample, len(improvable_ids))))
    print(f"Sampled for real LLM re-ranking: {len(sampled_ids)}")

    client = create_client(args.model)
    chatea_linker = ChatEALinker(
        base_linker=base_linker,
        client=client,
        top_k=args.top_k,
        relation_names=relation_names,
        temporal_triples=temporal1 + temporal2,
        desc_max_tokens=args.desc_max_tokens,
    )

    sampled_entities1 = [e for e in test_entities1 if e.id in sampled_ids]
    llm_results = chatea_linker.link(sampled_entities1, test_entities2, graph=graph)

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
        list(merged_ranking.items()), ground_truth=test_pairs, top_k=[1, 5, 10]
    )
    print("After ChatEA re-ranking (full test set):", final_report.metrics)


if __name__ == "__main__":
    main()

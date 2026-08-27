"""Runs one linker per complexity tier -- MVP, Classical ML, KGE (EA only),
Deep Learning and LLM-Oriented -- on the same real dataset per task (EA/EL/
WSD) and prints a single side-by-side table via `linkingtk.eval.harness`
(#24), so cost (wall-clock time, LLM-call count) and accuracy can be
compared across milestones directly.

The comparison harness itself (`BenchmarkRun`/`run_benchmarks`/
`format_table` in `linkingtk.eval.harness`) is generic -- it only times a
caller-supplied fit/link/evaluate callable and tabulates the result. All
the task-specific wiring below is exactly what this repo's other
`examples/*_benchmark.py` scripts already do, reused rather than
reinvented:

- EA uses `IcewsWikiDataset` (real train/test split + graph, per
  `simple_hhea_ea.py`/`chatea_ea.py`) since it's the only real EA dataset
  in this repo that both the KGE and Deep Learning tiers can run against.
  Tier 4/5 hyperparameters match those scripts exactly.
- EL uses `AidaConllDataset` (real native split, per `refined_benchmark.py`).
  There is no KGE row for EL -- that tier is EA-only (relation-triple
  knowledge graph embeddings have no EL analogue in this repo).
- WSD uses UFSAC's SemEval-2007 split (per `glossbert_reproduction.py`/
  `llm_benchmark.py`). No KGE row either. The Deep Learning row loads the
  existing full-SemCor-trained checkpoint
  (`models/glossbert_semcor_full/model.pt`, see
  `examples/glossbert_full_training.py`) rather than retraining, so it's a
  zero-training-cost inference-only row -- deliberately not the tiny
  from-scratch training slice `glossbert_benchmark.py` demonstrates.

Each task's LLM-Oriented row layers an LLM on top of that task's own
already-benchmarked Deep Learning linker (`ChatEALinker` on
`SimpleHHEALinker`, `LlmRerankerLinker` on `ReFinEDLinker`/`GlossBertLinker`)
rather than a bare `LlmBaseLinker` -- matching `chatea_ea.py`/
`*_llm_reranker_benchmark.py`'s own "cheap retrieval, LLM reranks only the
top-k, only on a random sample of improvable sources" pattern to keep real
LLM-call cost bounded (`--sample`, default 30).

Requires: the `kge` optional dependency group (EA); a local AIDA-CoNLL
checkout, fetched automatically (EL); a local UFSAC 2.1 checkout at
`~/data/ufsac-public-2.1/` and the full-SemCor GlossBERT checkpoint at
`./models/glossbert_semcor_full/model.pt`, see
`examples/glossbert_full_training.py` (WSD); a local Ollama server with
`ollama pull llama2:13b` (or pass `--model`).

Run with: `uv run python examples/comparative_benchmark.py`
Or a single, fast task: `uv run python examples/comparative_benchmark.py --tasks el --sample 5`
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import torch

from linkingtk.algorithms.ea import ChatEALinker, EntMatcherLinker, KGELinker, SimpleHHEALinker
from linkingtk.algorithms.el.refined import ReFinEDLinker
from linkingtk.algorithms.feature_classifier import FeatureClassifierLinker
from linkingtk.algorithms.llm_reranker import LlmRerankerLinker
from linkingtk.algorithms.string_similarity import StringSimilarityLinker
from linkingtk.algorithms.wsd import GlossBertLinker, LeskLinker
from linkingtk.blocking import ExactMatch, LabelOverlap
from linkingtk.core.entity import Entity, label_texts
from linkingtk.datasets.aida_conll import AidaConllDataset
from linkingtk.datasets.icews import IcewsWikiDataset
from linkingtk.datasets.ufsac import UfsacDataset
from linkingtk.eval import (
    BenchmarkRun,
    CountingLlmClient,
    EvaluationReport,
    Evaluator,
    format_table,
    rank_exhaustive,
    run_benchmarks,
)
from linkingtk.llm.client import create_client
from linkingtk.sources.wn import WnEntitySource
from linkingtk.train.arguments import TrainingArguments
from linkingtk.train.trainer import Trainer
from linkingtk.utils.graph import to_triples

_GLOSSBERT_CHECKPOINT = Path("./models/glossbert_semcor_full/model.pt")


def _merge_llm_rerank(base_ranking: list[str], top_k: int, llm_order: list[str]) -> list[str]:
    """Replaces `base_ranking`'s top-`top_k` prefix with `llm_order`, leaving
    everything below `top_k` untouched -- a reranker only ever reorders
    within its own candidate window, never reaches beyond it."""
    prefix_ids = set(base_ranking[:top_k])
    reordered = [target_id for target_id in llm_order if target_id in prefix_ids]
    reordered_set = set(reordered)
    missing = [target_id for target_id in base_ranking[:top_k] if target_id not in reordered_set]
    return reordered + missing + base_ranking[top_k:]


def _ea_runs(device: str, model: str, sample: int, seed: int) -> list[BenchmarkRun]:
    dataset = IcewsWikiDataset()
    entities1, entities2, _ground_truth = dataset.load()
    train_pairs, test_pairs, _val_pairs = dataset.load_splits()
    graph1, graph2 = dataset.load_graphs()
    graph = to_triples(graph1) + to_triples(graph2)
    temporal1, temporal2 = dataset.load_temporal_graphs()
    relation_names = dataset.load_relation_labels()

    train_source_ids = {s for s, _ in train_pairs}
    train_target_ids = {t for _, t in train_pairs}
    test_source_ids = {s for s, _ in test_pairs}
    test_target_ids = {t for _, t in test_pairs}
    train_entities1 = [e for e in entities1 if e.id in train_source_ids]
    train_entities2 = [e for e in entities2 if e.id in train_target_ids]
    test_entities1 = [e for e in entities1 if e.id in test_source_ids]
    test_entities2 = [e for e in entities2 if e.id in test_target_ids]

    top_k = 20
    # Populated by the Deep Learning run below and reused by the
    # LLM-Oriented run, exactly like chatea_ea.py's own single-script
    # layering -- ChatEALinker re-ranks the same fitted SimpleHHEALinker's
    # own candidates rather than fitting a second one. Requires the Deep
    # Learning BenchmarkRun to execute first (it does -- see the returned
    # list order below).
    shared: dict[str, Any] = {}

    def mvp() -> EvaluationReport:
        blocking = LabelOverlap(max_matches=10)
        results = StringSimilarityLinker().link(test_entities1, test_entities2, blocking=blocking)
        predictions = [(r.source_id, r.target_id) for r in results]
        return Evaluator.evaluate(predictions=predictions, ground_truth=test_pairs)

    def classical() -> EvaluationReport:
        blocking = LabelOverlap(max_matches=10)
        linker = EntMatcherLinker().fit(
            train_entities1, train_entities2, train_pairs, blocking=blocking, random_state=seed
        )
        results = linker.link(test_entities1, test_entities2, blocking=blocking)
        predictions = [(r.source_id, r.target_id) for r in results]
        return Evaluator.evaluate(predictions=predictions, ground_truth=test_pairs)

    def kge() -> EvaluationReport:
        linker = KGELinker(num_epochs=20)  # TransE, embedding_dim=50
        linker.fit(entities1, entities2, ground_truth=train_pairs, graph=graph, random_state=seed)
        # Unlike OpenEA's EnFr15K (kge_benchmark.py), ICEWS-WIKI has entities
        # with zero graph triples at all (isolated on one side of this
        # "highly heterogeneous" pairing, per simple_hhea_ea.py's own
        # docstring) -- a pure KGE method has no embedding to give them.
        # Excluded from ranking here rather than crashing; still counted as
        # misses by Evaluator.evaluate_ranked (their ground-truth pair has no
        # entry in ranked_predictions), not silently dropped from the
        # denominator.
        graph_ids = {head for head, _relation, _tail in graph} | {
            tail for _head, _relation, tail in graph
        }
        rankable_entities1 = [e for e in test_entities1 if e.id in graph_ids]
        rankable_entities2 = [e for e in test_entities2 if e.id in graph_ids]
        ranked_predictions = rank_exhaustive(linker, rankable_entities1, rankable_entities2)
        return Evaluator.evaluate_ranked(ranked_predictions, ground_truth=test_pairs, top_k=[1, 10])

    def deep_learning() -> EvaluationReport:
        linker = SimpleHHEALinker(device=device, use_structure=False)
        linker.fit(
            entities1,
            entities2,
            ground_truth=train_pairs,
            graph=graph,
            temporal_triples=temporal1 + temporal2,
            random_state=seed,
        )
        shared["linker"] = linker
        shared["base_ranking"] = dict(
            rank_exhaustive(linker, test_entities1, test_entities2, metric="cosine", csls_k=0)
        )
        ranked_predictions = rank_exhaustive(
            linker, test_entities1, test_entities2, metric="cosine", csls_k=10
        )
        return Evaluator.evaluate_ranked(ranked_predictions, ground_truth=test_pairs, top_k=[1, 10])

    client = CountingLlmClient(create_client(model))

    def llm_oriented() -> EvaluationReport:
        base_linker: SimpleHHEALinker = shared["linker"]
        base_ranking: dict[str, list[str]] = shared["base_ranking"]
        true_target_by_source = dict(test_pairs)
        improvable_ids = [
            source_id
            for source_id, ranked in base_ranking.items()
            if true_target_by_source[source_id] in ranked[:top_k]
        ]
        rng = random.Random(seed)
        sampled_ids = set(rng.sample(improvable_ids, min(sample, len(improvable_ids))))

        chatea_linker = ChatEALinker(
            base_linker=base_linker,
            client=client,
            top_k=top_k,
            relation_names=relation_names,
            temporal_triples=temporal1 + temporal2,
        )
        sampled_entities1 = [e for e in test_entities1 if e.id in sampled_ids]
        llm_results = chatea_linker.link(sampled_entities1, test_entities2, graph=graph)

        merged_ranking = dict(base_ranking)
        for result in llm_results:
            llm_order = [result.target_id, *result.alternatives]
            merged_ranking[result.source_id] = _merge_llm_rerank(
                base_ranking[result.source_id], top_k, llm_order
            )
        return Evaluator.evaluate_ranked(
            list(merged_ranking.items()), ground_truth=test_pairs, top_k=[1, 10]
        )

    return [
        BenchmarkRun("EA", "MVP", "StringSimilarityLinker", "ICEWS-WIKI", mvp),
        BenchmarkRun("EA", "Classical ML", "EntMatcherLinker", "ICEWS-WIKI", classical),
        BenchmarkRun("EA", "KGE", "KGELinker (TransE)", "ICEWS-WIKI", kge),
        BenchmarkRun("EA", "Deep Learning", "SimpleHHEALinker", "ICEWS-WIKI", deep_learning),
        BenchmarkRun("EA", "LLM-Oriented", "ChatEALinker", "ICEWS-WIKI", llm_oriented, client),
    ]


def _el_runs(device: str, model: str, sample: int, seed: int) -> list[BenchmarkRun]:
    dataset = AidaConllDataset()
    mentions, kb, _ground_truth = dataset.load()
    train_pairs, test_pairs, _val_pairs = dataset.load_splits()

    mentions_by_id = {entity.id: entity for entity in mentions}
    kb_by_id = {entity.id: entity for entity in kb}
    train_data = [(mentions_by_id[m], kb_by_id[e]) for m, e in train_pairs]
    test_source_ids = {m for m, _ in test_pairs}
    test_target_ids = {e for _, e in test_pairs}
    test_mentions = [entity for entity in mentions if entity.id in test_source_ids]
    test_kb = [kb_by_id[entity_id] for entity_id in test_target_ids]

    # LabelOverlap(max_matches=30) approximates ReFinED's own top-30
    # entity-prior candidate restriction -- see refined_benchmark.py's
    # docstring. Used for every EL tier here so the comparison is over the
    # same candidate space, not just the same dataset.
    blocking = LabelOverlap(ngram_size=3, max_matches=30)
    top_k = 10
    shared: dict[str, Any] = {}

    def mvp() -> EvaluationReport:
        results = StringSimilarityLinker().link(test_mentions, test_kb, blocking=blocking)
        predictions = [(r.source_id, r.target_id) for r in results]
        return Evaluator.evaluate(predictions=predictions, ground_truth=test_pairs)

    def classical() -> EvaluationReport:
        linker = FeatureClassifierLinker().fit(
            [m for m, _e in train_data],
            [e for _m, e in train_data],
            train_pairs,
            blocking=blocking,
            random_state=seed,
        )
        results = linker.link(test_mentions, test_kb, blocking=blocking)
        predictions = [(r.source_id, r.target_id) for r in results]
        return Evaluator.evaluate(predictions=predictions, ground_truth=test_pairs)

    def deep_learning() -> EvaluationReport:
        # Same hyperparameters as refined_benchmark.py -- see that script's
        # docstring for their provenance against ReFinED's own paper.
        linker = ReFinEDLinker(
            model_name="distilbert-base-uncased", embedding_dim=256, max_length=96
        )
        train_args = TrainingArguments(
            output_dir="./models/comparative_benchmark_refined",
            learning_rate=2e-5,
            num_epochs=3,
            batch_size=32,
            negative_samples_ratio=4,
            loss="infonce",
            device=device,
        )
        Trainer(
            model=linker.encoder, args=train_args, train_data=train_data, blocking=ExactMatch()
        ).train()

        results = linker.link(test_mentions, test_kb, blocking=blocking)
        shared["linker"] = linker
        shared["base_results_by_source"] = {r.source_id: r for r in results}
        predictions = [(r.source_id, r.target_id) for r in results]
        return Evaluator.evaluate(predictions=predictions, ground_truth=test_pairs)

    client = CountingLlmClient(create_client(model))

    def llm_oriented() -> EvaluationReport:
        linker = shared["linker"]
        base_results_by_source = shared["base_results_by_source"]
        true_target_by_source = dict(test_pairs)
        improvable_ids = [
            source_id
            for source_id, result in base_results_by_source.items()
            if source_id in true_target_by_source
            and true_target_by_source[source_id] in [result.target_id, *result.alternatives][:top_k]
        ]
        rng = random.Random(seed)
        sampled_ids = set(rng.sample(improvable_ids, min(sample, len(improvable_ids))))

        reranker = LlmRerankerLinker(base_linker=linker, client=client, task="el", top_k=top_k)
        sampled_mentions = [mentions_by_id[source_id] for source_id in sampled_ids]
        llm_results = reranker.link(sampled_mentions, test_kb, blocking=blocking)

        merged_prediction_by_source = {
            source_id: result.target_id for source_id, result in base_results_by_source.items()
        }
        for result in llm_results:
            merged_prediction_by_source[result.source_id] = result.target_id
        return Evaluator.evaluate(
            predictions=list(merged_prediction_by_source.items()), ground_truth=test_pairs
        )

    return [
        BenchmarkRun("EL", "MVP", "StringSimilarityLinker", "AIDA-CoNLL", mvp),
        BenchmarkRun("EL", "Classical ML", "FeatureClassifierLinker", "AIDA-CoNLL", classical),
        BenchmarkRun("EL", "Deep Learning", "ReFinEDLinker", "AIDA-CoNLL", deep_learning),
        BenchmarkRun("EL", "LLM-Oriented", "LlmRerankerLinker", "AIDA-CoNLL", llm_oriented, client),
    ]


def _document_id(mention_id: str) -> str:
    # UfsacDataset's positional mention ids (semcor.xml has no native
    # per-word id) -- "ufsac:{doc}:{sent}:{index}".
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


def _materialize_candidates(mentions: list[Entity], senses: WnEntitySource) -> list[Entity]:
    """Builds a concrete `list[Entity]` candidate pool from a `WnEntitySource`.

    `FeatureClassifierLinker` fits a TF-IDF vectorizer directly over its
    `dataset2`, so (unlike `LeskLinker`/`GlossBertLinker`/`LlmRerankerLinker`,
    which all accept an `EntitySource` directly) it needs a materialized
    list, not a query-driven source. Queries each mention's own lemma
    (`WnEntitySource.search`, same as `senses` itself resolves candidates
    for `ExactMatch`), wide enough (`top_k=30`) that no real lemma's sense
    count gets truncated -- see `glossbert_reproduction.py`'s identical
    `ExactMatch(top_k=50)` choice for the same reason.
    """
    pool: dict[str, Entity] = {}
    for mention in mentions:
        for candidate in senses.search(label_texts(mention)[0], top_k=30):
            pool[candidate.id] = candidate
    return list(pool.values())


def _wsd_runs(
    device: str, model: str, sample: int, seed: int, ufsac_dir: Path
) -> list[BenchmarkRun]:
    eval_mentions, senses, eval_ground_truth = UfsacDataset(
        source=str(ufsac_dir / "raganato_semeval2007.xml")
    ).load()
    blocking = ExactMatch(top_k=50)
    top_k = 10

    # A small SemCor slice for the Classical ML row's `.fit()` -- same
    # 6-document convention as glossbert_benchmark.py's from-scratch
    # training slice.
    all_train_mentions, _senses, all_train_ground_truth = UfsacDataset(
        source=str(ufsac_dir / "semcor.xml")
    ).load()
    train_doc_ids = sorted({_document_id(m.id) for m in all_train_mentions})[:6]
    train_docs = set(train_doc_ids)
    train_gt = [
        (mid, sid) for mid, sid in all_train_ground_truth if _document_id(mid) in train_docs
    ]
    all_train_mentions_by_id = {m.id: m for m in all_train_mentions}
    train_pairs = _resolve_pairs(train_gt, all_train_mentions_by_id, senses)
    train_mentions = list({m.id: m for m, _s in train_pairs}.values())

    shared: dict[str, Any] = {}

    def mvp() -> EvaluationReport:
        results = LeskLinker().link(eval_mentions, senses, blocking=blocking)
        predictions = [(r.source_id, r.target_id) for r in results]
        return Evaluator.evaluate(predictions=predictions, ground_truth=eval_ground_truth)

    def classical() -> EvaluationReport:
        candidate_pool = _materialize_candidates(train_mentions + eval_mentions, senses)
        exact = ExactMatch()
        linker = FeatureClassifierLinker().fit(
            train_mentions, candidate_pool, train_gt, blocking=exact, random_state=seed
        )
        results = linker.link(eval_mentions, candidate_pool, blocking=exact)
        predictions = [(r.source_id, r.target_id) for r in results]
        return Evaluator.evaluate(predictions=predictions, ground_truth=eval_ground_truth)

    def deep_learning() -> EvaluationReport:
        # Loads the existing full-SemCor checkpoint (examples/glossbert_full_training.py)
        # rather than retraining -- zero training cost, real inference cost only.
        linker = GlossBertLinker(model_name_or_path="bert-base-uncased", max_length=512)
        linker.model.load_state_dict(
            torch.load(_GLOSSBERT_CHECKPOINT, map_location=device, weights_only=True)
        )
        linker.model.to(device)
        results = linker.link(eval_mentions, senses, blocking=blocking)
        shared["linker"] = linker
        shared["base_results_by_source"] = {r.source_id: r for r in results}
        predictions = [(r.source_id, r.target_id) for r in results]
        return Evaluator.evaluate(predictions=predictions, ground_truth=eval_ground_truth)

    client = CountingLlmClient(create_client(model))

    def llm_oriented() -> EvaluationReport:
        linker = shared["linker"]
        base_results_by_source = shared["base_results_by_source"]
        true_target_by_source = dict(eval_ground_truth)
        improvable_ids = [
            source_id
            for source_id, result in base_results_by_source.items()
            if source_id in true_target_by_source
            and true_target_by_source[source_id] in [result.target_id, *result.alternatives][:top_k]
        ]
        rng = random.Random(seed)
        sampled_ids = set(rng.sample(improvable_ids, min(sample, len(improvable_ids))))

        mentions_by_id = {m.id: m for m in eval_mentions}
        reranker = LlmRerankerLinker(base_linker=linker, client=client, task="wsd", top_k=top_k)
        sampled_mentions = [mentions_by_id[source_id] for source_id in sampled_ids]
        llm_results = reranker.link(sampled_mentions, senses, blocking=blocking)

        merged_prediction_by_source = {
            source_id: result.target_id for source_id, result in base_results_by_source.items()
        }
        for result in llm_results:
            merged_prediction_by_source[result.source_id] = result.target_id
        return Evaluator.evaluate(
            predictions=list(merged_prediction_by_source.items()), ground_truth=eval_ground_truth
        )

    return [
        BenchmarkRun("WSD", "MVP", "LeskLinker", "SemEval-2007", mvp),
        BenchmarkRun("WSD", "Classical ML", "FeatureClassifierLinker", "SemEval-2007", classical),
        BenchmarkRun("WSD", "Deep Learning", "GlossBertLinker", "SemEval-2007", deep_learning),
        BenchmarkRun(
            "WSD", "LLM-Oriented", "LlmRerankerLinker", "SemEval-2007", llm_oriented, client
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="ollama/llama2:13b")
    parser.add_argument("--sample", type=int, default=30)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tasks", default="ea,el,wsd", help="Comma-separated subset of ea,el,wsd")
    parser.add_argument("--random-seed", type=int, default=20260827)
    parser.add_argument(
        "--ufsac-dir", type=Path, default=Path("~/data/ufsac-public-2.1").expanduser()
    )
    args = parser.parse_args()
    tasks = set(args.tasks.split(","))

    runs: list[BenchmarkRun] = []
    if "ea" in tasks:
        runs += _ea_runs(args.device, args.model, args.sample, args.random_seed)
    if "el" in tasks:
        runs += _el_runs(args.device, args.model, args.sample, args.random_seed)
    if "wsd" in tasks:
        runs += _wsd_runs(args.device, args.model, args.sample, args.random_seed, args.ufsac_dir)

    results = run_benchmarks(runs)
    print(format_table(results))


if __name__ == "__main__":
    main()

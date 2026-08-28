"""Benchmarks LlmBaseLinker on one small real dataset per task: Entity
Alignment (EA), Entity Linking (EL), and Word Sense Disambiguation (WSD).

Unlike every other benchmark in this repo, each prediction here is a real
LLM call -- free and local against Ollama, but seconds rather than
milliseconds per call, and a real dollar cost against a frontier API. EL
(MSNBC, 747 mentions) and WSD (SemEval-2007, 455 instances) are therefore
capped to `max_mentions` source entities by default (`--max-mentions`,
default 30) -- pass `--max-mentions 0` to run the full split. EA
(ConferenceDataset) is well under 100 classes total and isn't capped.

Datasets:

- EA: `linkingtk.datasets.naisc.ConferenceDataset` -- OAEI academic-
  conference ontology matching. Ontology labels across the two sides are
  related but rarely identical (e.g. "Conference_Paper" vs. "Paper"), so
  blocking uses `LabelOverlap`, not `ExactMatch`.
- EL: `linkingtk.datasets.wikification.MsnbcDataset` -- Cucerzan (2007)'s
  20-document gold standard. Fetches real Wikipedia lead-paragraph
  descriptions over the network the first time it's run (cached). Also
  uses `LabelOverlap`, not `ExactMatch` (the default) -- measured
  directly: `ExactMatch` gives at least one
  candidate to only 291/747 mentions (39%; mention surface forms in news
  text often don't exactly match their KB entity's canonical Wikipedia
  title), while `LabelOverlap` covers every mention checked. Using
  `ExactMatch` here would silently understate `LlmBaseLinker` by scoring
  it on well under half the dataset.
- WSD: `linkingtk.datasets.ufsac.UfsacDataset` on SemEval-2007 (the
  smallest Raganato eval split) against a real `WnEntitySource`, same
  setup as `glossbert_reproduction.py`. Requires a local UFSAC checkout
  (see that script's docstring for how to get one); point `--ufsac-path`
  elsewhere if yours isn't at `~/data/ufsac-public-2.1/`.

Defaults to a local Ollama model (`--model ollama/llama3.2:3b`) -- a
smaller `llama3.2:1b` was tried first and measured unreliable at this
task (frequently returned its own source entity's id as a "candidate",
and occasionally emitted truncated/invalid JSON under structured-output
mode); 3b was reliable in the same spot-check. Pass e.g.
`--model anthropic/claude-opus-4-6` to benchmark a frontier model instead
(reads the provider SDK's own API-key env var).

Run with: `uv run python examples/llm_benchmark.py`
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from linkingtk.algorithms.llm import LlmBaseLinker
from linkingtk.blocking.exact import ExactMatch
from linkingtk.blocking.label_overlap import LabelOverlap
from linkingtk.core.result import AlignmentResult
from linkingtk.datasets.naisc import ConferenceDataset
from linkingtk.datasets.ufsac import UfsacDataset
from linkingtk.datasets.wikification import MsnbcDataset
from linkingtk.eval import Evaluator
from linkingtk.llm.client import LlmClient, create_client


def _unranked_metrics(
    results: list[AlignmentResult], ground_truth: list[tuple[str, str]]
) -> dict[str, float | None]:
    predictions = [(r.source_id, r.target_id) for r in results]
    return Evaluator.evaluate(predictions=predictions, ground_truth=ground_truth).metrics


def run_ea(client: LlmClient) -> dict[str, float | None]:
    left, right, ground_truth = ConferenceDataset().load()
    linker = LlmBaseLinker(client=client, task="ea")
    results = linker.link(left, right, blocking=LabelOverlap(max_matches=5))
    ranked_predictions = [(r.source_id, [r.target_id, *r.alternatives]) for r in results]
    return Evaluator.evaluate_ranked(ranked_predictions, ground_truth, top_k=[1, 5]).metrics


def run_el(client: LlmClient, max_mentions: int | None) -> dict[str, float | None]:
    mentions, kb, ground_truth = MsnbcDataset().load()
    if max_mentions is not None:
        mentions = mentions[:max_mentions]
        mention_ids = {m.id for m in mentions}
        ground_truth = [(s, t) for s, t in ground_truth if s in mention_ids]
    linker = LlmBaseLinker(client=client, task="el")
    results = linker.link(mentions, kb, blocking=LabelOverlap(max_matches=5))
    return _unranked_metrics(results, ground_truth)


def run_wsd(
    client: LlmClient, ufsac_path: Path, max_mentions: int | None
) -> dict[str, float | None]:
    mentions, senses, ground_truth = UfsacDataset(source=str(ufsac_path)).load()
    if max_mentions is not None:
        mentions = mentions[:max_mentions]
        mention_ids = {m.id for m in mentions}
        ground_truth = [(s, t) for s, t in ground_truth if s in mention_ids]
    # top_k=50 (matching glossbert_reproduction.py -- wide enough that no
    # real lemma's sense count gets truncated) means some common verbs
    # ("make", "have") give the model up to 50 real candidates at once.
    # Measured directly: the smaller local model tried here doesn't reliably
    # stay within LlmBaseLinker's default 2048-token budget for prompts that
    # wide -- it over-generates (e.g. 95 rankings for 50 real candidates,
    # extras harmlessly ignored as hallucinated), and the response gets cut
    # off mid-JSON if max_tokens is too tight. 4096 fixes most of these
    # (a wider budget is a property of this specific wide-candidate WSD
    # setup, not a general default every LlmBaseLinker caller needs) but
    # not all -- a rare case or two still hits an apparent repetition loop
    # regardless of budget (raising max_tokens further only delayed, not
    # prevented, the eventual cutoff for one real mention observed during
    # benchmarking). LlmBaseLinker already handles this gracefully: that
    # one source entity is logged and skipped, not a crash.
    linker = LlmBaseLinker(client=client, task="wsd", max_tokens=4096)
    results = linker.link(mentions, senses, blocking=ExactMatch(top_k=50))
    return _unranked_metrics(results, ground_truth)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="ollama/llama3.2:3b")
    parser.add_argument("--max-mentions", type=int, default=30)
    parser.add_argument(
        "--ufsac-path",
        type=Path,
        default=Path("~/data/ufsac-public-2.1/raganato_semeval2007.xml").expanduser(),
    )
    args = parser.parse_args()
    max_mentions = args.max_mentions if args.max_mentions > 0 else None

    client = create_client(args.model)

    tasks = {
        "EA": lambda: run_ea(client),
        "EL": lambda: run_el(client, max_mentions),
        "WSD": lambda: run_wsd(client, args.ufsac_path, max_mentions),
    }
    for name, task in tasks.items():
        start = time.monotonic()
        metrics = task()
        elapsed = time.monotonic() - start
        print(f"{name}: {metrics} ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()

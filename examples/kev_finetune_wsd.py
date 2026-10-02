"""Exports SemCor as Kev fine-tuning data for Word Sense Disambiguation.

Writes `train.jsonl` and `heldout.jsonl` in `kev.train --data` format:
one choice question per SemCor instance, built by
[kev_choice_records][linkingtk.algorithms.kev.kev_choice_records] from
exactly the state, instructions and options that
`KevLinker(task="wsd", question_type="choice")` sends at inference.
Candidates are the lemma's WordNet senses of the mention's own POS
(`WordNetPosFilter(ExactMatch(top_k=50))`), the same as
`kev_benchmark.py`. Monosemous instances (one candidate) are skipped.

**Training options are shuffled** (`--no-shuffle` to keep WordNet order).
WordNet lists senses by SemCor frequency, so in WordNet order the gold
sense is option 1 for 65% of SemCor records, and a fine-tune could learn
"pick option 1" instead of reading the glosses. Each training record's
options are shuffled (seeded) and renumbered. The held-out file keeps
WordNet order, as at inference.

The instances are a seeded random sample of UFSAC's SemCor, split into
disjoint train and held-out sets. SemEval-2007 and the other Raganato
sets are never used for training.

Then, from a Kev checkout (see `kev_benchmark.py` for server setup):

    uv run python -m kev.train --data <out>/train.jsonl \\
        --base Qwen/Qwen3.5-4B-Base --init_from jaredpalmer/kev-4b \\
        --epochs 1 --lr 2e-5 --batch 1 --accum 8 --dtype bf16 \\
        --weights_dtype bf16 --checkpointing 1 --device cuda --out runs/kev-4b-semcor
    uv run python -m kev.benchmark --run runs/kev-4b-semcor \\
        --data <out>/heldout.jsonl --out runs/kev-4b-semcor-eval
    uv run --extra serve python -m kev.serve --run runs/kev-4b-semcor --port 8009

and evaluate on the Raganato sets with
`examples/kev_benchmark.py --task wsd --question-types choice --ufsac-path ...`.

Run with: `uv run python examples/kev_finetune_wsd.py --out data/kev_semcor`
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from linkingtk.algorithms.kev import kev_choice_records
from linkingtk.blocking.exact import ExactMatch
from linkingtk.blocking.pos import WordNetPosFilter
from linkingtk.datasets.ufsac import UfsacDataset

_SEED = 20261002


def _shuffle_options(record: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    """Shuffle a choice record's options, renumbering them and moving the label."""
    question = record["questions"]["q"]
    items = list(question["criteria"].items())
    rng.shuffle(items)
    names = [name for name, _ in items]
    criteria = {f"option {i + 1}": description for i, (_, description) in enumerate(items)}
    label = f"option {names.index(question['label']) + 1}"
    shuffled = {**question, "criteria": criteria, "label": label}
    return {**record, "questions": {"q": shuffled}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-size", type=int, default=20000)
    parser.add_argument("--heldout-size", type=int, default=1000)
    parser.add_argument("--no-shuffle", action="store_true", help="keep WordNet option order")
    parser.add_argument(
        "--semcor-path",
        type=Path,
        default=Path("~/data/ufsac-public-2.1/semcor.xml").expanduser(),
    )
    args = parser.parse_args()

    mentions, senses, ground_truth = UfsacDataset(source=str(args.semcor_path)).load()
    print(f"SemCor: {len(mentions)} instances")
    # Oversample: monosemous instances and ones whose gold isn't a
    # same-POS candidate are dropped by kev_choice_records.
    wanted = args.train_size + args.heldout_size
    sample = random.Random(_SEED).sample(mentions, min(len(mentions), 2 * wanted))
    ids = {m.id for m in sample}
    gold = [(s, t) for s, t in ground_truth if s in ids]

    records = kev_choice_records(
        sample,
        senses,
        gold,
        blocking=WordNetPosFilter(ExactMatch(top_k=50)),
        task="wsd",
    )
    print(f"usable records: {len(records)} of {len(sample)} sampled instances")
    if len(records) < wanted:
        raise SystemExit(f"only {len(records)} usable records, wanted {wanted}")

    train = records[args.heldout_size : wanted]
    if not args.no_shuffle:
        rng = random.Random(_SEED)
        train = [_shuffle_options(record, rng) for record in train]
    splits = {"heldout": records[: args.heldout_size], "train": train}
    args.out.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        path = args.out / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        options = Counter(len(r["questions"]["q"]["criteria"]) for r in rows)
        mean = sum(k * v for k, v in options.items()) / len(rows)
        first = sum(r["questions"]["q"]["label"] == "option 1" for r in rows) / len(rows)
        print(
            f"{path}: {len(rows)} records, {mean:.1f} options on average, "
            f"gold is option 1 in {first:.1%}"
        )


if __name__ == "__main__":
    main()

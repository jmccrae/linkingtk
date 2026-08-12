# Entity alignment: knowledge-graph-embedding benchmarks

Each linker below is trained on
[OpenEA's EN-FR-15K-V1](../datasets/real_world_ea.md#openea) dataset using
its **native train/test split** (`load_splits()`) and scored with
[`Evaluator.evaluate_ranked`](../reference/eval.md) for Hits@1, Hits@10 and
MRR. Only the train split's pairs are added to the graph as seed alignment
triples; the test split is held out and used only to score ranked
predictions — a genuine generalization signal, not a pipeline-correctness
check (compare to [the smaller `kge_ea.py` demo](kge_ea.md), which fully
seeds every known alignment pair). Candidate generation uses
[`LabelOverlap`](../reference/blocking.md) (not the tiny demos' `AllPairs`
helper, which is infeasible at 15K entities/side) — viable here because
many DBpedia entity labels are shared verbatim across the English/French
sides.

Requires the `kge` optional dependency group — install with
`uv sync --extra kge`. Fetches a ~28MB zip over the network the first time
it's run; cached under `~/.cache/linkingtk/downloads/` after that.

New knowledge-graph-embedding EA linkers are benchmarked here as an
additional section, rather than as a new page per method.

## KGELinker (baseline)

[`KGELinker`](../reference/algorithms.md) trains a single shared TransE
embedding space by folding the train split's alignment pairs into the
graph as extra seed triples, then reads off unaligned pairs by cosine
similarity of their trained embeddings — a simple, generic trick rather
than a reproduction of a specific published method.

```python
--8<-- "examples/kge_benchmark.py"
```

Run with:

```bash
uv run python examples/kge_benchmark.py
```

```text
3000 train / 10500 test pairs
Metrics: {'Hits@1': 0.07076635763356233, 'Hits@10': 0.6532381778163143, 'MRR': 0.20033756493771168}
```

## MTransE

[`MTransELinker`](../reference/algorithms.md) is a faithful reimplementation
of MTransE's (Chen, Tian, Yang & Zaniolo, IJCAI 2017) actual training
procedure, ported directly from
[OpenEA's reference implementation](https://github.com/nju-websoft/OpenEA)
rather than from a from-scratch reading of the paper: entities and
relations from both KGs share one embedding table, trained by minimizing
pure positive-triple loss (no negative sampling) with unit-L2-
normalization reapplied on every forward pass; a square mapping matrix,
orthogonally initialized, is trained jointly against the train split's
seed pairs via a soft-orthogonality-regularized loss, alternating one
epoch of triple training with one epoch of mapping training. The
validation split drives early stopping.

```python
--8<-- "examples/mtranse_benchmark.py"
```

Run with:

```bash
uv run python examples/mtranse_benchmark.py
```

```text
3000 train / 1500 val / 10500 test pairs
Metrics: {'Hits@1': 0.3972520509571133, 'Hits@10': 0.6532381778163143, 'MRR': 0.4901781942684364}
```

This exceeds MTransE's own published EN-FR-15K-V1 numbers outright
(Hits@1=0.247, Hits@10=0.564, MRR=0.351).

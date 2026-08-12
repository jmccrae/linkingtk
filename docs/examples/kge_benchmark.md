# KGE milestone acceptance demo

This is the Knowledge Graph Embeddings milestone's acceptance demo:
[`KGELinker`](../reference/algorithms.md) trained on
[OpenEA's EN-FR-15K-V1](../datasets/real_world_ea.md#openea) dataset using
its **native train/test split** (`load_splits()`), scored with
[`Evaluator.evaluate_ranked`](../reference/eval.md) for Hits@1, Hits@10 and
MRR.

Unlike [the smaller `kge_ea.py` demo](kge_ea.md), which fully seeds every
known alignment pair and so only checks that the training/scoring pipeline
recovers what it was directly taught, this trains with *only* the train
split's pairs added to the graph as seed alignment triples, then evaluates
ranked predictions against the held-out test split — a genuine
generalization signal, not a pipeline-correctness check. Candidate
generation uses [`LabelOverlap`](../reference/blocking.md) (not the tiny
demos' `AllPairs` helper, which is infeasible at 15K entities/side) —
viable here because many DBpedia entity labels are shared verbatim across
the English/French sides.

Requires the `kge` optional dependency group (for `pykeen`) — install
with `uv sync --extra kge`. Fetches a ~28MB zip over the network the first
time it's run; cached under `~/.cache/linkingtk/downloads/` after that.

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

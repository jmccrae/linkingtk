# MTransE benchmark

Per-method acceptance demo for the Knowledge Graph Embeddings milestone's
MTransE issue: [`MTransELinker`](../reference/algorithms.md) trained on
[OpenEA's EN-FR-15K-V1](../datasets/real_world_ea.md#openea) dataset using
its **native train/validation/test split** (`load_splits()`), scored with
[`Evaluator.evaluate_ranked`](../reference/eval.md) for Hits@1, Hits@10 and
MRR.

Unlike [`KGELinker`](kge_benchmark.md)'s simplified single-shared-
pseudo-relation trick, this is a faithful reimplementation of MTransE's
actual training procedure, ported directly from
[OpenEA's reference implementation](https://github.com/nju-websoft/OpenEA)
rather than from a from-scratch reading of the paper: entities and
relations from both KGs share one embedding table, trained by minimizing
pure positive-triple loss (no negative sampling) with unit-L2-
normalization reapplied on every forward pass; a square mapping matrix,
orthogonally initialized, is trained jointly against the train split's
seed pairs via a soft-orthogonality-regularized loss, alternating one
epoch of triple training with one epoch of mapping training. The
validation split drives early stopping; the test split is held out and
only used to score ranked predictions — a genuine generalization signal,
not a pipeline-correctness check. Candidate generation uses
[`LabelOverlap`](../reference/blocking.md) (not the tiny demos' `AllPairs`
helper, which is infeasible at 15K entities/side) — viable here because
many DBpedia entity labels are shared verbatim across the English/French
sides.

Requires the `kge` optional dependency group (for `torch`, pulled in
transitively by `pykeen` even though this linker doesn't call pykeen's
API directly) — install with `uv sync --extra kge`. Fetches a ~28MB zip
over the network the first time it's run; cached under
`~/.cache/linkingtk/downloads/` after that.

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

This comfortably clears the milestone's acceptance target for MTransE
(Hits@1 ≥ 0.222, ~10% relative of OpenEA's published EN-FR-15K-V1 number
of 0.247) and exceeds the published Hits@1/Hits@10/MRR outright.

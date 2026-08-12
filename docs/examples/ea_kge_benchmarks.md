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

## IPTransE

[`IPTransELinker`](../reference/algorithms.md) is a faithful reimplementation
of IPTransE's (Zhu, Xu, Yang, Lin & Cheng, IJCAI 2017) actual training
procedure, ported directly from
[OpenEA's reference implementation](https://github.com/nju-websoft/OpenEA)
rather than from a from-scratch reading of the paper: seed pairs are given a
single *shared* embedding row (not two rows plus a learned mapping, unlike
MTransE), trained by minimizing a joint margin-based TransE loss plus a
relation-composition ("path") loss over 2-hop chains, with negative sampling
restricted to each KG's own entities. Every `bootstrap_every` epochs, an
unsupervised self-training round finds new high-confidence matches by
structural embedding similarity alone (no labels), turns their real edges
into weighted pseudo-triples, and trains one extra epoch over them.

```python
--8<-- "examples/iptranse_benchmark.py"
```

Run with:

```bash
uv run python examples/iptranse_benchmark.py
```

```text
3000 train / 1500 val / 10500 test pairs
Metrics: {'Hits@1': 0.3454945641299273, 'Hits@10': 0.6532381778163143, 'MRR': 0.4492767259641801}
```

This exceeds IPTransE's own published EN-FR-15K-V1 numbers outright
(Hits@1=0.169, Hits@10=0.390, MRR=0.243).

## JAPE

[`JAPELinker`](../reference/algorithms.md) is a faithful reimplementation
of JAPE's (Sun, Hu & Li, ISWC 2017) actual training procedure, ported
directly from [OpenEA's reference implementation](https://github.com/nju-websoft/OpenEA)
rather than from a from-scratch reading of the paper: a shared-id
structural TransE embedding (trained via `pos_loss - neg_alpha * neg_loss`,
no margin/hinge) is regularized by an **attribute-correlation embedding**
-- a skip-gram-style model trained over which attribute predicates
co-occur on the same entity, with seed pairs' attribute vocabularies
merged for cross-lingual correlation signal. The resulting per-entity
attribute similarity, thresholded to keep only confident correlations,
pulls the structural embeddings of not-yet-seeded entities toward each
other during joint training.

**Note the different dataset**: `EnFr15KDataset` (used by the linkers
above) has no attribute triples at all, so this uses
[`EnFr15KAttrDataset`](../datasets/real_world_ea.md#openea-native-format-with-attributes)
instead -- a different, independently-sampled rehost of "EN-FR-15K" with
the same official split-size ratios but a different entity roster (see
that section for why). JAPE's numbers below aren't from the identical
entity sample as KGELinker's/MTransE's/IPTransE's above, though both are
genuinely OpenEA's official EN-FR-15K-V1 release.

```python
--8<-- "examples/jape_benchmark.py"
```

Run with:

```bash
uv run python examples/jape_benchmark.py
```

```text
3000 train / 1500 val / 10500 test pairs
Metrics: {'Hits@1': 0.3134089392928619, 'Hits@10': 0.656837891927952, 'MRR': 0.4240979700752883}
```

This exceeds JAPE's own published EN-FR-15K-V1 numbers outright
(Hits@1=0.263, Hits@10=0.595, MRR=0.372).

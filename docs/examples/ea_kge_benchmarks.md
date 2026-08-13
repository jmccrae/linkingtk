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

## KDCoE

[`KDCoELinker`][linkingtk.algorithms.ea.kdcoe.KDCoELinker] is a faithful
reimplementation of KDCoE's (Chen, Tian, Chang, Skiena & Zaniolo, IJCAI
2018) actual training procedure, ported directly from
[OpenEA's reference implementation](https://github.com/nju-websoft/OpenEA)
rather than from a from-scratch reading of the paper: a margin-based
structural TransE embedding, bridged across KGs by a learned mapping
matrix (the same "mapping" alignment module
[`MTransELinker`][linkingtk.algorithms.ea.mtranse.MTransELinker] uses), is
**co-trained** with a cross-lingual entity-description encoder (GRU ->
Conv1D -> attention-pool -> GRU -> attention-pool -> dense, over
pretrained fastText word vectors). Each outer co-training iteration trains
the description encoder to convergence, uses it to find new
high-confidence pairs among not-yet-seeded entities, trains the
structural+mapping model to convergence (now also pulled toward those
pairs), and uses *it* to find new pairs the same way -- stopping once a
round re-discovers nothing new.

**Same dataset note as JAPE above**: this uses
[`EnFr15KAttrDataset`][linkingtk.datasets.openea_native.EnFr15KAttrDataset],
not `EnFr15KDataset`, for the same attribute-triples reason.

**Word vectors**: unlike every other linker on this page, this also
downloads fastText's pretrained `wiki-news-300d-1M.vec.zip` (~681MB,
cached after the first run) -- the description encoder's whole point is
leveraging generic-language priors a 15K-entity alignment task alone can't
teach it.

```python
--8<-- "examples/kdcoe_benchmark.py"
```

Run with:

```bash
uv run python examples/kdcoe_benchmark.py
```

```text
3000 train / 1500 val / 10500 test pairs
Metrics: {'Hits@1': 0.386057371581054, 'Hits@10': 0.656837891927952, 'MRR': 0.47764649978292406}
```

**This falls short of KDCoE's own published EN-FR-15K-V1 numbers**
(Hits@1=0.581, Hits@10=0.721, MRR=0.628) -- unlike every other linker on
this page, which meets or exceeds its published target. Investigated
directly rather than assumed: a diagnostic run found the description
encoder's cosine similarities saturate near `1.0` for ~99.9% of
reference-pool pairs regardless of correctness (median ~0.9998), so
`desc_sim_th` couldn't act as a confidence filter -- feeding that many
low-precision pairs into the structural mapping loss collapsed structural
Hits@1 from a clean ~0.40 to near-zero (see `KDCoELinker`'s module
docstring for the fix: description-found and structurally-found bootstrap
pairs are now tracked and fed back separately). That fix alone recovered
Hits@1 from 0.343 to 0.386, but co-training still only roughly matches a
plain structural-only [`MTransELinker`][linkingtk.algorithms.ea.mtranse.MTransELinker]
baseline measured on this exact dataset/split (Hits@1=0.396, no
descriptions, no co-training) -- the description signal isn't adding the
value the published number implies. The likely root cause is data
availability rather than a training bug: `EnFr15KAttrDataset`'s real
`.../description`-predicate attribute-triple coverage is sparse (10.5% of
KG1 entities, 0.5% of KG2 entities -- see
[the datasets page](../datasets/real_world_ea.md#openea-native-format-with-attributes)),
so most entities' "descriptions" are single-word label fallbacks (see
`KDCoELinker`'s module docstring) rather than the rich descriptive text
KDCoE's method -- and its published benchmark -- depends on.

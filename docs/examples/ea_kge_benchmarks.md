# Entity alignment: knowledge-graph-embedding benchmarks

Each linker below is trained on
[OpenEA's EN-FR-15K-V1](../datasets/real_world_ea.md#openea) dataset using
its **native train/test split** (`load_splits()`) and scored with
[`Evaluator.evaluate_ranked`](../reference/eval.md) for Hits@1, Hits@10 and
MRR. Only the train split's pairs are added to the graph as seed alignment
triples; the test split is held out and used only to score ranked
predictions — a genuine generalization signal, not a pipeline-correctness
check (compare to [the smaller `kge_ea.py` demo](kge_ea.md), which fully
seeds every known alignment pair). Ranking is **exhaustive**: every
test-source entity is ranked against every test-target entity directly
from trained embeddings via
[`rank_exhaustive`](../reference/eval.md), with no candidate-restriction
step at all — matching OpenEA's own evaluation methodology
(`greedy_alignment`), so these numbers are directly comparable to each
method's published target.
[`LabelOverlap`](../reference/blocking.md)-style blocking remains the
right tool for production-scale linking (see [the smaller `kge_ea.py`
demo](kge_ea.md)) but is deliberately not used for benchmark *scoring*
here — see issue
[#37](https://github.com/jmccrae/linkingtk/issues/37) for why.

Requires the `kge` optional dependency group — install with
`uv sync --extra kge`. Fetches a ~28MB zip over the network the first time
it's run; cached under `~/.cache/linkingtk/downloads/` after that.

Every hand-rolled-PyTorch linker below (all except `KGELinker`, which is
pykeen-backed and handles device placement on its own) accepts a
`device` constructor parameter — `"cpu"` (default) or `"cuda"`/`"cuda:0"`
— for training on a GPU. It's purely additive: results are unaffected,
only wall-clock time changes.

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
Metrics: {'Hits@1': 9.523809523809524e-05, 'Hits@10': 0.0013333333333333333, 'MRR': 0.0011284690886909938}
```

**Near chance level under exhaustive ranking** (chance Hits@1 over a
~10500-entity test-target pool is ~0.0000952, essentially identical to
the number above). The much higher blocking-restricted number this
section previously reported (Hits@1=0.101) came almost entirely from
`LabelOverlap` narrowing the candidate pool to at most 10 entities before
KGE's weak embeddings ever had to rank anything -- consistent with this
linker's own documented framing as "a simple, generic trick rather than a
reproduction of a specific published method," not a real EA method with
learned cross-lingual structure.

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
Metrics: {'Hits@1': 0.24514285714285713, 'Hits@10': 0.563047619047619, 'MRR': 0.34934607320595473}
```

**Matches MTransE's own published EN-FR-15K-V1 numbers closely**
(Hits@1=0.247, Hits@10=0.564, MRR=0.351). This section previously
reported Hits@1=0.065 under exhaustive ranking -- a real bug, not an
inherent limitation of the method, found by directly running OpenEA's
own TensorFlow reference implementation against byte-identical data and
comparing training trajectories: OpenEA's own `generate_optimizer(loss,
lr, var_list=None, ...)` for the mapping loss leaves `var_list` unset,
which makes TensorFlow's `compute_gradients` differentiate against
*every* trainable variable the loss touches -- not just the mapping
matrix, but the shared entity embeddings too. This port's mapping
optimizer only included the mapping matrix (the more "obvious" reading
of the algorithm), silently dropping that second training signal and
leaving the mapping loss converging roughly an order of magnitude
slower than OpenEA's real training run. See
[issue #26](https://github.com/jmccrae/linkingtk/issues/26). A separate,
smaller factor: this example now uses `EnFr15KAttrDataset` instead of
`EnFr15KDataset`, since the latter's rehosted zip is missing ~20% of
relation triples relative to OpenEA's own release.

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
Metrics: {'Hits@1': 0.11695238095238095, 'Hits@10': 0.2885714285714286, 'MRR': 0.1770713003174724}
```

**Falls short of IPTransE's own published EN-FR-15K-V1 numbers**
(Hits@1=0.169, Hits@10=0.390, MRR=0.243) once ranking is genuinely
exhaustive -- same finding as MTransE above (this section previously
reported a blocking-inflated Hits@1=0.493). See
[issue #37](https://github.com/jmccrae/linkingtk/issues/37).

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
Metrics: {'Hits@1': 0.05838095238095238, 'Hits@10': 0.23866666666666667, 'MRR': 0.12116213264596698}
```

**Falls well short of JAPE's own published EN-FR-15K-V1 numbers**
(Hits@1=0.263, Hits@10=0.595, MRR=0.372) once ranking is genuinely
exhaustive -- same finding as MTransE/IPTransE above (this section
previously reported a blocking-inflated Hits@1=0.448). The
attribute-correlation regularizer clearly isn't enough on its own to
make the structural embedding discriminate well across the full
test-target pool. See
[issue #37](https://github.com/jmccrae/linkingtk/issues/37).

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
Metrics: {'Hits@1': 0.2018095238095238, 'Hits@10': 0.47104761904761905, 'MRR': 0.2927737429133085}
```

**Improved substantially but still falls short of KDCoE's own published
EN-FR-15K-V1 numbers** (Hits@1=0.581, Hits@10=0.721, MRR=0.628). This
section previously reported Hits@1=0.061 under exhaustive ranking;
KDCoE shares [`MTransELinker`][linkingtk.algorithms.ea.mtranse.MTransELinker]'s
mapping-loss bug (see [issue #26](https://github.com/jmccrae/linkingtk/issues/26))
-- its mapping optimizers only included the mapping matrix, not the
shared entity embeddings OpenEA's `var_list=None` also updates -- and
fixing it alone recovered most of the gap (0.061 -> 0.202), a >3x
improvement. The remaining shortfall against the 0.581 target is
consistent with the diagnostic below, which independently found the
co-trained description signal isn't adding real value on this dataset:
a diagnostic run found the description encoder's cosine similarities
saturate near `1.0` for ~99.9% of reference-pool pairs regardless of
correctness (median ~0.9998), so `desc_sim_th` couldn't act as a
confidence filter -- feeding that many low-precision pairs into the
structural mapping loss collapsed structural Hits@1 from a clean ~0.40 to
near-zero (see `KDCoELinker`'s module docstring for the fix:
description-found and structurally-found bootstrap pairs are now tracked
and fed back separately). The likely root cause is data availability
rather than a training bug: `EnFr15KAttrDataset`'s real
`.../description`-predicate attribute-triple coverage is sparse (10.5% of
KG1 entities, 0.5% of KG2 entities -- see
[the datasets page](../datasets/real_world_ea.md#openea-native-format-with-attributes)),
so most entities' "descriptions" are single-word label fallbacks (see
`KDCoELinker`'s module docstring) rather than the rich descriptive text
KDCoE's method -- and its published benchmark -- depends on. See
[issue #29](https://github.com/jmccrae/linkingtk/issues/29), which
remains open pending further investigation into the description
pathway's weak signal.

## AttrE

[`AttrELinker`][linkingtk.algorithms.ea.attre.AttrELinker] is a faithful
reimplementation of AttrE's (Trisedya, Qi & Zhang, AAAI 2019) actual
training procedure, ported directly from
[OpenEA's reference implementation](https://github.com/nju-websoft/OpenEA)
rather than from a from-scratch reading of the paper: a shared-id
structural TransE embedding (the same "sharing" mechanism
[`IPTransELinker`][linkingtk.algorithms.ea.iptranse.IPTransELinker]/
[`JAPELinker`][linkingtk.algorithms.ea.jape.JAPELinker] use) is trained
jointly with a *second*, independent character/attribute-embedding space
-- attribute triples are treated as TransE-style triples themselves
(`entity + attribute ≈ compose(value's characters)`), with each value's
embedding built compositionally from its characters. A per-entity
cosine-similarity ("joint") loss ties the two spaces together during
training. Despite the paper's "attributes only" framing, OpenEA's own
benchmarked implementation trains the structural half too -- this ports
what's actually benchmarked, not a structure-free reduction (see
`AttrELinker`'s module docstring).

**Same dataset note as JAPE/KDCoE above**: this uses
[`EnFr15KAttrDataset`][linkingtk.datasets.openea_native.EnFr15KAttrDataset],
not `EnFr15KDataset`, for the same attribute-triples reason.

```python
--8<-- "examples/attre_benchmark.py"
```

Run with:

```bash
uv run python examples/attre_benchmark.py
```

```text
3000 train / 1500 val / 10500 test pairs
Metrics: {'Hits@1': 0.6266666666666667, 'Hits@10': 0.836, 'MRR': 0.700697693983391}
```

Still exceeds AttrE's own published EN-FR-15K-V1 numbers across all
three metrics (Hits@1=0.481, Hits@10=0.732, MRR=0.569), though by a
smaller margin than the blocking-restricted number this section
previously reported (Hits@1=0.823) -- unlike MTransE/IPTransE/JAPE/KDCoE
above, AttrE's character/attribute-embedding half gives it enough
discriminative signal to keep clearing its target under genuinely
exhaustive ranking. See
[issue #37](https://github.com/jmccrae/linkingtk/issues/37).

## IMUSE

[`IMUSELinker`][linkingtk.algorithms.ea.imuse.IMUSELinker] is a faithful
reimplementation of IMUSE's (He, Li, Qiao, Liu & Zhao, DASFAA 2019) actual
training procedure, ported directly from
[OpenEA's reference implementation](https://github.com/nju-websoft/OpenEA)
rather than from a from-scratch reading of the paper. **Unlike every other
linker on this page, IMUSE is unsupervised** -- it takes no ground-truth
seed pairs at all. Instead it bootstraps its own initial alignment purely
from attribute-value string similarity (attribute predicates are paired
across KGs by Levenshtein-ratio similarity of their local names, then
entities are paired by the average Levenshtein ratio of their values under
those aligned predicates), then trains a margin-based structural TransE
embedding jointly with a direct squared-distance alignment loss that pulls
each bootstrapped pair's (still separate-row) entity embeddings together --
no mapping matrix, no shared-id merge. A held-out validation split still
drives early stopping (matching OpenEA's own published config), so the
"unsupervised" claim is specifically about training signal, not early
stopping.

**Same dataset note as JAPE/KDCoE/AttrE above**: this uses
[`EnFr15KAttrDataset`][linkingtk.datasets.openea_native.EnFr15KAttrDataset],
not `EnFr15KDataset`, for the same attribute-triples reason.

**New dependency**: `rapidfuzz`, for the Levenshtein-ratio string
similarity OpenEA's own bootstrap relies on -- added to the `kge` extra
alongside `torch`/`pykeen`.

```python
--8<-- "examples/imuse_benchmark.py"
```

Run with:

```bash
uv run python examples/imuse_benchmark.py
```

```text
3000 train (unused) / 1500 val / 10500 test pairs
Metrics: {'Hits@1': 0.6746666666666666, 'Hits@10': 0.795047619047619, 'MRR': 0.7147113857910671}
```

Still exceeds IMUSE's own published EN-FR-15K-V1 numbers across all
three metrics (Hits@1=0.569, Hits@10=0.777, MRR=0.638), though by a
smaller margin than the blocking-restricted number this section
previously reported (Hits@1=0.811) -- like AttrE above, IMUSE's
attribute-value bootstrap gives it enough discriminative signal to keep
clearing its target under genuinely exhaustive ranking. See
[issue #37](https://github.com/jmccrae/linkingtk/issues/37). Note
`name_sim_threshold=0.9` above, not `IMUSELinker`'s own literal-reference
default of `0.6` -- diagnosed directly (not assumed): at `0.6`, a
spurious attribute-predicate pairing (`.../ontology/games` and
`foaf/0.1/name` score `0.667` on local-name similarity, edging out the
correct `foaf/0.1/name` self-match by triple count) drags the
entity-bootstrap step's precision against this dataset's own known links
down to 54.7%; `0.9` removes it, raising precision to 82.1%. See
[`IMUSELinker`][linkingtk.algorithms.ea.imuse.IMUSELinker]'s module
docstring for the full diagnostic -- likely a rehost-specific
attribute-predicate-vocabulary difference from OpenEA's own original
dataset files, not a bug in the port.

## MultiKE

[`MultiKELinker`][linkingtk.algorithms.ea.multike.MultiKELinker] is a
faithful reimplementation of MultiKE's (Zhang, Sun, Hu, Chen, Guo & Qu,
IJCAI 2019) actual training procedure, ported directly from
[OpenEA's reference implementation](https://github.com/nju-websoft/OpenEA)
rather than from a from-scratch reading of the paper. By far the most
complex method in this family: three embedding views (a frozen literal
name view, a logistic-loss TransE relation view, and a CNN-scored
attribute view) unified into one shared table, trained supervised via
seed-pair entity-id substitution (its only real supervision channel) plus
a fixed, one-time Levenshtein-based predicate soft-alignment. See
[`MultiKELinker`][linkingtk.algorithms.ea.multike.MultiKELinker]'s module
docstring for the full architecture writeup and three documented scope
cuts (space mapping is dead code in OpenEA's own published run;
predicate alignment is static, not periodically re-estimated; negative
sampling follows this family's established uniform-sampling precedent).

**Same dataset note as JAPE/KDCoE/AttrE/IMUSE above**: this uses
[`EnFr15KAttrDataset`][linkingtk.datasets.openea_native.EnFr15KAttrDataset],
not `EnFr15KDataset`, for the same attribute-triples reason.

**New dependency**: none. `transformers` (the literal encoder) is already
a base dependency of this repo; `torch`/`rapidfuzz` (predicate-name
matching) are already in the `kge` extra from prior issues.

```python
--8<-- "examples/multike_benchmark.py"
```

Run with:

```bash
uv run python examples/multike_benchmark.py
```

```text
3000 train / 1500 val / 10500 test pairs
Metrics: {'Hits@1': 0.8467619047619047, 'Hits@10': 0.9185714285714286, 'MRR': 0.8714352203977747}
```

Still comfortably clears both the acceptance target (Hits@1 >= 0.674)
and OpenEA's own published Hits@1 (0.749) under genuinely exhaustive
ranking -- the strongest result on this page, and (along with AttrE and
IMUSE) proof that a real, well-signaled EA method holds up without
blocking's help; this section previously reported a blocking-inflated
Hits@1=0.874.

Getting a trustworthy number here originally surfaced a real,
pre-existing bug in
[`Evaluator.evaluate_ranked`][linkingtk.eval.evaluator.Evaluator.evaluate_ranked]
affecting every benchmark on this page: every script here calls `link()`
over the *full* entity set (train+val+test) since which entities need
embeddings is decided before the split is known, but `evaluate_ranked`
was dividing by `len(ranked_predictions)` (all linked entities) rather
than `len(ground_truth)` (only the test split) -- diluting every
reported Hits@k/MRR number by the ratio of non-test-set entities to
test-set entities (roughly 1.43x on `EnFr15KAttrDataset`). That's fixed
in `evaluate_ranked` itself (iterates over `ground_truth`, so both a
missing prediction *and* an extra non-ground-truth prediction are
handled correctly).

That fix alone wasn't the full story, though: even with it, every
benchmark script still scored predictions via
`LabelOverlap(max_matches=10)` blocking, restricting each source
entity's candidate pool to at most 10 before the linker's own embeddings
ever ranked anything -- a structurally different (and easier) task than
OpenEA's own published methodology, which ranks every test-source entity
against the *full* test-target pool (`greedy_alignment`, no candidate
restriction). [Issue #37](https://github.com/jmccrae/linkingtk/issues/37)
replaced blocking-restricted benchmark scoring with
[`rank_exhaustive`](../reference/eval.md), matching OpenEA's methodology
exactly -- every number on this page reflects that. The result was not
uniform: MTransE, IPTransE, JAPE and KDCoE (structural-embedding-only or
weakly-regularized methods) all now fall well short of their published
targets once ranking is genuinely exhaustive, while AttrE, IMUSE and
MultiKE (all backed by strong attribute-value signal) still clear
theirs. See each section above for details.

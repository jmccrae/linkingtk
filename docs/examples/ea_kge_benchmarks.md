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
Metrics: {'Hits@1': 0.10104761904761905, 'Hits@10': 0.9327619047619048, 'MRR': 0.28606296296296296}
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
Metrics: {'Hits@1': 0.5672380952380952, 'Hits@10': 0.9327619047619048, 'MRR': 0.6999277777777778}
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
Metrics: {'Hits@1': 0.49323809523809525, 'Hits@10': 0.9327619047619048, 'MRR': 0.6415356386999244}
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
Metrics: {'Hits@1': 0.4478095238095238, 'Hits@10': 0.9377142857142857, 'MRR': 0.6055761904761905}
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
Metrics: {'Hits@1': 0.5412380952380952, 'Hits@10': 0.9377142857142857, 'MRR': 0.6745544217687075}
```

**Hits@1 still falls a little short of KDCoE's own published EN-FR-15K-V1
number** (0.581 vs. 0.541 here), though Hits@10 (0.938) and MRR (0.675)
now exceed the published 0.721/0.628 -- unlike the pre-fix numbers below,
where all three metrics fell short. Investigated directly rather than
assumed: a diagnostic run found the description encoder's cosine
similarities saturate near `1.0` for ~99.9% of reference-pool pairs
regardless of correctness (median ~0.9998), so `desc_sim_th` couldn't act
as a confidence filter -- feeding that many low-precision pairs into the
structural mapping loss collapsed structural Hits@1 from a clean ~0.40 to
near-zero (see `KDCoELinker`'s module docstring for the fix:
description-found and structurally-found bootstrap pairs are now tracked
and fed back separately). That fix alone recovered Hits@1 from 0.343 to
0.386 (pre-evaluator-fix numbers); co-training still only roughly matches
a plain structural-only
[`MTransELinker`][linkingtk.algorithms.ea.mtranse.MTransELinker] baseline
measured on this exact dataset/split -- the description signal isn't
adding the value the published number implies. The likely root cause is
data availability rather than a training bug: `EnFr15KAttrDataset`'s real
`.../description`-predicate attribute-triple coverage is sparse (10.5% of
KG1 entities, 0.5% of KG2 entities -- see
[the datasets page](../datasets/real_world_ea.md#openea-native-format-with-attributes)),
so most entities' "descriptions" are single-word label fallbacks (see
`KDCoELinker`'s module docstring) rather than the rich descriptive text
KDCoE's method -- and its published benchmark -- depends on. **Note**: the
numbers above (and every other section on this page) were re-run after
fixing a denominator bug in
[`Evaluator.evaluate_ranked`][linkingtk.eval.evaluator.Evaluator.evaluate_ranked]
(see the MultiKE section below) -- KDCoE's own diagnostic investigation
above predates that fix and used the older, deflated numbers throughout
(0.343/0.386 are as originally measured, not re-verified against the
fix), but the sparse-description-data root cause is independent of the
evaluator bug and still explains the remaining Hits@1 gap.

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
Metrics: {'Hits@1': 0.8234285714285714, 'Hits@10': 0.9377142857142857, 'MRR': 0.8681930461073318}
```

This exceeds AttrE's own published EN-FR-15K-V1 numbers outright across
all three metrics (Hits@1=0.481, Hits@10=0.732, MRR=0.569).

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
Metrics: {'Hits@1': 0.8106666666666666, 'Hits@10': 0.9377142857142857, 'MRR': 0.8555479591836734}
```

This exceeds IMUSE's own published EN-FR-15K-V1 numbers outright across
all three metrics (Hits@1=0.569, Hits@10=0.777, MRR=0.638). Note
`name_sim_threshold=0.9` above, not `IMUSELinker`'s own literal-reference
default of `0.6` -- diagnosed directly (not assumed): at `0.6`, a
spurious attribute-predicate pairing (`.../ontology/games` and
`foaf/0.1/name` score `0.667` on local-name similarity, edging out the
correct `foaf/0.1/name` self-match by triple count) drags the
entity-bootstrap step's precision against this dataset's own known links
down to 54.7%; `0.9` removes it, raising precision to 82.1% (this
particular before/after comparison predates the evaluator-denominator fix
described in the MultiKE section below, so the `0.470`/`0.571` Hits@1
figures it was originally diagnosed against are the older, deflated
numbers -- the relative improvement from the threshold change itself
still holds). See
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
Metrics: {'Hits@1': 0.8738095238095238, 'Hits@10': 0.9377142857142857, 'MRR': 0.897093008314437}
```

This comfortably clears both the acceptance target (Hits@1 >= 0.674) and
OpenEA's own published Hits@1 (0.749) outright. Getting a trustworthy
number here surfaced a real, pre-existing bug in
[`Evaluator.evaluate_ranked`][linkingtk.eval.evaluator.Evaluator.evaluate_ranked]
affecting every benchmark on this page (not just MultiKE): every script
here calls `link()` over the *full* entity set (train+val+test) since
which entities need embeddings is decided before the split is known, but
`evaluate_ranked` was dividing by `len(ranked_predictions)` (all linked
entities) rather than `len(ground_truth)` (only the test split) --
diluting every reported Hits@k/MRR number by the ratio of
non-test-set-entities to test-set entities (roughly 1.43x on
`EnFr15KAttrDataset`). Confirmed directly: MultiKE's *un*trained,
name-embedding-only baseline went from Hits@1=0.593 (the bug) to
Hits@1=0.848 (fixed) on the exact same predictions. Now fixed in
`evaluate_ranked` itself (iterates over `ground_truth`, so both a missing
prediction *and* an extra non-ground-truth prediction are handled
correctly) -- every number on this page has been re-run and updated
against the fix. See
[issue #37](https://github.com/jmccrae/linkingtk/issues/37) for a related,
separate finding: blocking (`LabelOverlap(max_matches=10)`) still caps
what these numbers can reach, since it restricts candidates *before*
ranking, unlike OpenEA's own exhaustive (no-blocking) evaluation --
diagnosed blocking recall on this split is ~93.8%, which is why Hits@10
above sits so close to that ceiling regardless of embedding quality.

# Entity alignment: Simple-HHEA on ICEWS-WIKI

[`SimpleHHEALinker`](../reference/algorithms.md) is trained on
[ICEWS-WIKI](../datasets/real_world_ea.md#icews) using its native
train/test split and scored with
[`Evaluator.evaluate_ranked`](../reference/eval.md) for Hits@1, Hits@10 and
MRR, over an exhaustive ranking (every test-source entity against every
test-target entity via [`rank_exhaustive`](../reference/eval.md), no
candidate-restriction step) with Cross-domain Similarity Local Scaling
(`csls_k=10`) -- matching the reference's own `CSLS_.py` evaluation
methodology, same "exhaustive, not blocking-restricted" discipline as
[the EN-FR-15K KGE benchmarks](ea_kge_benchmarks.md).

Requires the `kge` optional dependency group (for `networkx`/`gensim`,
alongside `pykeen`) -- install with `uv sync --extra kge`. Fetches
ICEWS-WIKI's zip over the network the first time it's run (~75MB,
~3.5M ICEWS event triples); cached under `~/.cache/linkingtk/downloads/`
after that. Trains on GPU (`device="cuda"`) -- ICEWS-WIKI is a large,
heterogeneous dataset (~27K entities combined), unlike the 15K-scale
homogeneous OpenEA fixtures the KGE benchmarks page uses.

```python
--8<-- "examples/simple_hhea_ea.py"
```

Run with:

```bash
uv run python examples/simple_hhea_ea.py
```

```text
1518 train / 3540 test pairs
Metrics: {'Hits@1': 0.8146892655367232, 'Hits@10': 0.8822033898305085, 'MRR': 0.8383843542539284}
```

**Beats the paper's published Simple-HHEA number for ICEWS-WIKI**
(Hits@1=0.720, Hits@10=0.872, MRR=0.754, Table 2 of
[the ChatEA paper](https://aclanthology.org/2024.acl-long.408.pdf), which
reports Simple-HHEA as its own base-embedding baseline) -- but only after
a longer investigation than that summary suggests, including a real bug
in how this repo first ported the reference's node2vec walk generation,
and a final surprising finding that the structural (node2vec) branch
should be **disabled** for this dataset rather than fixed further.

## The node2vec convention bug

The first implementation used the third-party `node2vec` PyPI package
(a thin `gensim.Word2Vec` wrapper) instead of hand-porting the
reference's own `feature_perprocessing/longterm/node2vec.py`, reusing the
reference's literal `p=1e-100, q=1` hyperparameter values. This produced
a real number (Hits@1=0.262) but a wrong one: the reference's
`get_alias_edge` uses a **non-standard** transition-weight formula for
the "return to the previous node" term (`weight * p` -- the file even
has the textbook `weight / p` commented out directly above it, a
deliberate inversion of the standard Grover & Leskovec convention). The
third-party package implements the *standard* divide-by-`p` formula, so
copying the reference's literal `p` value into it did the *opposite* of
the intended "never immediately backtrack": sampled walks degenerated
into pure two-node oscillation (`['d', 'c', 'd', 'c', ...]`).

Retuning `p` for the standard package's own convention (`p=100`)
produced genuinely exploratory walks, but scored *worse* on the real
benchmark (Hits@1=0.037), even after ruling out undertraining (10x more
Word2Vec epochs: 0.041) and hub-node dilution (a 10x harder degree cap:
0.025) as explanations. The reference's actual formula isn't equivalent
to either end of the standard package's p/q dial -- at `q=1`, its `elif`
branch (a neighbor that's also adjacent to the previous node) gets
zeroed out too, alongside the return term, a third walk character the
standard package has no way to express at all.

The fix was to hand-port the reference's exact alias-sampling formula
directly (`_alias_setup`/`_alias_draw`/`_get_alias_edge`/`_node2vec_walk`/
`_simulate_walks` in
[`_simple_hhea_structure`][linkingtk.algorithms.ea._simple_hhea_structure]),
rather than continue searching for an equivalent parameterization of the
third-party package -- verified directly via toy-graph walk inspection
(no more oscillation) and via `mypy --strict`/unit tests. The `node2vec`
PyPI dependency was removed; `networkx` and `gensim` (previously only
transitive) are now explicit `kge`-extra dependencies.

## Structure hurts on this dataset

Surprisingly, the faithfully-ported reference formula scored **worse**
than the buggy convention-mismatched version: Hits@1=0.0147. This wasn't
predicted -- it was reported as a genuine surprise, then investigated
with a direct diagnostic: disabling the structural branch entirely
(`use_structure=False`, name + time embeddings only) scores
**Hits@1=0.8147**, comfortably beating every structure-enabled variant
tried, including the paper's own published number.

Across five structure-generation variants tested, node2vec structural
embeddings hurt performance on ICEWS-WIKI in every case:

| Variant | Hits@1 |
| --- | --- |
| Buggy convention-mismatched walk (third-party package, reference's `p` value) | 0.262 |
| Corrected standard-package walk (`p=100`) | 0.037 |
| Same, 10x more Word2Vec epochs | 0.041 |
| Same, 10x harder degree cap | 0.025 |
| Faithfully hand-ported reference formula | 0.0147 |
| **No structure (name + time only)** | **0.8147** |

The likely cause: ICEWS-WIKI is deliberately the paper's "highly
heterogeneous" stress-test dataset. The ICEWS side's structure is a
dense political/military event network; the Wikipedia side's is a
hyperlink graph -- two fundamentally different kinds of structure. Only
the 1,518 train-pair-merged nodes tie the two graphs together at all, so
a node2vec embedding trained mostly within one side has little reason to
be directly comparable across sides by cosine similarity. The name
branch has no such problem: it comes from one shared pretrained language
model (ALBERT), so embeddings from both sides land in the same space by
construction.

**[`examples/simple_hhea_ea.py`](https://github.com/jmccrae/linkingtk/blob/main/examples/simple_hhea_ea.py)
therefore uses `use_structure=False`** for this specific benchmark, based
on this evidence. `SimpleHHEALinker`'s own constructor default remains
`use_structure=True` -- faithful to the reference, and plausibly still
useful on more structurally-homogeneous datasets (e.g. the OpenEA
DBP15K-style fixtures, where both sides' graphs come from the same kind
of source).

## A scalability fix along the way

Independent of the convention bug above, ICEWS-WIKI's combined graph has
an extreme power-law degree distribution (median node degree 18, but the
single highest-degree node -- "United States" -- has 273,317 edges,
since ICEWS records every political/military event involving a state
actor as a triple). node2vec's alias-table transition-probability
precompute is `O(degree)` per edge, so a hub node alone costs
`O(degree^2)` -- confirmed to never finish in practice. Fixed by adding
[`cap_node_degree`][linkingtk.algorithms.ea._simple_hhea_structure.cap_node_degree],
which randomly subsamples any node's excess edges down to
`SimpleHHEALinker`'s `max_degree` (default 1000) before walk generation
-- a standard mitigation for random-walk embeddings on scale-free graphs,
not something either the reference implementation or a toy-scale
unit-test fixture needed to consider. (This fix stayed relevant across
the node2vec rewrite -- the hand-ported walk code still needs it.)

A related `PYTHONHASHSEED`-driven determinism bug was also found and
fixed: deduplicating edges via a plain Python `set` before capping meant
`cap_node_degree`'s seeded `rng.permutation` was shuffling *indices into
a hash-order-dependent list*, so the same `random_state` picked different
actual edges across separate process runs. Caught because it made a
fully-seeded unit test flaky across repeated runs. Fixed by sorting the
deduplicated edge list before capping.

## Evaluation metric fix

The first correctness pass also used the wrong evaluation metric
(`metric="inner"`): the reference's own `evaluate()` L2-normalizes both
embedding matrices *before* `CSLS_.py`'s raw `matmul`, making that raw
matmul equivalent to cosine similarity, not an unnormalized inner
product. `rank_exhaustive`'s `metric="cosine"` is the faithful match.

## Other simplifications vs. the reference

Made deliberately and documented up front, not found via debugging --
see
[`_simple_hhea_structure`][linkingtk.algorithms.ea._simple_hhea_structure]'s
module docstring for the full reasoning:

- Walks are entity-only, with no relation-id interleaving (an unablated,
  paper-text-absent reference-code embellishment).
- The reference's `node2same` node-merge trick is driven directly by
  `SimpleHHEALinker.fit()`'s own `ground_truth` argument, instead of
  reverse-engineering the reference's `ref_ent_ids[:1500]` file-position
  slice.

This is the base embedding method [#22 (ChatEA)](https://github.com/jmccrae/linkingtk/issues/22)
will build its LLM-reranking benchmark on top of, for a genuine
comparison against the ChatEA paper's own published llama2-13b number --
not [`KGELinker`](ea_kge_benchmarks.md), which uses a different
(structure-only TransE) base embedding entirely.

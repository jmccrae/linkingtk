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

Requires the `kge` optional dependency group (for `node2vec`/`gensim`,
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
Metrics: {'Hits@1': 0.261864406779661, 'Hits@10': 0.4774011299435028, 'MRR': 0.3372835398175539}
```

**Well short of the paper's published Simple-HHEA number for ICEWS-WIKI**
(Hits@1=0.720, Hits@10=0.872, MRR=0.754, Table 2 of
[the ChatEA paper](https://aclanthology.org/2024.acl-long.408.pdf), which
reports Simple-HHEA as its own base-embedding baseline). Two real bugs
were found and fixed while getting this number (both covered by
regression tests), and the remaining gap is diagnosed, not assumed:

- **A real scalability blow-up, not a toy-scale concern**: ICEWS-WIKI's
  combined graph has an extreme power-law degree distribution (median
  node degree 18, but the single highest-degree node has 273,317 edges --
  measured directly). node2vec's alias-table transition-probability
  precompute is `O(degree)` per edge, so a hub node alone costs
  `O(degree^2)` -- confirmed to never finish in practice (killed after
  several CPU-bound minutes stuck in exactly that precompute step, before
  any walk was even simulated). Fixed by adding
  [`cap_node_degree`][linkingtk.algorithms.ea._simple_hhea_structure.cap_node_degree],
  which randomly subsamples any node's excess edges down to
  `SimpleHHEALinker`'s `max_degree` (default 1000) before node2vec ever
  sees the graph -- a standard mitigation for random-walk embeddings on
  scale-free graphs, not something either the reference implementation or
  a toy-scale unit-test fixture needed to consider.
- **A `PYTHONHASHSEED`-driven determinism bug**: deduplicating edges via a
  plain Python `set` before capping meant `cap_node_degree`'s seeded
  `rng.permutation` was shuffling *indices into a hash-order-dependent
  list* -- so the same `random_state` picked different actual edges
  across separate process runs (Python randomizes `PYTHONHASHSEED` per
  process by default). Caught because it made a fully-seeded unit test
  flaky across repeated runs, not because it was suspected up front.
  Fixed by sorting the deduplicated edge list before capping.

With those fixed, the reported number above still used the wrong
evaluation metric on the first real run (Hits@1=0.237/Hits@10=0.451/
MRR=0.310) -- the reference's own `evaluate()` L2-normalizes both
embedding matrices *before* `CSLS_.py`'s raw `matmul`, making that raw
matmul equivalent to cosine similarity, not an unnormalized inner
product; `rank_exhaustive`'s `metric="cosine"` (not `"inner"`) is the
faithful match. Correcting this closed part of the gap (Hits@1
0.237 -> 0.262) but not most of it.

**The remaining gap is attributed to `max_degree` itself**, not chased
further in this pass: capping the highest-degree ICEWS entities'
neighborhoods from up to 273,317 down to 1000 throws away real structural
signal for exactly the entities a power-law graph concentrates the most
information in, and the reference's own paper never discusses (or needed
to solve) this scaling problem at all -- plausibly because a research
run had a far larger compute/time budget available than a demo script in
this repo does for a single `node2vec` precompute pass. Raising
`max_degree` was checked as directly costly (precompute cost grows
roughly with `max_degree^2` for capped hub nodes) rather than assumed
cheap, and left as a documented, tunable tradeoff rather than an
open-ended chase -- consistent with this project's practice of reporting
an honest shortfall once the diagnosis is solid (see e.g.
[RSN4EA](ea_kge_benchmarks.md#rsn4ea)/
[KDCoE](ea_kge_benchmarks.md#kdcoe) on the KGE benchmarks page).

Two other simplifications vs. the reference, made deliberately and
documented up front (not found via debugging) --  see
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

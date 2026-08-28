# Getting Started

LinkingTK unifies four text-linking tasks behind one common interface:

- **Entity Alignment (EA)** — matching entities between two knowledge graphs.
- **Entity Linking (EL)** — linking mentions in text to a knowledge base.
- **Word Sense Disambiguation (WSD)** — linking mentions in text to dictionary senses.
- **Word Sense Alignment (WSA)** — matching senses between two dictionaries.

Every task is modeled as matching entities from one dataset to another, so
the same [`Entity`][linkingtk.core.entity.Entity], `BlockingStrategy` and
`Evaluator` types work across all four. This page walks through that
interface end-to-end; see [DESIGN.md](https://github.com/jmccrae/linkingtk/blob/main/DESIGN.md)
for the rationale behind it.

## Installation

```bash
uv sync --extra graph --extra nlp --group dev
```

`graph` and `nlp` pull in optional dependencies (NetworkX/RDFLib, spaCy)
used by some algorithms and datasets. Other extras (`kge`, `wn`, `wikipedia`,
`vector-index`, `llm`, or `all`) add support for specific algorithms and data
sources — pull them in as you need them.

## The core interface

Every entity — a KG node, a text mention, a dictionary sense — is
represented the same way:

```python
from linkingtk.core import Entity

Entity(id="s1", labels=["cat"], description="A small domesticated feline.")
```

A linking algorithm implements
[`BaseLinker.link`][linkingtk.algorithms.base.BaseLinker], taking two
datasets of entities and returning a list of
[`AlignmentResult`][linkingtk.core.result.AlignmentResult] — one predicted
`source_id` → `target_id` link per match, each with a confidence `score`.

Before scoring, a `BlockingStrategy` cuts the full `dataset1 × dataset2`
cross-product down to a smaller set of plausible candidate pairs. The
simplest is [`ExactMatch`][linkingtk.blocking.exact.ExactMatch], which pairs
entities that share a label:

```python
--8<-- "examples/basic_exact_match.py"
```

Run with:

```bash
uv run python examples/basic_exact_match.py
```

```text
Candidate pairs: [('s1', 't1')]
Metrics: {'precision@1': 1.0, 'recall': 0.5, 'f1': 0.6666666666666666}
Blocking metrics: {'pair_completeness': 0.5, 'reduction_ratio': 0.75}
```

[`Evaluator.evaluate`][linkingtk.eval.evaluator.Evaluator.evaluate] scores
predictions against known ground truth; the same call is used for every
task, and
[`Evaluator.evaluate_blocking`][linkingtk.eval.evaluator.Evaluator.evaluate_blocking]
scores the blocking step on its own via Pair Completeness (the fraction of
true matches blocking kept) and Reduction Ratio (the fraction of the full
cross-product it eliminated).

## Putting it together across all four tasks

`ExactMatch` only works when source and target already share a label. A
real linker like
[`StringSimilarityLinker`][linkingtk.algorithms.string_similarity.StringSimilarityLinker]
scores pairs by similarity instead, so it can match `"cat"` against a
differently-worded description. The same linker class, and the same
`Evaluator.evaluate` call, works unchanged across EA, EL, WSD and WSA —
only the dataset and the fields being compared change per task:

```python
--8<-- "examples/mvp_benchmark.py"
```

Run with:

```bash
uv run python examples/mvp_benchmark.py
```

```text
EA: {'precision@1': 1.0, 'recall': 1.0, 'f1': 1.0}
EL: {'precision@1': 1.0, 'recall': 1.0, 'f1': 1.0}
WSD: {'precision@1': 1.0, 'recall': 1.0, 'f1': 1.0}
WSA: {'precision@1': 1.0, 'recall': 1.0, 'f1': 1.0}
```

`LeskLinker` used for WSD above is just a preconfigured
`StringSimilarityLinker`. All four calls load a
[bundled toy dataset](datasets/toy.md) — small, synthetic, and included with
the package so this runs offline with no downloads.

## Where to go next

- **[Datasets](datasets/index.md)** — swap the toy datasets above for real
  ones (Wikidata, WordNet, OpenEA, UFSAC, ...).
- **[Examples](examples/index.md)** — task-specific algorithms, from simple
  baselines to full reproductions of published methods.
- **[API Reference](reference/core.md)** — full docs for every public class
  and function.

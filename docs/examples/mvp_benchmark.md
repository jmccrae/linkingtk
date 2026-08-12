# Baseline across all four tasks

The same `StringSimilarityLinker` baseline (`LeskLinker` is a preconfigured
instance of it) run against a [bundled toy dataset](../datasets/toy.md) for
each of the four tasks — Entity Alignment, Entity Linking, Word Sense
Disambiguation and Word Sense Alignment — each scored with
[`Evaluator.evaluate`](../reference/eval.md). A good starting point for
seeing the common interface work end-to-end before moving on to the
task-specific examples.

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

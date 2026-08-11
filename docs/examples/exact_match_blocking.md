# Blocking and evaluation with ExactMatch

A minimal end-to-end example showing
[`ExactMatch`](../reference/blocking.md) blocking followed by
[`Evaluator.evaluate`](../reference/eval.md) against known ground truth, plus
[`Evaluator.evaluate_blocking`](../reference/eval.md) to assess the blocking
step itself — independent of any downstream linker — via Pair Completeness
(the fraction of true matches the blocking pass kept) and Reduction Ratio
(the fraction of the full cross-product it eliminated).

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

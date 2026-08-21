# GlossBERT: full-corpus training run

[`glossbert_benchmark.md`](glossbert_benchmark.md) verifies the training
path on a small, fast slice of SemCor. This script is the full-scale
counterpart: it trains
[`GlossBertEncoder`](../reference/algorithms.md) on the *entire* real
SemCor corpus (~230K sense-tagged instances) with near-paper-faithful
hyperparameters, then evaluates against UFSAC's real Raganato et al.
framework test sets -- directly comparable to
[`glossbert_reproduction.md`](glossbert_reproduction.md)'s checkpoint-based
numbers, but from a model this repo's own `CrossEncoderTrainer` trained,
not the paper's published weights.

**Long-running** -- meant to run for hours, unattended
(`nohup ... & disown`, not a foreground `uv run`). See the script's own
docstring for measured throughput and expected wall-clock time on a
single GPU.

```python
--8<-- "examples/glossbert_full_training.py"
```

Run with (from the repo root), detached so it survives the shell exiting:

```bash
nohup uv run python examples/glossbert_full_training.py \
    > /tmp/glossbert_full_training.log 2>&1 & disown
```

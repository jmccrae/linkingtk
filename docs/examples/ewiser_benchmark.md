# EWISER: verifying the training path

[`ewiser_reproduction.md`](ewiser_reproduction.md) validates
`EwiserEncoder`'s *inference* path by loading the paper's own published
checkpoints. It never touches
[`EwiserTrainer`](../reference/train.md) at all. This example is the
counterpart: it trains a fresh `bert-base-cased`
[`EwiserEncoder`](../reference/algorithms.md) from scratch via
`EwiserTrainer` on a real slice of SemCor, and checks that real learning
happens.

**Small subset, and not expected to approach published numbers** -- unlike
`ewiser_reproduction.md` (a hard acceptance gate for the checkpoint path),
this script has no such bar. EWISER's own published results critically
depend on initializing the output layer (`decoder.logits.weight`) from
externally pretrained LMMS/SensEmBERT sense embeddings; this package
doesn't bundle or reproduce those, so from-scratch training here starts
that layer randomly. The point is to verify the *training code itself* --
`EwiserTrainer`'s cross-entropy loss, per-sentence batching, and
freeze-then-thaw schedule -- end to end against a real pretrained encoder
and real data, not to reproduce the paper's numbers.

This is also the only place the freeze-then-thaw schedule
(`freeze_output_epochs`) gets exercised against a real training run, not
just synthetic-data unit tests
(`tests/algorithms/wsd/test_ewiser_trainer.py`).

```python
--8<-- "examples/ewiser_benchmark.py"
```

Run with:

```bash
uv run python examples/ewiser_benchmark.py
```

```text
2964 train instances (6 docs) / 669 eval instances (3 docs)
Most-frequent-sense baseline: {'precision@1': 0.205, 'recall': 0.205, 'f1': 0.205}
Vocabulary size: 1521
Untrained precision@1: {'precision@1': 0.428, 'recall': 0.428, 'f1': 0.428}
Per-epoch held-out Hits@1 (via EwiserTrainer.eval_history):
  epoch 1: {'Hits@1': 0.587, 'MRR': 0.740}
  epoch 2: {'Hits@1': 0.581, 'MRR': 0.732}
  epoch 3: {'Hits@1': 0.580, 'MRR': 0.733}
Trained precision@1 (via EwiserLinker.link, the real production path):
  {'precision@1': 0.580, 'recall': 0.580, 'f1': 0.580}
```

Real, meaningful learning on a real pretrained backbone: 42.8% (untrained,
already above the 20.5% most-frequent-sense floor purely from the graph
propagation and the encoder's own prior) → 58.7% (trained, after one
epoch on ~3K instances). `eval_history`'s per-epoch numbers match the
independent `EwiserLinker.link()` check exactly, confirming the training
path and the production inference path agree -- the same regression this
repo's own [`CrossEncoderTrainer`](../reference/train.md) history caught
a real bug on for GlossBERT (see
[`glossbert_benchmark.md`](glossbert_benchmark.md)).

Hits@1 dips slightly after the first epoch (58.7% → 58.0%) rather than
climbing further -- expected on a slice this small (~3K instances, no
pretrained sense-embedding init), not a sign of a training-path bug; the
freeze-then-thaw transition at epoch 1 (`decoder.logits.weight` unfreezes,
learning rate drops to `output_unfreeze_lr`) briefly perturbs a model
that had already found a good frozen-output-layer optimum on this small
slice.

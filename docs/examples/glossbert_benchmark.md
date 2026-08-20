# GlossBERT: verifying the training path

[`glossbert_reproduction.md`](glossbert_reproduction.md) validates
`GlossBertEncoder`'s *inference* path -- architecture, candidate
generation, text formatting -- by loading the paper's own published
checkpoint. It never touches `CrossEncoderTrainer` at all. This example
is the counterpart: it trains a fresh `bert-base-uncased`
[`GlossBertEncoder`](../reference/algorithms.md) from scratch via
[`CrossEncoderTrainer`](../reference/train.md) on a real slice of SemCor,
and checks that real learning happens.

**Small subset, not the paper's setup** -- the paper trains on all
~226K sense-tagged SemCor instances for 6 epochs; this trains on the
first 6 Brown Corpus documents (~3K instances) for 3, so it runs in well
under a minute. The point isn't to approach the published numbers (that's
what the checkpoint reproduction is for) -- it's to verify the *training
code itself* end to end: `CrossEncoderTrainer`'s BCE loss, hard-negative
mining via blocking, optimizer/warmup/weight-decay setup, all against a
real pretrained backbone and real data, not a toy synthetic model.

```python
--8<-- "examples/glossbert_benchmark.py"
```

Run with:

```bash
uv run python examples/glossbert_benchmark.py
```

```text
2964 train instances (6 docs) / 669 eval instances (3 docs)
Most-frequent-sense baseline: {'precision@1': 0.205, 'recall': 0.205, 'f1': 0.205}
Untrained precision@1: {'precision@1': 0.138, 'recall': 0.138, 'f1': 0.138}
Per-epoch held-out Hits@1 (via CrossEncoderTrainer.eval_history):
  epoch 1: {'Hits@1': 0.238, 'MRR': 0.423}
  epoch 2: {'Hits@1': 0.324, 'MRR': 0.490}
  epoch 3: {'Hits@1': 0.333, 'MRR': 0.496}
Trained precision@1 (via GlossBertLinker.link, the real production path):
  {'precision@1': 0.333, 'recall': 0.333, 'f1': 0.333}
```

Real, meaningful learning on a real pretrained backbone: 13.8% (untrained)
→ 33.3% (trained, 3 epochs on ~3K instances), clearing the most-frequent-sense
baseline (20.5%) by a wide margin.

## A real bug this caught

The first version of this benchmark reported a **60% Hits@1** from
`CrossEncoderTrainer.eval_history` -- but the same trained model, evaluated
independently via `GlossBertLinker.link()` on the identical held-out
mentions, measured only **33%**. Nearly 2x apart, on numbers that should
agree almost exactly.

The cause: `CrossEncoderTrainer._evaluate()` (mirroring
`Trainer._evaluate()`'s convention for EA/EL) derived its candidate pool
from `eval_data`'s own targets -- correct for EA/EL, where the eval
split's own KB subset already stands in for the full target set, but
silently wrong for WSD: the eval split's own gold senses are a far
narrower, far less confusable candidate pool than a mention's real
lemma-wide sense inventory. With only the *correct* answers ever offered
as candidates, the reported number was measuring something closer to
"can the model recognize its own training targets" than "can it
disambiguate."

Fixed by adding `CrossEncoderTrainer(..., eval_dataset2=...)`: pass the
real target set (here, the same `WnEntitySource` `link()` itself queries)
and `_evaluate()` blocks against that instead. After the fix,
`eval_history`'s last epoch (33.3%) matches the independent `link()`
check exactly -- see
[`CrossEncoderTrainer`](../reference/train.md)'s own docstring, and
`tests/train/test_cross_encoder.py::TestEvalDataset2` for the regression
coverage.

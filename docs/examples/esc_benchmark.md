# ESC: verifying the training path

[`esc_reproduction.md`](esc_reproduction.md) validates `EscEncoder`'s
*inference* path by loading the paper's own published checkpoint. It never
touches [`EscTrainer`](../reference/train.md) at all. This example is the
counterpart: it trains a fresh `facebook/bart-base`
[`EscEncoder`](../reference/algorithms.md) from scratch via `EscTrainer` on
a real slice of SemCor, and checks that real learning happens.

**Small subset and a smaller backbone, not expected to approach published
numbers** -- unlike `esc_reproduction.md` (a hard acceptance gate for the
checkpoint path), this script has no such bar. The reference's own full
recipe trains `facebook/bart-large` for far longer than a handful of
epochs on a 9-document slice, and also injects Poisson-sampled distractor
-gloss noise during training (`add_glosses_noise`) -- this uses
`facebook/bart-base` instead (same "smaller/faster backbone for the
training-path check" convention `ewiser_benchmark.md` already uses for
EWISER's `bert-base-cased` vs. the paper's `bert-large-cased`) and no
noise augmentation. The point is to verify the *training code itself* --
`EscTrainer`'s extractive-QA span-position loss, per-mention candidate
resolution/shuffling, and batching -- end to end against a real pretrained
encoder and real data, not to reproduce the paper's numbers.

```python
--8<-- "examples/esc_benchmark.py"
```

Run with:

```bash
uv run python examples/esc_benchmark.py
```

```text
2964 train instances (6 docs) / 669 eval instances (3 docs)
Most-frequent-sense baseline: {'precision@1': 0.205, 'recall': 0.205, 'f1': 0.205}
Untrained precision@1: {'precision@1': 0.172, 'recall': 0.172, 'f1': 0.172}
Per-epoch training loss (via EscTrainer.loss_history):
  epoch 1: 1.6357
  epoch 2: 1.1363
  epoch 3: 0.8366
  epoch 4: 0.6036
  epoch 5: 0.4051
Per-epoch held-out Hits@1 (via EscTrainer.eval_history):
  epoch 1: {'Hits@1': 0.426, 'MRR': 0.605}
  epoch 2: {'Hits@1': 0.495, 'MRR': 0.652}
  epoch 3: {'Hits@1': 0.505, 'MRR': 0.660}
  epoch 4: {'Hits@1': 0.505, 'MRR': 0.664}
  epoch 5: {'Hits@1': 0.489, 'MRR': 0.657}
Trained precision@1 (via EscLinker.link, the real production path):
  {'precision@1': 0.489, 'recall': 0.489, 'f1': 0.489}
```

Real, meaningful learning on a real pretrained backbone: the QA
span-position loss drops monotonically every epoch (1.64 → 0.41), and
held-out precision@1 climbs from 17.2% (untrained) to 48.9% (trained, 5
epochs on ~3K instances) -- clearing the most-frequent-sense baseline
(20.5%) by a wide margin. `eval_history`'s final epoch matches the
independent `EscLinker.link()` check exactly (48.9% both ways), confirming
the training path and the production inference path agree -- the same
regression this repo's own [`CrossEncoderTrainer`](../reference/train.md)
history caught a real bug on for GlossBERT (see
[`glossbert_benchmark.md`](glossbert_benchmark.md)).

Hits@1 dips slightly after the peak at epochs 3-4 (50.5% → 48.9%) --
expected noise from continuing to fine-tune a small backbone on a slice
this small (~3K instances, 5 epochs, no early stopping), not a sign of a
training-path bug; `loss_history` keeps decreasing every epoch throughout,
confirming the optimizer itself is behaving correctly even as held-out
generalization plateaus.

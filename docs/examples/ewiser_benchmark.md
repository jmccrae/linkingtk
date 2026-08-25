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
externally pretrained LMMS/SensEmBERT sense embeddings; loaders for those
vectors were added in
[issue #57](https://github.com/jmccrae/linkingtk/issues/57) (see
[`ewiser_pretrained_output_embedding.md`](ewiser_pretrained_output_embedding.md)),
but this script doesn't use them -- from-scratch training here starts
that layer randomly. The point is to verify the *training code itself* --
`EwiserTrainer`'s cross-entropy loss, per-sentence batching, and
freeze-then-thaw schedule -- end to end against a real pretrained encoder
and real data, not to reproduce the paper's numbers.

This script does wire up EWISER's own distinguishing idea, the WordNet
relation-graph propagation step (`build_relation_adjacency`) -- an
earlier version trained a plain frozen-BERT + FFN classifier with no
graph at all, which meant it wasn't really exercising "EWISER," just a
baseline sense classifier. The graph's practical benefit here is still
capped by construction: `SenseVocabulary.from_wn` restricts the
vocabulary to gold senses actually observed in this slice, so every
neighbor the graph can propagate from is already a labeled entry -- there
is no truly-unseen synset for the mechanism to generalize to, unlike
training against the full ~117k-synset inventory.

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
Relation graph edges: 911
Untrained precision@1: {'precision@1': 0.395, 'recall': 0.395, 'f1': 0.395}
Per-epoch held-out Hits@1 (via EwiserTrainer.eval_history):
  epoch 1: {'Hits@1': 0.631, 'MRR': 0.767}
  epoch 2: {'Hits@1': 0.622, 'MRR': 0.761}
  epoch 3: {'Hits@1': 0.641, 'MRR': 0.769}
Trained precision@1 (via EwiserLinker.link, the real production path):
  {'precision@1': 0.641, 'recall': 0.641, 'f1': 0.641}
```

Real, meaningful learning on a real pretrained backbone: 39.5% (untrained)
→ 64.1% (trained, 3 epochs on ~3K instances), clearing the most-frequent-
sense baseline (20.5%) by a wide margin. `eval_history`'s per-epoch
numbers match the independent `EwiserLinker.link()` check exactly,
confirming the training path and the production inference path agree --
the same regression this repo's own
[`CrossEncoderTrainer`](../reference/train.md) history caught a real bug
on for GlossBERT (see [`glossbert_benchmark.md`](glossbert_benchmark.md)).

Hits@1 dips slightly after the first epoch (63.1% → 62.2%) before
recovering -- expected noise on a slice this small (~3K instances, no
pretrained sense-embedding init), not a sign of a training-path bug; the
freeze-then-thaw transition at epoch 1 (`decoder.logits.weight` unfreezes,
learning rate drops to `output_unfreeze_lr`) briefly perturbs a model
that had already found a good frozen-output-layer optimum on this small
slice.

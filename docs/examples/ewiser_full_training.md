# EWISER: full-corpus training

[`ewiser_benchmark.md`](ewiser_benchmark.md) only verifies
`EwiserTrainer`'s training *code path* on a tiny slice (6 docs, ~3K
instances, a vocabulary restricted to gold senses observed in that
slice). This is the full-scale counterpart: the real 117,664-entry
WordNet vocabulary, a real from-scratch WordNet relation graph over that
whole vocabulary, and `decoder.logits.weight` initialized from the real
paper-best "LMMS + SensEmBERT" pretrained sense embeddings (issue #57)
rather than a random init -- trained on all 37,176 sentences of real
SemCor, evaluated on UFSAC's real Raganato et al. framework test sets.

**A real bug was fixed to make this possible** (issue #58): at the full
117,664-entry vocabulary, `StructuredLogits`'s graph-propagation step
with `structured_logits_trainable=True` (the default, and what the
reference's own `bin/train-ewiser.sh` uses in both training stages) used
to OOM -- `torch.sparse.mm`'s default backward materializes a dense
`[117664, 117664]` gradient (~51.6GB) instead of an `O(nnz)` one.
Confirmed directly: the OOM's own reported allocation size
(55,379,492,864 bytes) matches `117664**2 * 4 bytes` almost exactly, and
was invisible until now (`ewiser_benchmark.py`'s ~1,500-entry vocab
keeps that dense gradient to ~9MB). Fixed with `_SparsePropagate`, a
custom `torch.autograd.Function` computing the same gradient via an
`O(nnz)` indexed gather-multiply-sum -- see
[`_ewiser_structured_logits.py`](../reference/algorithms.md)'s own
module docstring, and its test file for a `gradcheck`-verified
correctness test. Measured directly on real hardware: 53.7ms per
8-sentence batch, 2.9GB peak GPU memory at the *full* vocabulary with a
trainable graph -- indistinguishable in cost from a non-trainable one.

**Acceptance threshold**: the closest real published number for this
exact training-data recipe (SemCor only, hypernymy-only relation graph)
is `ewiser.semcor_base.pt`'s own published ALL F1 of 77.0% -- the same
row [`ewiser_reproduction.md`](ewiser_reproduction.md) reproduced to
within 0.1 points against the released checkpoint. Not a hard bit-exact
gate here (this script trains its own relation graph from `wn` rather
than loading the checkpoint's baked-in one -- real, expected structural
differences, documented in `_ewiser_graph.py`'s own docstring). Two
checks, against measured ALL F1: a **floor** (must clear the
most-frequent-sense baseline by 10 points, or training is actually
broken) and a **target** (within 5 points of 77.0%, i.e. >= 72.0%,
reported but not asserted -- a multi-hour run's output shouldn't be
discarded on a near miss).

Requires UFSAC 2.1, EWISER's own `res/dictionaries/offsets.txt` and
`res/embeddings/sensembert+lmms.svd512.synset-centroid.vec` -- see the
script's own docstring for exact paths.

```python
--8<-- "examples/ewiser_full_training.py"
```

Run with (detached, since this takes hours):

```bash
nohup uv run python -u examples/ewiser_full_training.py \
    > /tmp/ewiser_full_training.log 2>&1 & disown
```

**Real run, this sandbox's single RTX 4090**: 70 epochs completed in
18,341s (~5.1 hours), matching the ~5-hour planning estimate closely.
Loss dropped from 4.99 (epoch 1) to 0.049 (epoch 70), with a visible
drop right at the freeze-to-thaw transition (epoch 51, lr 1e-4 -> 1e-5)
as the previously-frozen output layer started fine-tuning. Dev (SE07)
Hits@1 climbed from 0.669 (epoch 1) to a stable ~0.70-0.71 band by the
last ~15 epochs.

Final evaluation on the real held-out UFSAC test sets:

```text
dataset   precision@1   published (SemCor)
SE2             76.6%                77.5%
SE3             76.2%                77.9%
SE13            75.3%                76.4%
SE15            75.8%                77.8%
ALL             75.8%                77.0%
ALL F1 = 0.758
target met: ALL F1 within 0.05 of 0.77 (published ewiser.semcor_base.pt)
```

**Both acceptance checks passed.** The floor check (ALL F1 must clear
the most-frequent-sense baseline by 10 points) passed without needing to
print anything (an `assert`, not a verdict line). The target check
passed with real margin: 75.8% is 1.2 points short of the published
77.0%, well inside the 5-point (72.0%) bar -- 3.8 points of margin to
spare. The gap is consistent across every test set (0.9-2.0 points, no
outliers), matching what's expected from the already-documented
structural differences from the reference's own recipe (a from-scratch
WordNet relation graph via `wn`, not the checkpoint's baked-in
BabelNet-id-based one -- see `_ewiser_graph.py`'s own docstring) rather
than pointing at a bug to chase.

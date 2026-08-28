# ESC

[`EscLinker`](../reference/algorithms.md) ports
[Barba, Pasini & Navigli](https://www.aclweb.org/anthology/2021.naacl-main.371/)'s
ESC, which reframes WSD as extractive span comprehension over BART-large.
Two examples cover it: loading the paper's own checkpoint, and training
the encoder from scratch.

- **[Reproducing the paper's published results](#reproducing-the-papers-published-results)**
- **[Verifying the training path](#verifying-the-training-path)**

## Reproducing the paper's published results

Like [EWISER's reproduction](ewiser.md#reproducing-the-papers-published-results),
this runs **no training at all** -- see
[Verifying the training path](#verifying-the-training-path) for that side
instead. [`EscEncoder.from_checkpoint`](../reference/algorithms.md) loads
[Barba, Pasini & Navigli](https://www.aclweb.org/anthology/2021.naacl-main.371/)'s
own published SemCor checkpoint directly, and evaluates via
[`EscLinker.link`](../reference/algorithms.md), the exact same production
code path a freshly trained model would use.

ESC reframes WSD as extractive span comprehension: a mention's context
sentence (with the target word bracketed by literal `<classify>`/
`</classify>` markers) is paired with *every* candidate sense's gloss,
concatenated into one sequence, and a standard extractive-QA head
(`AutoModelForQuestionAnswering`, BART-large) predicts which candidate's
gloss span answers the "question" -- one forward pass per **mention**,
jointly observing every one of that mention's candidates at once. See
[`esc.py`](../reference/algorithms.md)'s module docstring for the real
architectural details confirmed directly from the checkpoint (the plain
`AutoModelForQuestionAnswering` path, not the reference's custom XLNet/
SQuAD-head/special-token variants; the full BART encoder-**decoder** stack,
not encoder-only).

Requires the paper's own SemCor checkpoint (linked from the [reference
repo's README](https://github.com/SapienzaNLP/esc#checkpoints)) at
`~/Downloads/escher_semcor_best.ckpt`, and
[UFSAC 2.1](https://github.com/getalp/UFSAC) extracted to
`~/data/ufsac-public-2.1/` -- see the script's own docstring. Candidates
are restricted to each mention's own tagged part of speech, matching the
reference's own `WordNetDataset.init_dataset` (confirmed directly against
`esc/esc_dataset.py`'s `wn_offsets_from_lemmapos`) -- the same convention
[EWISER's reproduction](ewiser.md#reproducing-the-papers-published-results)
already uses for EWISER's own candidate generation.

```python
--8<-- "examples/esc_reproduction.py"
```

Run with:

```bash
uv run python examples/esc_reproduction.py
```

```text
dataset   precision@1  published
SE07            76.3%      76.3%
SE2             81.5%        n/a
SE3             77.8%        n/a
SE13            82.2%        n/a
SE15            83.0%        n/a
ALL             80.6%      80.7%
```

**SE07 matches the published number exactly; `ALL` -- the paper's own
headline metric -- lands 0.1 points off.** Only SE07 and `ALL` are
published in the reference's own README; the other splits are reported
for reference without a stated target.

Unlike EWISER (#40) and GlossBERT (#39), this reproduction needed **no
bug hunting at all**: reading the checkpoint's own `hyper_parameters` and
`state_dict` directly (not just the paper text) identified the exact
architecture up front -- `facebook/bart-large`, `squad_head=False`,
`use_special_tokens=False`, the plain HF `AutoModelForQuestionAnswering`
path -- and porting the joint-sequence tokenization
([`build_joint_sequence`](../reference/algorithms.md)) against that
confirmed architecture landed within the checkpoint's own real numbers on
the first real run.

## Verifying the training path

[Reproducing the paper's published results](#reproducing-the-papers-published-results)
validates `EscEncoder`'s *inference* path by loading the paper's own
published checkpoint. It never touches [`EscTrainer`](../reference/train.md)
at all. This example is the counterpart: it trains a fresh
`facebook/bart-base` [`EscEncoder`](../reference/algorithms.md) from
scratch via `EscTrainer` on a real slice of SemCor, and checks that real
learning happens.

**Small subset and a smaller backbone, not expected to approach published
numbers** -- unlike [Reproducing the paper's published results](#reproducing-the-papers-published-results)
(a hard acceptance gate for the checkpoint path), this script has no such
bar. The reference's own full recipe trains `facebook/bart-large` for far
longer than a handful of epochs on a 9-document slice, and also injects
Poisson-sampled distractor-gloss noise during training
(`add_glosses_noise`) -- this uses `facebook/bart-base` instead (same
"smaller/faster backbone for the training-path check" convention
[EWISER's training verification](ewiser.md#verifying-the-training-path)
already uses for EWISER's `bert-base-cased` vs. the paper's
`bert-large-cased`) and no noise augmentation. The point is to verify the
*training code itself* -- `EscTrainer`'s extractive-QA span-position
loss, per-mention candidate resolution/shuffling, and batching -- end to
end against a real pretrained encoder and real data, not to reproduce the
paper's numbers.

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
[GlossBERT's training verification](glossbert.md#verifying-the-training-path)).

Hits@1 dips slightly after the peak at epochs 3-4 (50.5% → 48.9%) --
expected noise from continuing to fine-tune a small backbone on a slice
this small (~3K instances, 5 epochs, no early stopping), not a sign of a
training-path bug; `loss_history` keeps decreasing every epoch throughout,
confirming the optimizer itself is behaving correctly even as held-out
generalization plateaus.

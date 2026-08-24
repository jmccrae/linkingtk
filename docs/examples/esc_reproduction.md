# ESC: reproducing the paper's published results

Like [`ewiser_reproduction.md`](ewiser_reproduction.md), this runs **no
training at all** -- see [the training verification
example](esc_benchmark.md) for that side instead.
[`EscEncoder.from_checkpoint`](../reference/algorithms.md) loads
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
`ewiser_reproduction.py` already uses for EWISER's own candidate
generation.

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

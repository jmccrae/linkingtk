# Republishing UFSAC to the Hugging Face Hub

[`UfsacDataset`](../reference/datasets.md) normally requires a
manually-downloaded local UFSAC file, since UFSAC's whole collection is
a single Google Drive archive with no fetchable per-file URL. This
script uses the generic
[`linkingtk.datasets.hub_publish`](../reference/datasets.md) tooling to
give the corpora with a clear open license (`semcor.xml`, `wngt.xml`,
`masc.xml`, `omsti.xml`) a stable Hugging Face Hub URL instead --
`UfsacDataset("https://huggingface.co/datasets/linkingtk/ufsac/resolve/main/semcor.xml.xz")`
works the same as a local path once published.

The SensEval/SemEval all-words and lexical-sample corpora UFSAC also
bundles (`raganato_*.xml` and friends) are deliberately **never**
published this way -- their text is sourced from the LDC-licensed Wall
Street Journal / Penn Treebank corpus, the same tier as TAC KBP (see
[`ZeshelDataset`](../reference/datasets.md)'s docstring). Those still
need a local UFSAC download. See the script's own docstring for the
full per-corpus license reasoning.

It's a dry run by default -- prints the manifest of files it *would*
publish and the URLs they'd resolve to, with zero network calls. Pass
`--publish` (and an `HF_TOKEN`) to actually create/update the repo.

```python
--8<-- "examples/publish_ufsac.py"
```

Run with (from the repo root):

```bash
uv run python examples/publish_ufsac.py              # dry run
uv run python examples/publish_ufsac.py --publish     # actually publishes
```

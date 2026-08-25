"""Republishes UFSAC's clearly-redistributable corpora to the Hugging Face Hub.

[UfsacDataset][linkingtk.datasets.ufsac.UfsacDataset]'s `source` normally
has to be a manually-downloaded local path: UFSAC's whole 16+-corpus
collection is a single Google Drive archive with no fetchable per-file
URL (see its module docstring). This script uses
[linkingtk.datasets.hub_publish][] to give the corpora with a clear open
license a stable ``https://huggingface.co/datasets/...`` URL instead --
one that `source` (via
[fetch_cached][linkingtk.datasets._util.fetch_cached]) already accepts
directly.

**Only a hand-picked allowlist gets published, never the whole
archive.** UFSAC bundles corpora of very different provenance:

- `semcor.xml`, `wngt.xml` (WordNet Gloss Tagged), `masc.xml`,
  `omsti.xml` -- **included below**. Princeton WordNet's own permissive
  license (`wngt.xml`, and the WordNet-annotation layer of `semcor.xml`),
  CC-BY-3.0 (`masc.xml`, per https://www.anc.org/data/masc/), and
  CC-BY-4.0 (`omsti.xml`'s WordNet-sense mapping layer). SemCor's
  underlying Brown Corpus text has no clean modern open license, but has
  been freely redistributed in NLP tooling (e.g. NLTK) for decades.
- `senseval2*.xml`, `senseval3*.xml`, `semeval2007*.xml`,
  `semeval2013*.xml`, `semeval2015*.xml`, and every `raganato_*.xml` --
  **never published here.** These SensEval/SemEval all-words and
  lexical-sample task corpora are built over Wall Street Journal / Penn
  Treebank text, which is genuinely LDC-licensed (LDC99T42) -- the same
  tier as TAC KBP, which this project already deliberately skips rather
  than routes around (see ``linkingtk/datasets/zeshel.py``'s docstring).
  `UfsacDataset` still reads these directly from a local UFSAC download,
  same as today.
- `trainomatic.xml` -- **never published here** either; its own site
  only documents it as "available for research purposes," not a clear
  open redistribution license.

**Dry-run by default.** Without `--publish`, this only prints the
manifest of files it *would* upload and the URLs they'd resolve to --
zero network calls, safe to run without any Hugging Face credentials.
Pass `--publish` (and either `--token` or an `HF_TOKEN` environment
variable) to actually create/update the repo.

Requires UFSAC 2.1 extracted locally first (Google Drive, see
[UfsacDataset][linkingtk.datasets.ufsac.UfsacDataset]'s docstring) --
same `~/data/ufsac-public-2.1/` convention as
``examples/glossbert_full_training.py``.

Run with (from the repo root):
```
uv run python examples/publish_ufsac.py                    # dry run, prints the manifest
uv run python examples/publish_ufsac.py --publish           # actually publishes
```
"""

from __future__ import annotations

import argparse
from pathlib import Path

from linkingtk.datasets.hub_publish import package_files, publish_dataset_files, resolve_url

_DEFAULT_UFSAC_DIR = Path.home() / "data" / "ufsac-public-2.1"
_DEFAULT_REPO_ID = "linkingtk/ufsac"

# name -> why it's safe to republish (see the module docstring for the
# corpora deliberately left out, and why).
_INCLUDED_CORPORA = {
    "semcor.xml": "Princeton WordNet's annotation layer over the Brown Corpus",
    "wngt.xml": "WordNet Gloss Tagged -- Princeton WordNet's own permissive license",
    "masc.xml": "MASC -- CC-BY-3.0 (anc.org/data/masc)",
    "omsti.xml": "OMSTI -- CC-BY-4.0 WordNet-sense mapping layer",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ufsac-dir", type=Path, default=_DEFAULT_UFSAC_DIR)
    parser.add_argument("--repo-id", default=_DEFAULT_REPO_ID)
    parser.add_argument("--token", default=None, help="Defaults to the HF_TOKEN env var")
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Actually create/update the repo. Without this, only prints the manifest.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    files = package_files(args.ufsac_dir, _INCLUDED_CORPORA)

    if args.publish:
        published = publish_dataset_files(args.repo_id, files, token=args.token)
        print(f"Published to https://huggingface.co/datasets/{args.repo_id}:")
        for entry in published:
            print(f"  {entry.path_in_repo} -> {entry.url}")
    else:
        print(f"Dry run -- would publish to https://huggingface.co/datasets/{args.repo_id}:")
        for path_in_repo in files:
            print(f"  {path_in_repo} -> {resolve_url(args.repo_id, path_in_repo)}")
        print("Pass --publish to actually create/update the repo.")


if __name__ == "__main__":
    main()

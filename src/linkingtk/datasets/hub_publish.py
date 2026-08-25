"""Tooling to package and publish a linkingtk-maintained dataset copy to the Hugging Face Hub.

For datasets that are hard to load directly from their original source
(no fetchable per-file URL, an unreliable host, ...) but whose content is
clearly redistributable, this republishes it under a linkingtk-controlled
Hugging Face Hub dataset repo so a `DatasetLoader` can point `source` at
a stable ``https://huggingface.co/datasets/...`` URL the way
[fetch_cached][linkingtk.datasets._util.fetch_cached] already accepts --
see [UfsacDataset][linkingtk.datasets.ufsac.UfsacDataset] and
``examples/publish_ufsac.py`` for the one dataset this has actually been
applied to.

**This is not a license clearance tool.** Republishing under a
linkingtk-controlled repo is only appropriate for content that's
actually redistributable (a clear open license, or -- like
[AidaConllDataset][linkingtk.datasets.aida_conll.AidaConllDataset]'s and
[ZeshelDataset][linkingtk.datasets.zeshel.ZeshelDataset]'s community
mirrors -- content someone else has already cleared). Callers are
responsible for choosing which files to package; this module doesn't
inspect content or licenses. Genuinely non-redistributable data (e.g.
LDC-licensed corpora, per TAC KBP's precedent -- see ``zeshel.py``'s
docstring) should never be passed to
[publish_dataset_files][linkingtk.datasets.hub_publish.publish_dataset_files],
not even to a private repo.
"""

from __future__ import annotations

import lzma
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class HubApi(Protocol):
    """The subset of ``huggingface_hub.HfApi``'s interface this module needs.

    A real ``HfApi()`` instance already satisfies this directly (its
    ``create_repo``/``upload_file`` methods accept these same keyword
    arguments) -- this exists only so tests can inject a fake and never
    touch the network, the same dependency-injection convention as
    [fetch_wikipedia_extracts][linkingtk.datasets.aida_conll.fetch_wikipedia_extracts]'s
    ``description_fetcher``.
    """

    def create_repo(
        self,
        repo_id: str,
        *,
        repo_type: str,
        private: bool,
        token: str | None,
        exist_ok: bool,
    ) -> object: ...

    def upload_file(
        self,
        *,
        path_or_fileobj: bytes,
        path_in_repo: str,
        repo_id: str,
        repo_type: str,
        token: str | None,
    ) -> object: ...


@dataclass(frozen=True)
class PublishedFile:
    """One file's location within a published Hugging Face Hub dataset repo."""

    path_in_repo: str
    url: str


def resolve_url(repo_id: str, path_in_repo: str) -> str:
    """The stable download URL for a file in a Hugging Face Hub dataset repo.

    Same ``.../resolve/main/...`` shape
    [fetch_cached][linkingtk.datasets._util.fetch_cached] already accepts
    as a `DatasetLoader` ``source``.
    """
    return f"https://huggingface.co/datasets/{repo_id}/resolve/main/{path_in_repo}"


def package_files(
    source_dir: Path, filenames: Iterable[str], *, compress: bool = True
) -> dict[str, bytes]:
    """Read named files out of `source_dir`, optionally xz-compressing each.

    Args:
        source_dir: Directory each of `filenames` is read from directly
            (no recursion -- callers pass an explicit allowlist, not a
            directory sweep, so a new file dropped into `source_dir`
            never gets silently swept up).
        filenames: The files to package, relative to `source_dir`.
        compress: When true (the default), each file's bytes are
            xz-compressed and the output key gets a ``.xz`` suffix --
            matching UFSAC's own distributed compression, already
            handled by [UfsacDataset][linkingtk.datasets.ufsac.UfsacDataset]
            and any other loader that reads a ``.xz``-suffixed `source`.

    Returns:
        ``{output_filename: bytes}`` for each of `filenames`, ready to
        pass to [publish_dataset_files][linkingtk.datasets.hub_publish.publish_dataset_files].
    """
    packaged: dict[str, bytes] = {}
    for name in filenames:
        data = (source_dir / name).read_bytes()
        if compress:
            packaged[f"{name}.xz"] = lzma.compress(data)
        else:
            packaged[name] = data
    return packaged


def publish_dataset_files(
    repo_id: str,
    files: dict[str, bytes],
    *,
    token: str | None = None,
    private: bool = False,
    api: HubApi | None = None,
) -> list[PublishedFile]:
    """Create (or reuse) a Hugging Face Hub dataset repo and upload `files` to it.

    This always performs the real create-repo/upload-file calls -- there
    is no `dry_run` here. Callers that want a preview without touching
    the network should compute [resolve_url][linkingtk.datasets.hub_publish.resolve_url]
    for each file themselves instead of calling this function (see
    ``examples/publish_ufsac.py``'s ``--publish`` gate).

    Args:
        repo_id: The target Hugging Face Hub dataset repo, e.g.
            ``"linkingtk/ufsac"``.
        files: ``{path_in_repo: content_bytes}`` -- typically
            [package_files][linkingtk.datasets.hub_publish.package_files]'s
            return value.
        token: A Hugging Face Hub write token. Defaults to `None`, which
            lets `HfApi` fall back to its own ``HF_TOKEN`` environment
            variable / cached CLI login.
        private: Whether to create the repo as private, if it doesn't
            already exist.
        api: Overrides the Hugging Face Hub client. Defaults to a real
            ``huggingface_hub.HfApi()``, lazily imported (this package's
            other Hub-backed loaders import ``datasets`` the same way --
            see e.g. ``aida_conll.py``'s ``_build``) -- pass a fake here
            in tests.

    Returns:
        One [PublishedFile][linkingtk.datasets.hub_publish.PublishedFile]
        per input file, in `files`' iteration order.
    """
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()

    api.create_repo(repo_id, repo_type="dataset", private=private, token=token, exist_ok=True)

    published = []
    for path_in_repo, content in files.items():
        api.upload_file(
            path_or_fileobj=content,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
        )
        published.append(PublishedFile(path_in_repo, resolve_url(repo_id, path_in_repo)))
    return published

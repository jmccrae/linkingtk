from __future__ import annotations

import lzma
from pathlib import Path

from linkingtk.datasets.hub_publish import (
    PublishedFile,
    package_files,
    publish_dataset_files,
    resolve_url,
)


class _FakeHubApi:
    def __init__(self) -> None:
        self.create_repo_calls: list[dict[str, object]] = []
        self.upload_file_calls: list[dict[str, object]] = []

    def create_repo(
        self, repo_id: str, *, repo_type: str, private: bool, token: str | None, exist_ok: bool
    ) -> None:
        self.create_repo_calls.append(
            {
                "repo_id": repo_id,
                "repo_type": repo_type,
                "private": private,
                "token": token,
                "exist_ok": exist_ok,
            }
        )

    def upload_file(
        self,
        *,
        path_or_fileobj: bytes,
        path_in_repo: str,
        repo_id: str,
        repo_type: str,
        token: str | None,
    ) -> None:
        self.upload_file_calls.append(
            {
                "path_or_fileobj": path_or_fileobj,
                "path_in_repo": path_in_repo,
                "repo_id": repo_id,
                "repo_type": repo_type,
                "token": token,
            }
        )


class TestResolveUrl:
    def test_builds_the_resolve_main_url_shape(self) -> None:
        assert (
            resolve_url("linkingtk/ufsac", "semcor.xml.xz")
            == "https://huggingface.co/datasets/linkingtk/ufsac/resolve/main/semcor.xml.xz"
        )


class TestPackageFiles:
    def test_compresses_and_suffixes_by_default(self, tmp_path: Path) -> None:
        (tmp_path / "semcor.xml").write_text("<corpus/>")

        packaged = package_files(tmp_path, ["semcor.xml"])

        assert set(packaged) == {"semcor.xml.xz"}
        assert lzma.decompress(packaged["semcor.xml.xz"]) == b"<corpus/>"

    def test_compress_false_leaves_bytes_and_name_unchanged(self, tmp_path: Path) -> None:
        (tmp_path / "masc.xml").write_bytes(b"<corpus/>")

        packaged = package_files(tmp_path, ["masc.xml"], compress=False)

        assert packaged == {"masc.xml": b"<corpus/>"}

    def test_packages_multiple_files(self, tmp_path: Path) -> None:
        (tmp_path / "semcor.xml").write_text("a")
        (tmp_path / "masc.xml").write_text("b")

        packaged = package_files(tmp_path, ["semcor.xml", "masc.xml"])

        assert set(packaged) == {"semcor.xml.xz", "masc.xml.xz"}


class TestPublishDatasetFiles:
    def test_creates_repo_once_and_uploads_each_file(self) -> None:
        api = _FakeHubApi()

        publish_dataset_files(
            "linkingtk/ufsac",
            {"semcor.xml.xz": b"one", "masc.xml.xz": b"two"},
            token="tok",
            api=api,
        )

        assert len(api.create_repo_calls) == 1
        assert api.create_repo_calls[0] == {
            "repo_id": "linkingtk/ufsac",
            "repo_type": "dataset",
            "private": False,
            "token": "tok",
            "exist_ok": True,
        }
        assert [call["path_in_repo"] for call in api.upload_file_calls] == [
            "semcor.xml.xz",
            "masc.xml.xz",
        ]
        assert all(call["repo_id"] == "linkingtk/ufsac" for call in api.upload_file_calls)

    def test_returns_a_published_file_per_input_with_resolved_urls(self) -> None:
        api = _FakeHubApi()

        result = publish_dataset_files("linkingtk/ufsac", {"semcor.xml.xz": b"data"}, api=api)

        assert result == [
            PublishedFile(
                "semcor.xml.xz",
                "https://huggingface.co/datasets/linkingtk/ufsac/resolve/main/semcor.xml.xz",
            )
        ]

    def test_private_flag_is_forwarded_to_create_repo(self) -> None:
        api = _FakeHubApi()

        publish_dataset_files("linkingtk/ufsac", {}, private=True, api=api)

        assert api.create_repo_calls[0]["private"] is True

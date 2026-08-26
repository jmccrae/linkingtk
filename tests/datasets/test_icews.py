import zipfile
from pathlib import Path

from linkingtk.datasets.icews import IcewsWikiDataset

_FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


def _zip_url(tmp_path: Path, fixture_dir: str, folder: str) -> str:
    """Zip a fixtures/<fixture_dir> directory's files under <folder>/ in a tmp zip."""
    zip_path = tmp_path / "data.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        for file in (_FIXTURES_DIR / fixture_dir).iterdir():
            archive.write(file, arcname=f"{folder}/{file.name}")
    return f"file://{zip_path}"


class TestLoadTemporalGraphs:
    def test_resolves_timestamp_ids_to_year_month_labels(self, tmp_path: Path) -> None:
        dataset = IcewsWikiDataset(zip_url=_zip_url(tmp_path, "icews_temporal_toy", "icews_wiki"))
        graph1, graph2 = dataset.load_temporal_graphs()

        # triples_1's row references time_id 0/1 -> both resolve.
        assert graph1 == [("icews_wiki:1:0", "5", "icews_wiki:1:1", "2012-01", "2012-03")]

    def test_unresolvable_timestamp_falls_back_to_none(self, tmp_path: Path) -> None:
        dataset = IcewsWikiDataset(zip_url=_zip_url(tmp_path, "icews_temporal_toy", "icews_wiki"))
        graph1, graph2 = dataset.load_temporal_graphs()

        # triples_2's row references time_id 2 (resolvable) and 3 (a bare
        # negative year, "-400000" -- unresolvable per _TIME_LABEL_RE).
        assert graph2 == [("icews_wiki:2:50", "6", "icews_wiki:2:51", "2012-01", None)]

    def test_entity_ids_are_namespaced_like_load_graphs(self, tmp_path: Path) -> None:
        dataset = IcewsWikiDataset(zip_url=_zip_url(tmp_path, "icews_temporal_toy", "icews_wiki"))
        graph1, _ = dataset.load_temporal_graphs()

        assert graph1[0][0] == "icews_wiki:1:0"
        assert graph1[0][2] == "icews_wiki:1:1"

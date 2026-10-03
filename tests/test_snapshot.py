"""Tests for src/data/snapshot.py: checksums catch any change; the committed snapshot is intact (no network)."""
import pytest

from src.data import snapshot


def test_checksums_detect_changes(tmp_path):
    snap = tmp_path / "data" / "snapshot"
    snap.mkdir(parents=True)
    a = snapshot.write_text("observation_date,DGS10\n2024-09-30,3.81\n", "fred_DGS10.csv", snap)
    snapshot.write_text("x\n1\n", "auctions.csv", snap)
    ck = snap / "CHECKSUMS.sha256"
    snapshot.write_checksums(snapshot.snapshot_files(snap, extra=()), path=ck, root=tmp_path)
    assert "data/snapshot/fred_DGS10.csv" in ck.read_text()
    assert snapshot.verify_checksums(ck, root=tmp_path) == []

    a.write_text("observation_date,DGS10\n2024-09-30,3.82\n")
    snapshot.write_text("y\n", "new.csv", snap)
    (snap / "auctions.csv").unlink()
    problems = snapshot.verify_checksums(ck, root=tmp_path)
    assert "checksum mismatch data/snapshot/fred_DGS10.csv" in problems
    assert "missing file data/snapshot/auctions.csv" in problems
    assert "file not in checksums data/snapshot/new.csv" in problems


def test_vintage_render(tmp_path):
    v = snapshot.record_vintage({"auctions.csv": {"source": "Fiscal Data", "url": "https://x", "rows": 3,
                                                  "downloaded_utc": "2026-10-03T06:00:00Z"}},
                                {"auctions.csv": "note text"}, path=tmp_path / "vintage.json")
    md = (tmp_path / "VINTAGE.md").read_text()
    assert v["files"]["auctions.csv"]["rows"] == 3 and "| `auctions.csv` |" in md and "note text" in md


@pytest.mark.skipif(not snapshot.CHECKSUMS.exists(), reason="no committed snapshot")
def test_committed_snapshot_intact():
    assert snapshot.verify_checksums() == []

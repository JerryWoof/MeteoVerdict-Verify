"""verify_public.check_asset on a policy public-v1 row: public files checked against the
ledger, withheld files committed by hash and never in the tar."""

import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import verify_public as vp  # noqa: E402


def h(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def tar(members: dict) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as t:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            t.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def row_and_asset(extra=None):
    okresy, stations, obce, public_st = b"okresy", b"all stations", b"villages", b"synoptic"
    files = {"okresy.parquet": h(okresy), "stations.parquet": h(stations), "obce.parquet": h(obce)}
    manifest = json.dumps({"files": {n: {"sha256": s} for n, s in files.items()}}).encode()
    members = {"manifest.json": manifest, "okresy.parquet": okresy, "stations-public.parquet": public_st, **(extra or {})}
    data = tar(members)
    row = {"snapshot_time": "2026-10-23T00:00:00Z", "content_sha256": h(manifest), "files": files,
           "public": ["okresy.parquet", "stations-public.parquet"], "withheld": ["obce.parquet", "stations.parquet"],
           "derived": {"stations-public.parquet": {"sha256": h(public_st)}}, "register": None,
           "asset": {"tag": "snap-x", "name": "snap-x.tar", "sha256": h(data), "bytes": len(data)}}
    return row, data


def test_public_split_row_is_in_order(monkeypatch):
    row, data = row_and_asset()
    monkeypatch.setattr(vp, "fetch", lambda url, token=None: data)
    assert vp.check_asset(row, "x/y", None) == []


def test_a_withheld_file_in_the_tar_is_a_problem(monkeypatch):
    row, data = row_and_asset({"obce.parquet": b"villages"})
    monkeypatch.setattr(vp, "fetch", lambda url, token=None: data)
    problems = vp.check_asset(row, "x/y", None)
    assert any("withheld" in p for p in problems)


def test_a_changed_public_file_is_a_problem(monkeypatch):
    row, data = row_and_asset()
    row["derived"]["stations-public.parquet"]["sha256"] = "0" * 64
    monkeypatch.setattr(vp, "fetch", lambda url, token=None: data)
    assert any("public files" in p for p in vp.check_asset(row, "x/y", None))

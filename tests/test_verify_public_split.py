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


def station_file(ids) -> bytes:
    """A Parquet file with these station ids if pyarrow is installed, else placeholder bytes."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        return b"synoptic"
    buf = io.BytesIO()
    pq.write_table(pa.table({"station_id": ids}), buf)
    return buf.getvalue()


def row_and_asset(extra=None):
    okresy, stations, obce = b"okresy", b"all stations", b"villages"
    public_st = station_file(["0-20000-0-11518", "0-20000-0-11723"])
    files = {"okresy.parquet": h(okresy), "stations.parquet": h(stations), "obce.parquet": h(obce)}
    manifest = json.dumps({"files": {n: {"sha256": s} for n, s in files.items()}}).encode()
    members = {"manifest.json": manifest, "okresy.parquet": okresy, "stations-public.parquet": public_st, **(extra or {})}
    data = tar(members)
    row = {"snapshot_time": "2026-10-23T00:00:00Z", "content_sha256": "c" * 64, "files": files,
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


IDS = [f"0-20000-0-{n}" for n in (11518, 11723)]


def with_station_list(tmp_path, row, ids=IDS, version=1):
    reg = tmp_path / "register"
    reg.mkdir(exist_ok=True)
    (reg / "stations40.json").write_text(json.dumps({"version": version, "stations": [{"id": i} for i in ids]}))
    row["public_stations"] = {"version": 1, "count": len(IDS), "sha256": h(vp.canonical(sorted(IDS)))}
    return reg


def test_public_stations_match_the_fixed_list(monkeypatch, tmp_path):
    row, data = row_and_asset()
    reg = with_station_list(tmp_path, row)
    monkeypatch.setattr(vp, "fetch", lambda url, token=None: data)
    assert vp.check_asset(row, "x/y", reg) == []


def test_public_stations_other_than_the_list_are_a_problem(monkeypatch, tmp_path):
    row, data = row_and_asset()
    reg = with_station_list(tmp_path, row, ids=[*IDS, "0-20000-0-11624"])
    monkeypatch.setattr(vp, "fetch", lambda url, token=None: data)
    assert any("public_stations" in p for p in vp.check_asset(row, "x/y", reg))


def test_a_committed_file_neither_public_nor_withheld_is_a_problem(monkeypatch):
    row, data = row_and_asset()
    row["withheld"] = ["obce.parquet"]  # stations.parquet is committed but in neither list
    monkeypatch.setattr(vp, "fetch", lambda url, token=None: data)
    assert any("neither public nor withheld" in p for p in vp.check_asset(row, "x/y", None))


def test_the_village_register_is_not_looked_for_in_public_rows(monkeypatch, tmp_path):
    row, data = row_and_asset()
    row["register"] = {"version": "abc", "file_sha256": "1" * 64}
    monkeypatch.setattr(vp, "fetch", lambda url, token=None: data)
    assert vp.check_asset(row, "x/y", tmp_path) == []


def test_a_station_outside_the_list_in_the_public_file_is_a_problem(monkeypatch, tmp_path):
    import pytest

    pytest.importorskip("pyarrow")
    leak = station_file([*IDS, "0-203-0-11603"])
    row, data = row_and_asset()
    with tarfile.open(fileobj=io.BytesIO(data)) as t:
        members = {m.name: t.extractfile(m).read() for m in t.getmembers()}
    members["stations-public.parquet"] = leak
    data = tar(members)
    row["derived"]["stations-public.parquet"]["sha256"] = h(leak)
    row["asset"] |= {"sha256": h(data), "bytes": len(data)}
    reg = with_station_list(tmp_path, row)
    monkeypatch.setattr(vp, "fetch", lambda url, token=None: data)
    assert any("outside the public list" in p for p in vp.check_asset(row, "x/y", reg))


def test_public_v2_withheld_weights_are_committed_by_hash(monkeypatch):
    """public-v2: blend_weights.parquet is withheld (hash in the row, not in the tar)."""
    row, data = row_and_asset()
    row["policy"] = "public-v2"
    row["files"]["blend_weights.parquet"] = h(b"weights")
    row["withheld"].append("blend_weights.parquet")
    # the public manifest lists every committed file, the tar has no weights
    members = {"manifest.json": json.dumps({"files": {n: {"sha256": s} for n, s in row["files"].items()}}).encode(),
               "okresy.parquet": b"okresy", "stations-public.parquet": station_file(IDS)}
    data = tar(members)
    row["asset"].update(sha256=h(data), bytes=len(data))
    row["derived"]["stations-public.parquet"]["sha256"] = h(members["stations-public.parquet"])
    monkeypatch.setattr(vp, "fetch", lambda url, token=None: data)
    assert vp.check_asset(row, "x/y", None) == []
    members["blend_weights.parquet"] = b"weights"
    data = tar(members)
    row["asset"].update(sha256=h(data), bytes=len(data))
    monkeypatch.setattr(vp, "fetch", lambda url, token=None: data)
    assert any("withheld" in p for p in vp.check_asset(row, "x/y", None))


def test_withdrawn_tags_are_read(tmp_path):
    assert vp.read_withdrawn(tmp_path) == {}
    (tmp_path / "withdrawn.json").write_text(json.dumps({"tags": {"snap-x": "weights"}}))
    assert vp.read_withdrawn(tmp_path) == {"snap-x": "weights"}

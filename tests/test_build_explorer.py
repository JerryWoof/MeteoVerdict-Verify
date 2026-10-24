"""build_explorer: one JSON per district and station from a snapshot's public files."""

import io
import json
import sys
import tarfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_explorer as be  # noqa: E402

T0 = datetime(2026, 10, 24, 0, tzinfo=UTC)
SID = "0-20000-0-11518"


def pq(df: pl.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.write_parquet(buf)
    return buf.getvalue()


def make_tar(members: dict) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as t:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            t.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def test_build_writes_district_and_station_files(tmp_path):
    times = [T0 + timedelta(hours=h) for h in range(3)]
    allv = pl.DataFrame({"issued": [T0] * 6, "okres": ["CZ0100"] * 6, "series": ["ai"] * 3 + ["m"] * 3,
                         "variable": ["pressure_msl"] * 6, "time": times * 2, "value": [1000.0, 1001.0, 1002.0, 999.0, 998.0, 997.0]})
    hourly = pl.DataFrame({"okres": ["CZ0100"] * 3, "variable": ["precipitation"] * 3, "time": times,
                           "raw": [0.1] * 3, "value": [0.1, 0.2, 0.3], "issued": [T0] * 3})
    models = pl.DataFrame({"model": ["ai", "m"], "station_id": [SID] * 2, "valid_time": [times[0]] * 2, "run": [T0] * 2,
                           "temperature_2m": [1.0, 2.0]})
    # 10-minute temperature at the full hour and six precipitation values ending at 01:00
    obs_rows = [(SID, T0 - timedelta(hours=1) + timedelta(minutes=10 * k), "10min", "SRA10M", 0.1, None, 0) for k in range(7)]
    obs_rows += [(SID, T0, "10min", "T", 5.5, None, 0), (SID, T0, "10min", "T", 99.0, None, 4)]
    obs = pl.DataFrame(obs_rows, schema={"station_id": pl.String, "time": pl.Datetime("us", "UTC"), "step": pl.String,
                                         "element": pl.String, "value": pl.Float32, "flag": pl.String, "quality": pl.Int8}, orient="row")
    data = make_tar({"okresy_all.parquet": pq(allv), "hourly.parquet": pq(hourly), "stations_models-public.parquet": pq(models),
                     "obs-public.parquet": pq(obs), "manifest.json": b"{}"})
    row = {"snapshot_time": "2026-10-24T00:00:00Z", "issue_time": "x", "model_version": "v", "asset": {}, "chain_hash": "h",
           "files": {}, "kind": "live", "policy": "public-v2"}
    stations = {"stations": [{"id": SID, "name": "Praha", "lat": 50.1, "lon": 14.2, "elev": 364, "okres": "Praha"}]}
    index = be.build(row, data, stations, tmp_path)
    okres = json.loads((tmp_path / "okresy" / "CZ0100.json").read_text())
    assert okres["variables"]["pressure_msl"]["ai"] == [1000.0, 1001.0, 1002.0]
    assert okres["variables"]["precipitation"]["ai"] == [0.1, 0.2, 0.3]  # the AI's hourly values fill the gap
    st = json.loads((tmp_path / "stations" / f"{SID}.json").read_text())
    assert st["variables"]["temperature_2m"]["m"][0] == 2.0
    assert st["variables"]["temperature_2m"]["ai"][0] == 1.0
    assert st["obs"]["series"]["temperature_2m"] == [5.5]  # quality 4 dropped
    assert index["okresy"] == ["CZ0100"] and "precipitation" in index["variables"]

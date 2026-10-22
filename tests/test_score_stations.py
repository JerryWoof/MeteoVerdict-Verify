"""Tests of scripts/score_stations.py and scripts/fetch_obs.py on a small synthetic snapshot.

    pip install polars pytest && python -m pytest tests
"""

import io
import json
import sys
import tarfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import fetch_obs  # noqa: E402
import score_stations as ss  # noqa: E402
import verify_public as vp  # noqa: E402

UTC = timezone.utc
SNAP = datetime(2026, 10, 20, 0, 0, tzinfo=UTC)
ST = "0-20000-0-11406"


def tar_of(files: dict[str, pl.DataFrame]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for name, df in files.items():
            b = io.BytesIO()
            df.write_parquet(b)
            info = tarfile.TarInfo(name)
            info.size = len(b.getvalue())
            tar.addfile(info, io.BytesIO(b.getvalue()))
    return buf.getvalue()


def synthetic(tmp_path: Path):
    """One snapshot, one station, 30 hours at lead 24-53 h (D+1 and a few D+2); observed T = 10 °C.
    AI is +1 °C everywhere, model A +2 °C everywhere, model B -3 °C but missing on 5 hours, model C
    -0.5 °C from a run that was not yet available (R1), model D +0.1 °C but with only 3 hours."""
    hours = [SNAP + timedelta(hours=24 + i) for i in range(30)]
    issued = SNAP - timedelta(hours=1)
    ai = pl.DataFrame({"issued": [issued] * 30, "station_id": [ST] * 30, "variable": ["temperature_2m"] * 30,
                       "time": hours, "value": [11.0] * 30}).with_columns(pl.col("value").cast(pl.Float32))
    rows = []
    for i, t in enumerate(hours):
        rows.append(("model_a", t, SNAP - timedelta(hours=12), 12.0))
        if i >= 5:
            rows.append(("model_b", t, SNAP - timedelta(hours=12), 7.0))
        rows.append(("ecmwf_ifs", t, SNAP - timedelta(hours=3), 9.5))  # delay 7 h: not available at SNAP
        if i < 3:
            rows.append(("model_d", t, SNAP - timedelta(hours=12), 10.1))
    sm = pl.DataFrame(rows, schema=["model", "valid_time", "run", "temperature_2m"], orient="row").with_columns(
        pl.lit(ST).alias("station_id"), pl.col("temperature_2m").cast(pl.Float32),
        *[pl.lit(None, pl.Float32).alias(c) for c in ("precipitation", "wind_speed_10m", "wind_gusts_10m", "cloud_cover", "sunshine_duration")])
    data = tar_of({"stations.parquet": ai, "stations_models.parquet": sm})
    obs = pl.DataFrame({"station_id": [ST] * 48, "time": [(SNAP + timedelta(hours=24 + i)).strftime("%Y-%m-%dT%H:%M:%SZ") for i in range(48)],
                        "temperature_2m": [10.0] * 48})
    return data, hours, obs


def test_reduce_and_score_common_pairs_and_r1(tmp_path):
    data, hours, obs_df = synthetic(tmp_path)
    frame = ss.reduce_snapshot(data, SNAP, [ST])
    assert set(frame["series"]) == {"ai", "model_a", "model_b", "ecmwf_ifs", "model_d"}
    obs = obs_df.with_columns(pl.col("time").str.to_datetime("%Y-%m-%dT%H:%M:%SZ", time_zone="UTC").dt.cast_time_unit("us").alias("valid_time"),
                              pl.lit("temperature_2m").alias("variable"), pl.lit(10.0).alias("obs")).select("station_id", "valid_time", "variable", "obs")
    out = ss.score([frame], obs, [{"id": ST, "name": "Cheb"}], hours[-1])
    o = out["overall"]["temperature_2m|1"]  # leads 24..47 h = hours[0:24]
    assert o["ai_pairs"] == 24
    assert o["dropped_run_unavailable"] == {"ecmwf_ifs": 24}  # R1: dropped, not replaced by another run
    pw = {m["k"]: m for m in o["models"]}
    assert "ecmwf_ifs" not in pw
    assert pw["model_a"]["pairs"] == 24 and pw["model_a"]["mae"] == 2.0 and pw["model_a"]["bias"] == 2.0
    assert pw["model_b"]["pairs"] == 19 and pw["model_b"]["bias"] == -3.0  # 5 hours without B: not filled
    assert pw["model_d"]["pairs"] == 3
    assert pw["model_b"]["ai_mae"] == 1.0
    # per station: ranked models need >= MIN_N pairs (D has 3), the top two are A and B (no third)
    s = out["data"][f"{ST}|temperature_2m|1"]
    assert [x["k"] for x in s["stats"]] == ["ai", "model_a", "model_b"]
    assert s["n"] == 19  # common pairs of AI, A and B only
    assert s["stats"][0] == {"k": "ai", "n": "MeteoVerdict AI", "mae": 1.0, "bias": 1.0, "rmse": 1.0}
    assert len(s["obs"]) == 48 and len(s["ai"]) == 48 and s["top"][0]["n"] == NAME_A


NAME_A = "model_a"


def test_chain_must_be_intact(tmp_path):
    ledger = tmp_path / "ledger"
    ledger.mkdir()
    row = {"snapshot_time": "2026-10-20T00:00:00Z", "status": "missing", "asset": None, "prev_hash": vp.GENESIS}
    row["chain_hash"] = vp.chain_hash(vp.GENESIS, row)
    (ledger / "2026-10-20.jsonl").write_text(json.dumps(row) + "\n")
    assert vp.check_chain(vp.read_ledger(ledger)) == []
    row["status"] = "ok"
    (ledger / "2026-10-20.jsonl").write_text(json.dumps(row) + "\n")
    assert vp.check_chain(vp.read_ledger(ledger))


def test_hourly_from_ten_minute_rows():
    day = datetime(2026, 10, 21, tzinfo=UTC).date()
    rows = []
    for m in range(-60, 24 * 60 + 1, 10):  # from 23:00 the day before to 00:00 the day after
        t = datetime(2026, 10, 21, tzinfo=UTC) + timedelta(minutes=m)
        s = t.strftime("%Y-%m-%dT%H:%M:%SZ")
        rows += [[ST, "T", s, 5.0 + t.hour, "", 0.0], [ST, "F", s, 2.0, "", 0.0],
                 [ST, "SRA10M", s, 0.1, "", 0.0], [ST, "Fmax", s, 3.0 + t.minute / 10, "", 0.0]]
    # hour 03:00: one rain slot with a bad quality code (4) -> the hour has no precipitation
    rows = [r for r in rows if not (r[1] == "SRA10M" and r[2] == "2026-10-21T02:30:00Z")]
    h = fetch_obs.hourly(rows, day)
    t3 = datetime(2026, 10, 21, 3, tzinfo=UTC)
    assert h[t3]["temperature_2m"] == 8.0 and h[t3]["wind_speed_10m"] == 2.0
    assert "precipitation" not in h[t3]
    assert round(h[t3 + timedelta(hours=1)]["precipitation"], 1) == 0.6
    assert h[t3]["wind_gusts_10m"] == 3.0 + 5  # largest of 02:10 .. 03:00 is the 02:50 slot (5) -> 8.0
    assert round(h[datetime(2026, 10, 21, 0, tzinfo=UTC)]["precipitation"], 1) == 0.6  # slots of the day before count


def test_public_v1_files_are_read(tmp_path):
    data, _, _ = synthetic(tmp_path)
    with tarfile.open(fileobj=io.BytesIO(data)) as t:
        members = {m.name.replace(".parquet", "-public.parquet"): t.extractfile(m).read() for m in t.getmembers()}
    renamed = io.BytesIO()
    with tarfile.open(fileobj=renamed, mode="w") as t:
        for name, b in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(b)
            t.addfile(info, io.BytesIO(b))
    assert ss.reduce_snapshot(renamed.getvalue(), SNAP, [ST]).equals(ss.reduce_snapshot(data, SNAP, [ST]))

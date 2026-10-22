"""Tests of scripts/score_stations.py and scripts/fetch_obs.py on a small synthetic snapshot.

    pip install polars pytest && python -m pytest tests
"""

import io
import json
import sys
import tarfile
from datetime import date, datetime, timedelta, timezone
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
def station_table(hours, extra_gust=True):
    """stations_models of the new layout: ai, blend and two models (A, B), temperature, rain, gusts."""
    rows = []
    for i, h in enumerate(hours):
        rows.append(("ai", h, SNAP - timedelta(hours=1), 11.0, 1.0, None))
        rows.append(("blend", h, SNAP - timedelta(hours=1), 10.5, 0.5, None))
        rows.append(("model_a", h, SNAP - timedelta(hours=12), 12.0, 0.0, 6.0 + (i == 8) * 4))
        rows.append(("model_b", h, SNAP - timedelta(hours=12), 7.0, 2.0, 5.0))
    df = pl.DataFrame(rows, schema=["model", "valid_time", "run", "temperature_2m", "precipitation", "wind_gusts_10m"], orient="row")
    return df.with_columns(pl.lit(ST).alias("station_id"), *[pl.col(c).cast(pl.Float32) for c in ("temperature_2m", "precipitation", "wind_gusts_10m")],
                           *[pl.lit(None, pl.Float32).alias(c) for c in ("wind_speed_10m", "cloud_cover", "sunshine_duration")])


def test_blend_and_new_ai_variables_on_common_pairs():
    hours = [SNAP + timedelta(hours=24 + i) for i in range(30)]
    sm = station_table(hours).filter(~((pl.col("model") == "blend") & (pl.col("valid_time") == hours[0])))  # blend missing at one hour
    parts = pl.DataFrame({"station_id": [ST], "day": [date(2026, 10, 21)], "part": ["afternoon"], "gust": [9.0]}, schema_overrides={"gust": pl.Float32})
    frame = ss.reduce_snapshot(tar_of({"stations_models.parquet": sm, "stations_parts.parquet": parts}), SNAP, [ST])
    assert set(frame["series"]) == {"ai", "blend", "model_a", "model_b"}  # the AI comes from the table, no stations.parquet needed
    assert {"temperature_2m", "precipitation", ss.PART_GUST} <= set(frame["variable"])
    # models' gusts per part (Prague parts, hour-ending values); the spike is in the morning of 21 Oct
    g = frame.filter(pl.col("variable") == ss.PART_GUST, pl.col("series") == "model_a")
    assert g["value"].max() == 10.0 and set(g["valid_time"].dt.hour()) == {4, 10, 16, 22}  # part starts in UTC (summer time +2 h)
    assert frame.filter(pl.col("variable") == ss.PART_GUST, pl.col("series") == "ai")["value"].to_list() == [9.0]
    obs = pl.DataFrame({"station_id": [ST] * 30 * 2, "variable": ["temperature_2m"] * 30 + ["precipitation"] * 30,
                        "valid_time": hours * 2, "obs": [10.0] * 30 + [0.2] * 30}).with_columns(pl.col("valid_time").dt.cast_time_unit("us"))
    out = ss.score([frame], obs, [{"id": ST, "name": "Cheb"}], hours[-1])
    t = out["overall"]["temperature_2m|1"]
    assert t["ai_pairs"] == 24  # the pooled statistics need 30 pairs per model: no common set here
    s = out["data"][f"{ST}|temperature_2m|1"]
    assert s["with_blend"] and s["n"] == 23  # the hour without blend leaves the common set
    assert [x["k"] for x in s["stats"]] == ["ai", "blend", "model_a", "model_b"]
    assert [x["mae"] for x in s["stats"]] == [1.0, 0.5, 2.0, 3.0] and len(s["blend"]) == 48
    assert t["blend"]["pairs"] == 23 and t["blend"]["mae"] == 0.5 and t["blend"]["ai_mae"] == 1.0
    p = out["overall"]["precipitation|1"]
    assert {m["k"]: m["mae"] for m in p["models"]} == {"model_a": 0.2, "model_b": 1.8} and p["blend"]["mae"] == 0.3
    assert "wind_gusts_10m|1" not in out["overall"]


def test_old_snapshot_without_blend_still_scores():
    data, hours, _ = synthetic(Path("."))
    frame = ss.reduce_snapshot(data, SNAP, [ST])
    obs = pl.DataFrame({"station_id": [ST] * 30, "variable": ["temperature_2m"] * 30, "valid_time": hours, "obs": [10.0] * 30}).with_columns(pl.col("valid_time").dt.cast_time_unit("us"))
    out = ss.score([frame], obs, [{"id": ST, "name": "Cheb"}], hours[-1])
    assert out["overall"]["temperature_2m|1"]["blend"] is None and not out["overall"]["temperature_2m|1"]["with_blend"]


def test_hourly_cloud_and_sunshine_from_hourly_rows():
    day = datetime(2026, 10, 21, tzinfo=UTC).date()
    rows = [[ST, "N", "2026-10-21T10:00:00Z", 3.0, "", 5.0], [ST, "N", "2026-10-21T11:00:00Z", 8.0, "", 5.0],
            [ST, "N", "2026-10-21T12:00:00Z", 2.0, "", 4.0],  # missing (quality 4)
            [ST, "SSV1H", "2026-10-21T11:00:00Z", 5.0, "", 0.0], [ST, "SSV1H", "2026-10-21T12:00:00Z", 12.0, "", 5.0],
            [ST, "N", "2026-10-22T00:00:00Z", 1.0, "", 5.0]]  # another day
    h = fetch_obs.hourly_sky(rows, day)
    t10, t11, t12 = (datetime(2026, 10, 21, x, tzinfo=UTC) for x in (10, 11, 12))
    assert h[t10] == {"cloud_cover": 37.5} and h[t11] == {"cloud_cover": 100.0, "sunshine_duration": 1800.0}
    assert h[t12] == {"sunshine_duration": 3600.0} and len(h) == 3


def test_cloud_and_sunshine_scored_and_old_obs_files_still_load(tmp_path):
    hours = [SNAP + timedelta(hours=24 + i) for i in range(30)]
    rows = [(m, h, SNAP - timedelta(hours=1 if m in ("ai", "blend") else 12), c, s) for m, c, s in
            (("ai", 50.0, 1000.0), ("model_a", 70.0, 2000.0)) for h in hours]
    sm = pl.DataFrame(rows, schema=["model", "valid_time", "run", "cloud_cover", "sunshine_duration"], orient="row").with_columns(
        pl.lit(ST).alias("station_id"), pl.col("cloud_cover").cast(pl.Float32), pl.col("sunshine_duration").cast(pl.Float32),
        *[pl.lit(None, pl.Float32).alias(c) for c in ("temperature_2m", "precipitation", "wind_speed_10m", "wind_gusts_10m")])
    frame = ss.reduce_snapshot(tar_of({"stations_models.parquet": sm}), SNAP, [ST])
    obs_dir = tmp_path / "obs"
    obs_dir.mkdir()
    (obs_dir / "obs-20261020.csv").write_text("station_id,time,temperature_2m,wind_speed_10m,precipitation,wind_gusts_10m\n"
                                               f"{ST},2026-10-21T00:00:00Z,10.0,2.0,0.0,5.0\n")  # an old file without the new columns
    lines = ["station_id,time,temperature_2m,wind_speed_10m,precipitation,wind_gusts_10m,cloud_cover,sunshine_duration"]
    lines += [f"{ST},{h:%Y-%m-%dT%H:%M:%SZ},10.0,2.0,0.0,5.0,60.0,1500.0" for h in hours]
    (obs_dir / "obs-20261021.csv").write_text("\n".join(lines) + "\n")
    obs, _ = ss.load_obs(obs_dir)
    out = ss.score([frame], obs, [{"id": ST, "name": "Cheb"}], hours[-1])
    c = out["data"][f"{ST}|cloud_cover|1"]
    assert [x["k"] for x in c["stats"]] == ["ai", "model_a"] and [x["mae"] for x in c["stats"]] == [10.0, 10.0]
    assert out["data"][f"{ST}|sunshine_duration|1"]["stats"][0]["mae"] == 500.0


def _obs_csv(path, value):
    day = path.name[4:12]  # the file's own day, as the real files have
    path.write_text(f"station_id,time,temperature_2m\n{ST},{day[:4]}-{day[4:6]}-{day[6:8]}T00:00:00Z,{value}\n")


def test_checked_file_is_preferred_per_day_and_labelled(tmp_path):
    obs = tmp_path / "obs"
    obs.mkdir()
    _obs_csv(obs / "obs-20261020.csv", 1.0)
    _obs_csv(obs / "obs-20261021.csv", 2.0)
    _obs_csv(obs / "obs-20261021.final.csv", 3.0)
    _obs_csv(obs / "obs-20261021.final2.csv", 4.0)
    files, quality = ss.obs_files(obs)
    assert [f.name for f in files] == ["obs-20261020.csv", "obs-20261021.final2.csv"]
    assert quality == {"20261020": "provisional", "20261021": "final"}
    frame, sums = ss.load_obs(obs)
    assert frame.filter(frame["valid_time"] == datetime(2026, 10, 21, tzinfo=timezone.utc))["obs"].to_list() == [4.0]
    assert len(sums) == 2


def test_final_day_never_overwrites_and_versions_changes(tmp_path, monkeypatch):
    import argparse

    obs = tmp_path / "obs"
    obs.mkdir()
    stations = tmp_path / "stations.json"
    stations.write_text(json.dumps({"stations": [{"id": ST}]}))
    args = argparse.Namespace(out=obs, stations=stations)
    day = date(2026, 10, 21)
    today = date(2026, 11, 30)
    assert fetch_obs.final_day(args, day, today) == 2  # no provisional file yet
    _obs_csv(obs / "obs-20261021.csv", 2.0)
    monkeypatch.setattr(fetch_obs, "day_rows", lambda *a, **k: ([], "none"))
    assert fetch_obs.final_day(args, day, today) == 2  # recent/ has not got the day
    value = {"v": 3.0}
    monkeypatch.setattr(fetch_obs, "day_rows", lambda *a, **k: ([["x"]], "recent"))
    cell = lambda: {datetime(2026, 10, 21, h, tzinfo=timezone.utc): {"temperature_2m": value["v"]} for h in range(24)}  # noqa: E731
    monkeypatch.setattr(fetch_obs, "fetch_day", lambda ids, d, t, checked_only=False: ({ST: cell()}, {ST: "recent"}))
    monkeypatch.setattr(fetch_obs, "to_csv", lambda ids, hours: f"v={value['v']}\n".encode())
    assert fetch_obs.final_day(args, day, today) == 2  # fewer than 35 stations: nothing written
    monkeypatch.setattr(fetch_obs, "final_day", fetch_obs.final_day)
    # one station only: lower the bar through a 1-station register to exercise the writing path
    monkeypatch.setattr(fetch_obs, "fetch_day", lambda ids, d, t, checked_only=False: ({i: cell() for i in ids}, {}))
    stations.write_text(json.dumps({"stations": [{"id": f"S{i}"} for i in range(40)]}))
    assert fetch_obs.final_day(args, day, today) == 0
    assert (obs / "obs-20261021.final.csv").read_text() == "v=3.0\n"
    assert (obs / "obs-20261021.csv").read_text().endswith("2.0\n")  # provisional untouched
    assert fetch_obs.final_day(args, day, today) == 0 and not (obs / "obs-20261021.final2.csv").exists()
    value["v"] = 3.5  # checked value changed later
    assert fetch_obs.final_day(args, day, today) == 0
    assert (obs / "obs-20261021.final.csv").read_text() == "v=3.0\n" and (obs / "obs-20261021.final2.csv").read_text() == "v=3.5\n"
    assert "obs-20261021.final2.csv" in (obs / "SHA256SUMS").read_text()


def test_r1_uses_measured_availability_and_falls_back_to_the_delay_table():
    snap = datetime(2026, 10, 22, 12, tzinfo=timezone.utc)

    def row(series, run_h, avail_h):
        run = datetime(2026, 10, 22, run_h, tzinfo=timezone.utc)
        return {"snap": snap, "station_id": ST, "variable": "temperature_2m", "valid_time": snap + timedelta(hours=30),
                "series": series, "value": 1.0, "run": run,
                "avail": None if avail_h is None else datetime(2026, 10, 22, avail_h, tzinfo=timezone.utc)}

    long = pl.DataFrame([
        row("dwd_icon_d2", 6, 8),       # first seen 08:00 <= 12:00: kept (the table would also keep it)
        row("ecmwf_ifs", 6, 9),         # measured 09:00 <= 12:00: kept, although run + 7 h = 13:00 would drop it
        row("ecmwf_ifs", 6, 13),        # measured 13:00 > 12:00: dropped
        row("ecmwf_ifs025", 6, None),   # no measurement: table 6 + 7 = 13:00 > 12:00: dropped
        row("ai", 11, None),            # a product, never delayed
    ], schema=ss.SCHEMA)
    kept, dropped = ss.r1_filter(long)
    assert sorted(kept["series"].to_list()) == ["ai", "dwd_icon_d2", "ecmwf_ifs"]
    assert dropped == {"ecmwf_ifs": 1, "ecmwf_ifs025": 1}

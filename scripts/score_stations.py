#!/usr/bin/env python3
"""Score the MeteoVerdict AI and the public weather models on the 40 main ČHMÚ stations.

Reads only public material: the ledger (``ledger/``), the snapshot assets ``snap-*.tar`` of the
releases (each checked against the sha256 in the ledger), the frozen ČHMÚ observations
(``obs/obs-YYYYMMDD.csv``, ``scripts/fetch_obs.py``) and the station list
(``register/stations40.json``). Writes ``derived/stations.json`` for the verification page.

Dependency: ``polars`` (``pip install polars``); the rest is the standard library (and
``scripts/verify_public.py``, which checks the ledger's hash chain first).

What is computed (pre-registration rules R1-R5, ``preregistration/v1.md``):

* Lead band D+k = ``valid_time - snapshot time`` in [24k, 24k+24) hours, k = 1..5.
* Pairs: (snapshot, station, hour, variable) with an observation. A public model's value counts
  only if its run was available at the snapshot time (R1: run start + the delay below <= snapshot
  time); a value from an unavailable run is dropped, never replaced by an older run (R3).
* Per station, variable and lead: the three public models with the lowest MAE against the AI's
  pairs (at least ``MIN_N`` pairs) are shown next to the AI; MAE, bias and RMSE of the AI and the
  three are taken on the **common pairs** of all four (R3). ``all`` pools the 40 stations and
  lists every model pairwise with the AI, the dropped pairs per reason, and the blend if the
  snapshot has one (the engine has no station-level blend yet: ``blend`` is then ``null``).
* Temperature and wind speed first (with 48-hour chart windows); precipitation (hourly sum) and wind
  gusts as secondary, statistics only. Cloud cover has no ČHMÚ hourly observation: not scored.
* The chart window of lead k uses, for each hour, the snapshot 24k..24k+5 hours before it (one of
  the four daily snapshots); the statistics use all snapshots of the band.

    python scripts/score_stations.py                       # downloads the releases
    python scripts/score_stations.py --snap-dir DIR        # snap-*.tar already downloaded
    python scripts/score_stations.py --ledger test-ledger --test --out /tmp/stations.json
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import tarfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verify_public as vp  # noqa: E402

VERSION = "1"
UTC = timezone.utc
PRIMARY = ("temperature_2m", "wind_speed_10m")
SECONDARY = ("precipitation", "wind_gusts_10m")
VARIABLES = PRIMARY + SECONDARY
UNITS = {"temperature_2m": "°C", "wind_speed_10m": "m/s", "precipitation": "mm", "wind_gusts_10m": "m/s"}
LEADS = (1, 2, 3, 4, 5)
MIN_N = 6  # pairs a model needs (per station) to be ranked
#: R1 (conservative, hours after the run start until the run can be used)
DELAY_H = {"dwd_icon_d2": 2, "ecmwf_ifs": 7, "ecmwf_ifs025": 7, "ecmwf_aifs025_single": 7,
           "chmi_aladin_cz_1km": 4, "chmi_aladin_central_europe_2km": 4}  # fmt: skip
DEFAULT_DELAY_H = 5
NAMES = {
    "ai": "MeteoVerdict AI", "combined": "Naše směs (blend)",
    "dwd_icon_d2": "ICON-D2", "dwd_icon_eu": "ICON-EU", "dwd_icon": "ICON", "ecmwf_ifs": "ECMWF IFS",
    "ecmwf_ifs025": "ECMWF IFS 0,25°", "ecmwf_aifs025_single": "ECMWF AIFS", "chmi_aladin_cz_1km": "ALADIN 1 km",
    "chmi_aladin_central_europe_2km": "ALADIN 2 km", "geosphere_arome_austria": "AROME Austria",
    "knmi_harmonie_arome_europe": "KNMI HARMONIE", "dmi_harmonie_arome_europe": "DMI HARMONIE",
    "meteofrance_arpege_europe": "ARPEGE EU", "meteofrance_arpege_world": "ARPEGE světový", "ncep_gfs_global": "GFS",
    "ncep_aigfs025": "AIGFS", "gem_global": "GEM", "jma_gsm": "JMA GSM", "cma_grapes_global": "CMA GRAPES",
    "ukmo_global_deterministic_10km": "UKMO",
}  # fmt: skip
MODEL_VARS = ["temperature_2m", "precipitation", "wind_speed_10m", "wind_gusts_10m"]
SCHEMA = {"snap": pl.Datetime("us", "UTC"), "station_id": pl.String, "variable": pl.String,
          "valid_time": pl.Datetime("us", "UTC"), "series": pl.String, "value": pl.Float32,
          "run": pl.Datetime("us", "UTC")}  # fmt: skip


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---- snapshots -----------------------------------------------------------------------------


def reduce_snapshot(tar_bytes: bytes, snap: datetime, stations: list[str]) -> pl.DataFrame:
    """One snapshot's values at ``stations`` for the scored variables, leads 24-143 h: columns of
    ``SCHEMA`` (``series`` = ``ai`` or a model id, ``run`` = start of the run behind the value; for
    the AI the issue time)."""
    with tarfile.open(fileobj=io.BytesIO(tar_bytes)) as tar:
        members = {m.name: tar.extractfile(m).read() for m in tar.getmembers() if m.isfile()}
    parts = []
    if "stations.parquet" in members:
        ai = pl.read_parquet(io.BytesIO(members["stations.parquet"]))
        parts.append(ai.filter(pl.col("station_id").is_in(stations), pl.col("variable").is_in(VARIABLES))
                     .select("station_id", "variable", pl.col("time").alias("valid_time"),
                             pl.lit("ai").alias("series"), pl.col("value").cast(pl.Float32),
                             pl.col("issued").alias("run")))  # fmt: skip
    if "stations_models.parquet" in members:
        sm = pl.read_parquet(io.BytesIO(members["stations_models.parquet"]))
        sm = sm.filter(pl.col("station_id").is_in(stations))
        for var in MODEL_VARS:
            if var not in sm.columns:
                continue
            parts.append(sm.filter(pl.col(var).is_not_null())
                         .select("station_id", pl.lit(var).alias("variable"), "valid_time",
                                 pl.col("model").alias("series"), pl.col(var).cast(pl.Float32).alias("value"),
                                 "run"))  # fmt: skip
    if not parts:
        return pl.DataFrame(schema=SCHEMA)
    lead = (pl.col("valid_time") - pl.lit(snap)).dt.total_hours()
    out = pl.concat([p.cast({"run": pl.Datetime("us", "UTC"), "valid_time": pl.Datetime("us", "UTC")}) for p in parts])
    return (out.with_columns(pl.lit(snap).cast(pl.Datetime("us", "UTC")).alias("snap"))
            .filter(lead >= 24, lead < 24 * (LEADS[-1] + 1))
            .select(list(SCHEMA)).cast(SCHEMA))  # fmt: skip


def load_snapshots(rows: list[dict], stations: list[str], snap_dir: Path | None, cache: Path | None, repo: str):
    """Reduced frames of the ok snapshots; the tars are checked against the ledger's sha256."""
    frames = []
    for row in rows:
        asset, snap = row["asset"], vp.parse_time(row["snapshot_time"])
        cached = cache / f"{asset['sha256']}.parquet" if cache else None
        if cached and cached.exists():
            frames.append(pl.read_parquet(cached))
            continue
        if snap_dir and (snap_dir / asset["name"]).exists():
            data = (snap_dir / asset["name"]).read_bytes()
        else:
            data = vp.fetch(f"https://github.com/{repo}/releases/download/{asset['tag']}/{asset['name']}")
        if vp.sha256(data) != asset["sha256"]:
            raise SystemExit(f"FAIL {row['snapshot_time']}: {asset['name']} does not match the ledger's sha256")
        frame = reduce_snapshot(data, snap, stations)
        if cached:
            cached.parent.mkdir(parents=True, exist_ok=True)
            frame.write_parquet(cached, compression="zstd")
        frames.append(frame)
    return frames


def load_obs(folder: Path) -> tuple[pl.DataFrame, list[str]]:
    files = sorted(folder.glob("obs-*.csv"))
    frames = [pl.read_csv(f, schema_overrides={v: pl.Float64 for v in VARIABLES}) for f in files]
    if not frames:
        return pl.DataFrame(schema={"station_id": pl.String, "valid_time": pl.Datetime("us", "UTC"),
                                    "variable": pl.String, "obs": pl.Float64}), []  # fmt: skip
    obs = (pl.concat(frames).unpivot(index=["station_id", "time"], variable_name="variable", value_name="obs")
           .drop_nulls("obs")
           .select("station_id", pl.col("time").str.to_datetime("%Y-%m-%dT%H:%M:%SZ", time_zone="UTC")
                   .dt.cast_time_unit("us").alias("valid_time"), "variable", "obs"))  # fmt: skip
    return obs, [f"{sha256_file(f)}  {f.name}" for f in files]


# ---- statistics ----------------------------------------------------------------------------


def err_stats(wide: pl.DataFrame, series: str) -> dict:
    e = wide[series] - wide["obs"]
    return {"k": series, "n": NAMES.get(series, series), "mae": round(float(e.abs().mean()), 2),
            "bias": round(float(e.mean()), 2), "rmse": round(float((e**2).mean() ** 0.5), 2)}  # fmt: skip


def rank_models(wide: pl.DataFrame, models: list[str], min_n: int) -> list[tuple[float, str, int]]:
    ranked = []
    for m in models:
        p = wide.drop_nulls(["ai", m])
        if p.height >= min_n:
            ranked.append((float((p[m] - p["obs"]).abs().mean()), m, p.height))
    return sorted(ranked)


def group_stats(wide: pl.DataFrame, models: list[str], min_n: int) -> dict | None:
    """AI and its three best public models on their common pairs (``wide``: one row per pair, a
    column per series); None without AI pairs."""
    ai = wide.drop_nulls("ai")
    if ai.is_empty():
        return None
    top = [m for _, m, _ in rank_models(ai, models, min_n)[:3]]
    common = ai.drop_nulls(top) if top else ai
    out = {"ai_pairs": ai.height, "n": common.height, "top": top, "stats": []}
    if common.height:
        out["stats"] = [err_stats(common, s) for s in ["ai", *top]]
    return out


def wide_pairs(long: pl.DataFrame) -> pl.DataFrame:
    """One row per (snapshot, station, hour) with a column per series, plus ``obs``."""
    return long.pivot(on="series", index=["snap", "station_id", "valid_time", "obs", "lead_h"], values="value",
                      aggregate_function="first")  # fmt: skip


def r1_filter(long: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, int]]:
    """Drop public-model values whose run was not yet available at the snapshot (R1); count them."""
    delay = pl.col("series").replace_strict(DELAY_H, default=DEFAULT_DELAY_H, return_dtype=pl.Int64)
    late = (pl.col("series") != "ai") & (pl.col("run") + pl.duration(hours=delay) > pl.col("snap"))
    dropped = long.filter(late).group_by("series").len()
    return long.filter(~late), {r["series"]: r["len"] for r in dropped.iter_rows(named=True)}


def window(hours: list[datetime], obs_wide: pl.DataFrame, cols: list[str]) -> dict[str, list]:
    by_time = {r["valid_time"]: r for r in obs_wide.iter_rows(named=True)}
    return {c: [None if (r := by_time.get(t)) is None or r.get(c) is None else round(float(r[c]), 1) for t in hours]
            for c in cols}  # fmt: skip


def score(frames: list[pl.DataFrame], obs: pl.DataFrame, stations: list[dict], asof: datetime | None) -> dict:
    ids = [s["id"] for s in stations]
    long = pl.concat(frames) if frames else pl.DataFrame(schema=SCHEMA)
    long = long.join(obs, on=["station_id", "valid_time", "variable"], how="inner")
    long = long.with_columns(((pl.col("valid_time") - pl.col("snap")).dt.total_hours()).alias("lead_h"))
    asof = asof or (obs["valid_time"].max() if obs.height else None)
    hours = [asof - timedelta(hours=47 - i) for i in range(48)] if asof else []
    data, overall = {}, {}
    for var in VARIABLES:
        lv_all = long.filter(pl.col("variable") == var)
        for k in LEADS:
            lv = lv_all.filter((pl.col("lead_h") >= 24 * k) & (pl.col("lead_h") < 24 * k + 24))
            lv, dropped = r1_filter(lv)
            if lv.is_empty() or lv.filter(pl.col("series") == "ai").is_empty():
                continue
            wide = wide_pairs(lv)
            models = [c for c in wide.columns if c not in ("snap", "station_id", "valid_time", "obs", "lead_h", "ai")]
            # the whole network: every model pairwise with the AI, and the common set of the top three
            pw = []
            ai_rows = wide.drop_nulls("ai")
            for m in models:
                p = ai_rows.drop_nulls(m)
                if p.height:
                    pw.append({"k": m, "n": NAMES.get(m, m), "pairs": p.height,
                               "mae": round(float((p[m] - p["obs"]).abs().mean()), 2),
                               "bias": round(float((p[m] - p["obs"]).mean()), 2),
                               "ai_mae": round(float((p["ai"] - p["obs"]).abs().mean()), 2)})  # fmt: skip
            pw.sort(key=lambda r: r["mae"])
            g = group_stats(wide, models, 30)
            overall[f"{var}|{k}"] = {
                **(g or {}), "models": pw,
                "blend": err_stats(ai_rows.drop_nulls("combined"), "combined") if "combined" in models and
                ai_rows.drop_nulls("combined").height else None,
                "dropped_not_common": (g["ai_pairs"] - g["n"]) if g else 0,
                "dropped_run_unavailable": dropped,
            }  # fmt: skip
            for s in ids:
                ws = wide.filter(pl.col("station_id") == s)
                gs = group_stats(ws, models, MIN_N)
                if gs is None:
                    continue
                entry = {"n": gs["n"], "stats": gs["stats"]}
                if var in PRIMARY and hours:
                    chart = ws.filter((pl.col("lead_h") < 24 * k + 6)).drop("snap", "lead_h")
                    series = ["ai", *gs["top"]]
                    cw = window(hours, chart, series)
                    entry |= {"t0": f"{hours[0]:%Y-%m-%dT%H:%MZ}", "ai": cw["ai"],
                              "top": [{"n": NAMES.get(m, m), "v": cw[m]} for m in gs["top"]]}  # fmt: skip
                data[f"{s}|{var}|{k}"] = entry
    if hours:  # the observed values of the chart window, once per station and variable
        for var in PRIMARY:
            for s in ids:
                o = obs.filter((pl.col("station_id") == s) & (pl.col("variable") == var))
                w = window(hours, o.rename({"obs": "o"}), ["o"])["o"]
                for k in LEADS:
                    if (e := data.get(f"{s}|{var}|{k}")) is not None and "t0" in e:
                        e["obs"] = w
    return {"data": data, "overall": overall, "asof": f"{asof:%Y-%m-%dT%H:%MZ}" if asof else None}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ledger", type=Path, default=Path("ledger"))
    ap.add_argument("--obs", type=Path, default=Path("obs"))
    ap.add_argument("--stations", type=Path, default=Path("register/stations40.json"))
    ap.add_argument("--out", type=Path, default=Path("derived/stations.json"))
    ap.add_argument("--snap-dir", type=Path, help="folder with the downloaded snap-*.tar (else: from the releases)")
    ap.add_argument("--cache", type=Path, help="folder for reduced snapshots (keyed by the tar's sha256)")
    ap.add_argument("--days", type=int, default=60, help="only snapshots of the last N days (0: all)")
    ap.add_argument("--repo", default=vp.REPO)
    ap.add_argument("--test", action="store_true", help="mark the output as test data (dry-run ledger)")
    ap.add_argument("--asof", help="end of the chart window (default: the last observation hour)")
    args = ap.parse_args()

    ledger = vp.read_ledger(args.ledger)
    problems = vp.check_chain(ledger)
    if problems:
        print("\n".join(f"FAIL {p}" for p in problems))
        return 1
    ok = [r for r in ledger if r["status"] == "ok"]
    if args.days and ledger:
        since = vp.parse_time(ledger[-1]["snapshot_time"]) - timedelta(days=args.days)
        ok = [r for r in ok if vp.parse_time(r["snapshot_time"]) > since]
    reg = json.loads(args.stations.read_text())
    stations = reg["stations"]
    frames = load_snapshots(ok, [s["id"] for s in stations], args.snap_dir, args.cache, args.repo)
    obs, sums = load_obs(args.obs)
    result = score(frames, obs, stations, vp.parse_time(args.asof) if args.asof else None)
    me = Path(__file__)
    doc = {
        "version": 1, "kind": "test" if args.test else "live",
        "generated": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "stations": stations, "names": NAMES, "units": UNITS, "primary": list(PRIMARY), "leads": list(LEADS),
        "asof": result["asof"], "overall": result["overall"], "data": result["data"],
        "provenance": {
            "snapshots": {"ok_used": len(ok), "ledger_rows": len(ledger),
                          "missing": sum(r["status"] != "ok" for r in ledger),
                          "first": ok[0]["snapshot_time"] if ok else None, "last": ok[-1]["snapshot_time"] if ok else None,
                          "assets_sha256": vp.sha256("\n".join(f"{r['asset']['name']} {r['asset']['sha256']}" for r in ok).encode()),
                          "last_assets": [{"name": r["asset"]["name"], "sha256": r["asset"]["sha256"]} for r in ok[-4:]]},
            "chain_head": ledger[-1]["chain_hash"] if ledger else None,
            "obs": {"files": len(sums), "first": sums[0].split("obs-")[1][:8] if sums else None,
                    "last": sums[-1].split("obs-")[1][:8] if sums else None,
                    "sums_sha256": vp.sha256("\n".join(sums).encode()), "quality": "provisional (ČHMÚ now/, unchecked)",
                    "source": "https://opendata.chmi.cz/meteorology/climate/"},
            "stations": {"file": "register/stations40.json", "sha256": sha256_file(args.stations), "n": len(stations)},
            "script": {"version": VERSION, "sha256": sha256_file(me), "polars": pl.__version__},
            "rules": "preregistration/v1.md R1-R5; leads D+k = [24k, 24k+24) h; common pairs; no filling",
            "command": "python3 scripts/score_stations.py" + (f" --ledger {args.ledger}" if str(args.ledger) != "ledger" else ""),
        },
    }  # fmt: skip
    args.out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(doc, ensure_ascii=False, separators=(",", ":"))
    args.out.write_text(text)
    print(f"{args.out}: {len(text) / 1e3:.0f} kB, {len(ok)} snapshots, {len(result['data'])} station series, asof {result['asof']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

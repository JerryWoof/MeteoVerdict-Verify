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
  only if its run was available at the snapshot time (R1: the measured time the run was first seen
  complete, ``release_inputs.parquet`` of the snapshot; without it run start + the delay below <= snapshot
  time); a value from an unavailable run is dropped, never replaced by an older run (R3).
* Per station, variable and lead: the three public models with the lowest MAE against the AI's
  pairs (at least ``MIN_N`` pairs) are shown next to the AI; MAE, bias and RMSE of the AI and the
  three are taken on the **common pairs** of all of them (R3), the blend (the engine's combination
  of the models at the station, ``series`` ``blend``) among them when the snapshot has one. ``all``
  pools the 40 stations and lists every model pairwise with the AI, the dropped pairs per reason,
  and the blend against the AI on their common pairs (``blend`` is ``null`` for older snapshots).
* Temperature and wind speed first (with 48-hour chart windows); precipitation (hourly sum) and wind
  gusts as secondary, statistics only. Gusts are scored per part of the day (Prague night, morning,
  afternoon, evening; the largest gust of the part, as the AI is made): the models' value is the
  largest of their hourly gusts in the part (at least ``MIN_PART_MODEL`` hours with a value), the
  measured one the largest of the part's hourly gusts (at least ``MIN_PART_OBS``). Cloud cover (%) and
  sunshine (s of the hour) are scored hourly like precipitation, where the frozen day has them
  (``scripts/fetch_obs.py``, hourly elements N and SSV1H; days frozen earlier do not).
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
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verify_public as vp  # noqa: E402

VERSION = "3"
UTC = timezone.utc
PRIMARY = ("temperature_2m", "wind_speed_10m")
#: gusts are scored per part of the day, with the value at the part's start as ``valid_time``
PART_GUST = "wind_gusts_part"
SECONDARY = ("precipitation", "cloud_cover", "sunshine_duration", PART_GUST)
VARIABLES = PRIMARY + SECONDARY
#: the columns of the frozen observation files
OBS_COLUMNS = ("temperature_2m", "wind_speed_10m", "precipitation", "wind_gusts_10m", "cloud_cover", "sunshine_duration")
UNITS = {"temperature_2m": "°C", "wind_speed_10m": "m/s", "precipitation": "mm", "cloud_cover": "%",
         "sunshine_duration": "s", PART_GUST: "m/s"}  # fmt: skip
PARTS = {"night": (0, 6), "morning": (6, 12), "afternoon": (12, 18), "evening": (18, 24)}
PRAGUE = "Europe/Prague"
MIN_PART_MODEL = 2
MIN_PART_OBS = 4
#: series of the engine that are products issued at the snapshot (no model run delay, R1)
PRODUCTS = ("ai", "blend")
LEADS = (1, 2, 3, 4, 5)
MIN_N = 6  # pairs a model needs (per station) to be ranked
#: R1 (conservative, hours after the run start until the run can be used)
DELAY_H = {"dwd_icon_d2": 2, "ecmwf_ifs": 7, "ecmwf_ifs025": 7, "ecmwf_aifs025_single": 7,
           "chmi_aladin_cz_1km": 4, "chmi_aladin_central_europe_2km": 4}  # fmt: skip
DEFAULT_DELAY_H = 5
NAMES = {
    "ai": "MeteoVerdict AI", "blend": "Naše směs (blend)", "combined": "Naše směs (blend)",
    "dwd_icon_d2": "ICON-D2", "dwd_icon_eu": "ICON-EU", "dwd_icon": "ICON", "ecmwf_ifs": "ECMWF IFS",
    "ecmwf_ifs025": "ECMWF IFS 0,25°", "ecmwf_aifs025_single": "ECMWF AIFS", "chmi_aladin_cz_1km": "ALADIN 1 km",
    "chmi_aladin_central_europe_2km": "ALADIN 2 km", "geosphere_arome_austria": "AROME Austria",
    "knmi_harmonie_arome_europe": "KNMI HARMONIE", "dmi_harmonie_arome_europe": "DMI HARMONIE",
    "meteofrance_arpege_europe": "ARPEGE EU", "meteofrance_arpege_world": "ARPEGE světový", "ncep_gfs_global": "GFS",
    "ncep_aigfs025": "AIGFS", "gem_global": "GEM", "jma_gsm": "JMA GSM", "cma_grapes_global": "CMA GRAPES",
    "ukmo_global_deterministic_10km": "UKMO",
}  # fmt: skip
MODEL_VARS = ["temperature_2m", "precipitation", "wind_speed_10m", "cloud_cover", "sunshine_duration"]
SCHEMA = {"snap": pl.Datetime("us", "UTC"), "station_id": pl.String, "variable": pl.String,
          "valid_time": pl.Datetime("us", "UTC"), "series": pl.String, "value": pl.Float32,
          "run": pl.Datetime("us", "UTC"), "avail": pl.Datetime("us", "UTC")}  # fmt: skip
#: ``avail`` = when the run behind the value was first seen complete (the snapshot's
#: ``release_inputs.parquet``); null where the log has no such time (R1 then uses ``DELAY_H``)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def part_start(day: date, part: str) -> datetime:
    """Start (UTC) of the Prague ``part`` of ``day``."""
    return datetime.combine(day, time(PARTS[part][0]), ZoneInfo(PRAGUE)).astimezone(UTC)


def local_part(t: pl.Expr) -> tuple[pl.Expr, pl.Expr]:
    """The Prague ``day`` and ``part`` of a UTC time (the caller shifts hour-ending values)."""
    local = t.dt.convert_time_zone(PRAGUE)
    part = pl.lit(None, pl.String)
    for name, (first, end) in PARTS.items():
        part = pl.when(local.dt.hour().is_between(first, end - 1)).then(pl.lit(name)).otherwise(part)
    return local.dt.date().alias("day"), part.alias("part")


def with_part_start(frame: pl.DataFrame) -> pl.DataFrame:
    """``valid_time`` = the start of the row's ``day`` and ``part`` (UTC), those columns dropped."""
    starts = {(d, p): part_start(d, p) for d, p in frame.select("day", "part").unique().iter_rows()}
    return frame.with_columns(
        pl.struct("day", "part").map_elements(lambda r: starts[(r["day"], r["part"])], return_dtype=pl.Datetime("us", "UTC"))
        .alias("valid_time")).drop("day", "part")  # fmt: skip


def hourly_gust_parts(frame: pl.DataFrame, keys: list[str], min_hours: int) -> pl.DataFrame:
    """``frame``: hourly gusts (``valid_time`` = the hour's end, ``value``, the ``keys`` columns and
    optionally ``run``). Per ``keys`` group and Prague part of day: the largest value (and the
    latest ``run``), ``valid_time`` = the part's start; parts with fewer than ``min_hours`` values
    are left out."""
    day, part = local_part(pl.col("valid_time") - pl.duration(hours=1))
    agg = [pl.col("value").max(), pl.len().alias("_n")] + ([pl.col("run").max()] if "run" in frame.columns else [])
    out = (frame.drop_nulls("value").with_columns(day, part).group_by([*keys, "day", "part"]).agg(agg)
           .filter(pl.col("_n") >= min_hours).drop("_n"))  # fmt: skip
    if out.is_empty():
        return out.with_columns(pl.lit(None, pl.Datetime("us", "UTC")).alias("valid_time")).drop("day", "part")
    return with_part_start(out)


# ---- snapshots -----------------------------------------------------------------------------


def reduce_snapshot(tar_bytes: bytes, snap: datetime, stations: list[str]) -> pl.DataFrame:
    """One snapshot's values at ``stations`` for the scored variables, leads 24-143 h: columns of
    ``SCHEMA`` (``series`` = ``ai`` or a model id, ``run`` = start of the run behind the value; for
    the AI the issue time)."""
    with tarfile.open(fileobj=io.BytesIO(tar_bytes)) as tar:
        members = {m.name: tar.extractfile(m).read() for m in tar.getmembers() if m.isfile()}
    for base in ("stations", "stations_models", "stations_parts"):  # policy public-v1: the files cut to the 40 stations
        if f"{base}-public.parquet" in members:
            members[f"{base}.parquet"] = members.pop(f"{base}-public.parquet")
    parts = []
    sm = None
    if "stations_models.parquet" in members:
        sm = pl.read_parquet(io.BytesIO(members["stations_models.parquet"])).filter(pl.col("station_id").is_in(stations))
    if "stations.parquet" in members and (sm is None or "ai" not in set(sm["model"])):  # snapshots before the station table had the AI
        ai = pl.read_parquet(io.BytesIO(members["stations.parquet"]))
        parts.append(ai.filter(pl.col("station_id").is_in(stations), pl.col("variable").is_in(VARIABLES))
                     .select("station_id", "variable", pl.col("time").alias("valid_time"),
                             pl.lit("ai").alias("series"), pl.col("value").cast(pl.Float32),
                             pl.col("issued").alias("run")))  # fmt: skip
    if sm is not None:
        for var in MODEL_VARS:
            if var not in sm.columns:
                continue
            parts.append(sm.filter(pl.col(var).is_not_null())
                         .select("station_id", pl.lit(var).alias("variable"), "valid_time",
                                 pl.col("model").alias("series"), pl.col(var).cast(pl.Float32).alias("value"),
                                 "run"))  # fmt: skip
        if "wind_gusts_10m" in sm.columns:  # the models' gusts per part of the day
            g = sm.filter(pl.col("wind_gusts_10m").is_not_null(), ~pl.col("model").is_in(list(PRODUCTS)))
            g = hourly_gust_parts(g.select("station_id", pl.col("model").alias("series"), "valid_time",
                                           pl.col("wind_gusts_10m").cast(pl.Float32).alias("value"), "run"),
                                  ["station_id", "series"], MIN_PART_MODEL)
            if g.height:
                parts.append(g.select("station_id", pl.lit(PART_GUST).alias("variable"), "valid_time", "series", "value", "run"))
    if "stations_parts.parquet" in members:  # the AI's parts of the day at the stations
        sp = pl.read_parquet(io.BytesIO(members["stations_parts.parquet"])).filter(pl.col("station_id").is_in(stations))
        if "gust" in sp.columns:
            sp = sp.filter(pl.col("gust").is_not_null())
            if sp.height:
                parts.append(with_part_start(sp).select("station_id", pl.lit(PART_GUST).alias("variable"), "valid_time",
                                                         pl.lit("ai").alias("series"), pl.col("gust").cast(pl.Float32).alias("value"),
                                                         pl.lit(snap).cast(pl.Datetime("us", "UTC")).alias("run")))  # fmt: skip
    if not parts:
        return pl.DataFrame(schema=SCHEMA)
    lead = (pl.col("valid_time") - pl.lit(snap)).dt.total_hours()
    out = pl.concat([p.cast({"run": pl.Datetime("us", "UTC"), "valid_time": pl.Datetime("us", "UTC")}) for p in parts])
    if "release_inputs.parquet" in members:  # measured availability of the public models' runs (R1)
        ri = (pl.read_parquet(io.BytesIO(members["release_inputs.parquet"]))
              .filter(pl.col("available_at").is_not_null())
              .select(pl.col("source").alias("series"), pl.col("run_time").alias("run"), pl.col("available_at").alias("avail"))
              .cast({"run": pl.Datetime("us", "UTC"), "avail": pl.Datetime("us", "UTC")}).unique(["series", "run"]))  # fmt: skip
        out = out.join(ri, on=["series", "run"], how="left")
    else:
        out = out.with_columns(pl.lit(None, pl.Datetime("us", "UTC")).alias("avail"))
    return (out.with_columns(pl.lit(snap).cast(pl.Datetime("us", "UTC")).alias("snap"))
            .filter(lead >= 24, lead < 24 * (LEADS[-1] + 1))
            .select(list(SCHEMA)).cast(SCHEMA))  # fmt: skip


def load_snapshots(rows: list[dict], stations: list[str], snap_dir: Path | None, cache: Path | None, repo: str):
    """Reduced frames of the ok snapshots; the tars are checked against the ledger's sha256."""
    frames = []
    for row in rows:
        asset, snap = row["asset"], vp.parse_time(row["snapshot_time"])
        cached = cache / f"v{VERSION}-{asset['sha256']}.parquet" if cache else None
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


def obs_files(folder: Path) -> tuple[list[Path], dict[str, str]]:
    """One file per day: the newest checked one (``obs-YYYYMMDD.final*.csv``) if there is one, else the
    provisional file; and the quality of each day (rule R5)."""
    by_day: dict[str, list[Path]] = {}
    for f in folder.glob("obs-*.csv"):
        by_day.setdefault(f.name[4:12], []).append(f)
    files, quality = [], {}
    for day in sorted(by_day):
        fin = sorted((p for p in by_day[day] if ".final" in p.name), key=lambda p: (len(p.name), p.name))
        pick = fin[-1] if fin else next(p for p in by_day[day] if ".final" not in p.name)
        files.append(pick)
        quality[day] = "final" if fin else "provisional"
    return files, quality


def load_obs(folder: Path) -> tuple[pl.DataFrame, list[str]]:
    files, _ = obs_files(folder)
    # days frozen before the cloud and sunshine columns existed simply lack them
    frames = [pl.read_csv(f, schema_overrides={v: pl.Float64 for v in OBS_COLUMNS if v in f.read_text().split("\n", 1)[0]})
              for f in files]  # fmt: skip
    if not frames:
        return pl.DataFrame(schema={"station_id": pl.String, "valid_time": pl.Datetime("us", "UTC"),
                                    "variable": pl.String, "obs": pl.Float64}), []  # fmt: skip
    obs = (pl.concat(frames, how="diagonal_relaxed").unpivot(index=["station_id", "time"], variable_name="variable", value_name="obs")
           .drop_nulls("obs")
           .select("station_id", pl.col("time").str.to_datetime("%Y-%m-%dT%H:%M:%SZ", time_zone="UTC")
                   .dt.cast_time_unit("us").alias("valid_time"), "variable", "obs"))  # fmt: skip
    gusts = hourly_gust_parts(obs.filter(pl.col("variable") == "wind_gusts_10m")
                              .select("station_id", "valid_time", pl.col("obs").alias("value")), ["station_id"], MIN_PART_OBS)
    obs = pl.concat([obs.filter(pl.col("variable").is_in(PRIMARY + ("precipitation", "cloud_cover", "sunshine_duration"))),
                     gusts.select("station_id", "valid_time", pl.lit(PART_GUST).alias("variable"), pl.col("value").alias("obs"))])
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
    """AI, the blend (when ``wide`` has one with at least ``min_n`` AI pairs) and the AI's three best
    public models on their common pairs (``wide``: one row per pair, a column per series); None
    without AI pairs."""
    ai = wide.drop_nulls("ai")
    if ai.is_empty():
        return None
    top = [m for _, m, _ in rank_models(ai, models, min_n)[:3]]
    blend = ["blend"] if "blend" in ai.columns and ai.drop_nulls("blend").height >= min_n else []
    common = ai.drop_nulls([*blend, *top]) if (top or blend) else ai
    out = {"ai_pairs": ai.height, "n": common.height, "top": top, "with_blend": bool(blend), "stats": []}
    if common.height:
        out["stats"] = [err_stats(common, s) for s in ["ai", *blend, *top]]
    return out


def wide_pairs(long: pl.DataFrame) -> pl.DataFrame:
    """One row per (snapshot, station, hour) with a column per series, plus ``obs``."""
    return long.pivot(on="series", index=["snap", "station_id", "valid_time", "obs", "lead_h"], values="value",
                      aggregate_function="first")  # fmt: skip


def r1_filter(long: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, int]]:
    """Drop public-model values whose run was not yet available at the snapshot (R1); count them.
    The AI and the blend are issued by the snapshot (the log only keeps issues up to it): no delay.
    Availability is the measured time the run was first seen complete (``avail``, from the
    snapshot's ``release_inputs``); where the log has none, the conservative ``DELAY_H`` table."""
    delay = pl.col("series").replace_strict(DELAY_H, default=DEFAULT_DELAY_H, return_dtype=pl.Int64)
    available = pl.coalesce(pl.col("avail"), pl.col("run") + pl.duration(hours=delay))
    late = ~pl.col("series").is_in(list(PRODUCTS)) & (available > pl.col("snap"))
    dropped = long.filter(late).group_by("series").len()
    return long.filter(~late), {r["series"]: r["len"] for r in dropped.iter_rows(named=True)}


def blend_pairwise(ai_rows: pl.DataFrame) -> dict | None:
    """The blend on the pairs where the AI also has a value: its errors and the AI's."""
    p = ai_rows.drop_nulls("blend") if "blend" in ai_rows.columns else ai_rows.clear()
    if p.is_empty():
        return None
    return {**err_stats(p, "blend"), "pairs": p.height, "ai_mae": round(float((p["ai"] - p["obs"]).abs().mean()), 2)}


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
            models = [c for c in wide.columns if c not in ("snap", "station_id", "valid_time", "obs", "lead_h", *PRODUCTS)]
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
                "blend": blend_pairwise(ai_rows),
                "dropped_not_common": (g["ai_pairs"] - g["n"]) if g else 0,
                "dropped_run_unavailable": dropped,
            }  # fmt: skip
            for s in ids:
                ws = wide.filter(pl.col("station_id") == s)
                gs = group_stats(ws, models, MIN_N)
                if gs is None:
                    continue
                entry = {"n": gs["n"], "stats": gs["stats"], "with_blend": gs["with_blend"]}
                if var in PRIMARY and hours:
                    chart = ws.filter((pl.col("lead_h") < 24 * k + 6)).drop("snap", "lead_h")
                    series = ["ai", *(["blend"] if gs["with_blend"] else []), *gs["top"]]
                    cw = window(hours, chart, series)
                    entry |= {"t0": f"{hours[0]:%Y-%m-%dT%H:%MZ}", "ai": cw["ai"],
                              "top": [{"n": NAMES.get(m, m), "v": cw[m]} for m in gs["top"]]}  # fmt: skip
                    if gs["with_blend"]:
                        entry["blend"] = cw["blend"]
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
    withdrawn = vp.read_withdrawn(args.ledger)
    ok = [r for r in ok if r["asset"]["tag"] not in withdrawn]  # taken down on purpose, see withdrawn.json
    if not ok:
        print("no snapshot with a published asset yet; nothing to score")
        return 0
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
                    "sums_sha256": vp.sha256("\n".join(sums).encode()),
                    "days": obs_files(args.obs)[1],
                    "quality": "per day: provisional (ČHMÚ now/, unchecked) or final (ČHMÚ recent/, checked); see days",
                    "source": "https://opendata.chmi.cz/meteorology/climate/"},
            "stations": {"file": "register/stations40.json", "sha256": sha256_file(args.stations), "n": len(stations)},
            "script": {"version": VERSION, "sha256": sha256_file(me), "polars": pl.__version__},
            "rules": "preregistration/v1.md R1-R5; leads D+k = [24k, 24k+24) h; common pairs; no filling",
            "command": "python3 scripts/score_stations.py" + (f" --ledger {args.ledger.name}" if args.ledger.name != "ledger" else ""),
        },
    }  # fmt: skip
    args.out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(doc, ensure_ascii=False, separators=(",", ":"))
    args.out.write_text(text)
    print(f"{args.out}: {len(text) / 1e3:.0f} kB, {len(ok)} snapshots, {len(result['data'])} station series, asof {result['asof']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

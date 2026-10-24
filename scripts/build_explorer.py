#!/usr/bin/env python3
"""The data of the verification page's snapshot explorer, from the public material only.

Takes the newest ``ok`` snapshot of the ledger whose release asset still exists (not listed in
``ledger/withdrawn.json``), checks the tar against the ledger's sha256 (``verify_public``) and
writes one small JSON file per district and per station into ``--out`` (default
``derived/explorer``), so the page loads only what is clicked:

* ``index.json``: the snapshot (times, hashes), the districts, the 40 stations, the series, the
  variables and the observation elements.
* ``okresy/<code>.json``: every variable of the district for every series (``ai``, ``combined``,
  ``reference`` and every public model) on one hourly time axis, with the rain and sky parts of the day.
* ``stations/<id>.json``: every variable of every series at the station (every model, ``ai``,
  ``blend``), the parts of the day, and what ČHMÚ measured (``obs-public``): the hourly values of the
  variables that have a model counterpart, and the last value of every element.

Needs ``polars``. Anyone can run it: ``python3 scripts/build_explorer.py``.

    python3 scripts/build_explorer.py --out derived/explorer [--snap-dir folder-with-tars]
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
import tarfile
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verify_public as vp  # noqa: E402

REPO = vp.REPO
#: variable -> unit shown (only where the unit is sure)
UNITS = {
    "temperature_2m": "°C", "temperature_2m_lo": "°C", "temperature_2m_hi": "°C", "temperature_5cm": "°C",
    "dew_point_2m": "°C", "surface_temperature": "°C", "wind_speed_10m": "m/s", "wind_gusts_10m": "m/s",
    "wind_direction_10m": "°", "precipitation": "mm/h", "snowfall": "cm/h", "snow_depth": "m",
    "cloud_cover": "%", "cloud_cover_low": "%", "relative_humidity_2m": "%", "pressure_msl": "hPa",
    "sunshine_duration": "s v hodině", "shortwave_radiation": "W/m²", "visibility": "m", "cape": "J/kg",
    "snow_depth_water_equivalent": "mm", "fog_probability": "%",
}  # fmt: skip


def read_tar(data: bytes) -> dict[str, bytes]:
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        return {m.name: tar.extractfile(m).read() for m in tar.getmembers() if m.isfile()}


def table(members: dict[str, bytes], name: str) -> pl.DataFrame | None:
    return pl.read_parquet(io.BytesIO(members[name])) if name in members else None


def num(v):
    """JSON number: one decimal for large values, three significant digits for small ones; None for NaN."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    v = float(v)
    return round(v, 1) if abs(v) >= 10 else round(v, 3 if abs(v) < 1 else 2)


def iso(t: datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%MZ")


def axis_series(frame: pl.DataFrame, time_col: str, series_col: str, value_col: str, times: list) -> dict:
    """``{series: [value or None per time of the axis]}`` from long rows."""
    index = {t: i for i, t in enumerate(times)}
    out: dict[str, list] = {}
    for s, t, v in frame.select(series_col, time_col, value_col).iter_rows():
        out.setdefault(s, [None] * len(times))[index[t]] = num(v)
    return out


def okres_file(code: str, allv: pl.DataFrame, hourly: pl.DataFrame | None, rain, sky) -> dict:
    rows = allv.filter(pl.col("okres") == code)
    times = sorted(rows["time"].unique().to_list())
    variables: dict[str, dict] = {}
    for variable in sorted(rows["variable"].unique().to_list()):
        sub = rows.filter(pl.col("variable") == variable)
        variables[variable] = axis_series(sub, "time", "series", "value", times)
    if hourly is not None:  # the AI's hourly precipitation, cloud and sunshine where the main table has none
        h = hourly.filter(pl.col("okres") == code).with_columns(pl.lit("ai").alias("series"))
        for variable in h["variable"].unique().to_list():
            if "ai" in variables.get(variable, {}):
                continue
            hv = h.filter(pl.col("variable") == variable)
            extra_times = sorted(set(hv["time"].to_list()) - set(times))
            if extra_times:  # keep one axis: extend it and pad the rest
                new = sorted(set(times) | set(extra_times))
                pad = {t: i for i, t in enumerate(new)}
                for v in variables.values():
                    for s, arr in v.items():
                        full = [None] * len(new)
                        for t, x in zip(times, arr, strict=True):
                            full[pad[t]] = x
                        v[s] = full
                times = new
            variables.setdefault(variable, {})["ai"] = axis_series(hv, "time", "series", "value", times)["ai"]
    doc = {"code": code, "times": [iso(t) for t in times], "variables": variables}
    parts = {}
    if rain is not None:
        r = rain.filter(pl.col("okres") == code).drop("okres", "issued").sort("day", "part")
        parts["rain"] = [{**row, "day": str(row["day"])} for row in r.to_dicts()]
    if sky is not None:
        s = sky.filter(pl.col("okres") == code).drop("okres", "issued").sort("day", "part")
        parts["sky"] = [{**row, "day": str(row["day"])} for row in s.to_dicts()]
    for key, rows_ in parts.items():
        parts[key] = [{k: (num(v) if isinstance(v, float) else v) for k, v in row.items()} for row in rows_]
    return doc | {"parts": parts}


OBS_MAP = {  # (step, element) -> (variable, how, scale)
    ("10min", "T"): ("temperature_2m", "at", 1), ("10min", "F"): ("wind_speed_10m", "at", 1),
    ("10min", "H"): ("relative_humidity_2m", "at", 1), ("10min", "D"): ("wind_direction_10m", "at", 1),
    ("10min", "TPM"): ("temperature_5cm", "at", 1), ("10min", "RGLB10"): ("shortwave_radiation", "at", 1),
    ("10min", "SRA10M"): ("precipitation", "sum6", 1), ("10min", "Fmax"): ("wind_gusts_10m", "max6", 1),
    ("1h", "Td"): ("dew_point_2m", "at", 1), ("1h", "P_hm"): ("pressure_msl", "at", 1),
    ("1h", "N"): ("cloud_cover", "okta", 1), ("1h", "SSV1H"): ("sunshine_duration", "at", 360),
    ("1h", "VV"): ("visibility", "at", 1000), ("1h", "SCEa"): ("snow_depth", "at", 0.01),
}  # fmt: skip


def observations(obs: pl.DataFrame, station: str) -> dict:
    """Hourly values of the measured variables that have a model counterpart (ČHMÚ quality codes 0, 3 and
    5 only, as the scores) on an hourly axis, and the last value of every element with its time and quality."""
    rows = obs.filter(pl.col("station_id") == station)
    if rows.is_empty():
        return {}
    good = rows.filter(pl.col("quality").is_in([0, 3, 5]), pl.col("value").is_not_null())
    series: dict[str, dict] = {}
    for (step, element), (variable, how, scale) in OBS_MAP.items():
        sub = good.filter(pl.col("step") == step, pl.col("element") == element).sort("time")
        if sub.is_empty():
            continue
        vals = dict(zip(sub["time"].to_list(), sub["value"].to_list(), strict=True))
        out: dict = {}
        if how in ("at", "okta"):
            for t, v in vals.items():
                if t.minute == 0:
                    out[t] = (100.0 if v >= 8 else v * 12.5) if how == "okta" else v * scale
        else:
            for t in vals:
                if t.minute != 0:
                    continue
                group = [vals.get(t.replace(minute=0) - timedelta(minutes=10 * k)) for k in range(6)]
                if how == "sum6" and all(g is not None for g in group):
                    out[t] = sum(group)
                if how == "max6" and any(g is not None for g in group):
                    out[t] = max(g for g in group if g is not None)
        if out:
            series[variable] = out
    times = sorted({t for v in series.values() for t in v})
    doc = {"times": [iso(t) for t in times],
           "series": {v: [num(out.get(t)) for t in times] for v, out in sorted(series.items())}}
    last = (
        rows.filter(pl.col("value").is_not_null()).sort("time")
        .group_by("step", "element", maintain_order=True).last()
        .select("step", "element", "time", "value", "quality").sort("step", "element")
    )
    doc["last"] = [{"step": r["step"], "element": r["element"], "time": iso(r["time"]), "value": num(r["value"]),
                    "quality": r["quality"]} for r in last.to_dicts()]
    return doc


def station_file(sid: str, meta: dict, models: pl.DataFrame, parts, ai_parts, obs) -> dict:
    rows = models.filter(pl.col("station_id") == sid)
    times = sorted(rows["valid_time"].unique().to_list())
    index = {t: i for i, t in enumerate(times)}
    variables: dict[str, dict] = {}
    skip = {"model", "station_id", "valid_time", "run"}
    for variable in [c for c in rows.columns if c not in skip]:
        out: dict[str, list] = {}
        for model, t, v in rows.select("model", "valid_time", variable).iter_rows():
            if v is None or (isinstance(v, float) and math.isnan(v)):
                continue
            out.setdefault(model, [None] * len(times))[index[t]] = num(v)
        if out:
            variables[variable] = out
    doc = {"id": sid, **{k: meta[k] for k in ("name", "lat", "lon", "elev", "okres") if k in meta},
           "times": [iso(t) for t in times], "variables": variables,
           "runs": {m: iso(r) for m, r in rows.filter(pl.col("run").is_not_null()).group_by("model").agg(pl.col("run").max()).iter_rows()}}
    if parts is not None:
        p = parts.filter(pl.col("station_id") == sid).drop("station_id").sort("model", "day", "part")
        doc["parts"] = [{k: (str(v) if k == "day" else (num(v) if isinstance(v, float) else v)) for k, v in r.items()}
                        for r in p.to_dicts()]
    if ai_parts is not None:
        p = ai_parts.filter(pl.col("station_id") == sid).drop("station_id").sort("day", "part")
        doc["ai_parts"] = [{k: (str(v) if k == "day" else (num(v) if isinstance(v, float) else v)) for k, v in r.items()}
                           for r in p.to_dicts()]
    if obs is not None:
        doc["obs"] = observations(obs, sid)
    return doc


def pick_row(ledger_dir: Path) -> dict:
    rows = vp.read_ledger(ledger_dir)
    problems = vp.check_chain(rows)
    if problems:
        raise SystemExit("\n".join(f"FAIL {p}" for p in problems))
    gone = vp.read_withdrawn(ledger_dir)
    ok = [r for r in rows if r["status"] == "ok" and r["asset"]["tag"] not in gone]
    if not ok:
        raise SystemExit("no snapshot with a published asset yet")
    return ok[-1]


def write(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, ensure_ascii=False, separators=(",", ":")))


def build(row: dict, data: bytes, stations: dict, out: Path) -> dict:
    members = read_tar(data)
    allv = table(members, "okresy_all.parquet")
    if allv is None:  # a snapshot before okresy_all: the two-variable district table
        allv = table(members, "okresy.parquet")
    hourly, rain, sky = (table(members, f"{n}.parquet") for n in ("hourly", "rain", "sky"))
    models = table(members, "stations_models-public.parquet")
    parts, ai_parts = table(members, "stations_models_parts-public.parquet"), table(members, "stations_parts-public.parquet")
    obs = table(members, "obs-public.parquet")
    codes = sorted(allv["okres"].unique().to_list())
    for code in codes:
        write(out / "okresy" / f"{code}.json", okres_file(code, allv, hourly, rain, sky))
    by_id = {s["id"]: s for s in stations["stations"]}
    for sid in sorted(by_id):
        write(out / "stations" / f"{sid}.json", station_file(sid, by_id[sid], models, parts, ai_parts, obs))
    variables = sorted(set(allv["variable"].unique().to_list()) | (set(hourly["variable"].unique().to_list()) if hourly is not None else set()))
    index = {
        "version": 1, "kind": row.get("kind"), "policy": row.get("policy"),
        "snapshot": {"time": row["snapshot_time"], "issue_time": row["issue_time"], "model_version": row["model_version"],
                     "asset": row["asset"], "chain_hash": row["chain_hash"], "files": row["files"],
                     "public": row.get("public"), "withheld": row.get("withheld"), "derived": row.get("derived")},
        "okresy": codes, "stations": stations["stations"],
        "series": {"okresy": sorted(allv["series"].unique().to_list()),
                   "stations": sorted(models["model"].unique().to_list()) if models is not None else []},
        "variables": {v: UNITS.get(v, "") for v in variables},
        "files": sorted(members),
    }
    write(out / "index.json", index)
    return index


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ledger", type=Path, default=Path("ledger"))
    ap.add_argument("--stations", type=Path, default=Path("register/stations40.json"))
    ap.add_argument("--out", type=Path, default=Path("derived/explorer"))
    ap.add_argument("--snap-dir", type=Path, help="folder with the downloaded snap-*.tar (else: from the release)")
    ap.add_argument("--repo", default=REPO)
    args = ap.parse_args()
    row = pick_row(args.ledger)
    asset = row["asset"]
    if args.snap_dir and (args.snap_dir / asset["name"]).exists():
        data = (args.snap_dir / asset["name"]).read_bytes()
    else:
        data = vp.fetch(f"https://github.com/{args.repo}/releases/download/{asset['tag']}/{asset['name']}")
    if vp.sha256(data) != asset["sha256"]:
        raise SystemExit(f"FAIL {row['snapshot_time']}: {asset['name']} does not match the ledger's sha256")
    index = build(row, data, json.loads(args.stations.read_text()), args.out)
    size = sum(p.stat().st_size for p in args.out.rglob("*.json"))
    print(f"{args.out}: snapshot {row['snapshot_time']}, {len(index['okresy'])} districts, "
          f"{len(index['stations'])} stations, {size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())

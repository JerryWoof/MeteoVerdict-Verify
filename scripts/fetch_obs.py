#!/usr/bin/env python3
"""Freeze one UTC day of hourly ČHMÚ observations of the 40 main stations (standard library only).

Source: ČHMÚ open data, 10-minute files, https://opendata.chmi.cz/meteorology/climate/
(``now/data/10m-<station>-<YYYYMMDD>.json``, the last ~4 days, unchecked; older days from the
monthly ``recent/data/10min/<MM>/10m-<station>-<YYYYMM>.json``, checked). The hourly values are built
exactly as MeteoVerdict's own pipeline does (same cleaning, rule R5):

* ``temperature_2m`` (element T) and ``wind_speed_10m`` (F): the value at the full hour;
* ``precipitation``: sum of the six SRA10M values ending at the hour (HH-1:10 .. HH:00), only if
  all six are present; ``wind_gusts_10m``: largest of the six Fmax values of that hour;
* quality codes 0, 3, 5 are kept (1 suspect, 2 poor, 4 missing are dropped); values outside
  physically plausible limits are dropped.

* ``cloud_cover`` (hourly element N, octas x 12.5 %, 8 and more = 100 %, at the full hour) and
  ``sunshine_duration`` (SSV1H, tenths of the hour ending at the time, here seconds of that hour), from the
  hourly files ``1h-<station>-<day>.json`` (days frozen before 23 Oct 2026 do not have these two columns);

Output ``obs/obs-YYYYMMDD.csv`` (stations in the order of ``register/stations40.json``, hours
00:00..23:00 UTC, an empty cell = no valid value) and its line in ``obs/SHA256SUMS``. A frozen day
is never overwritten (the file's hash is what the scores cite). The day is **provisional** when it
comes from ``now/`` (unchecked data). Nothing is written if fewer than 35 stations delivered
temperature for at least 20 hours (data not published yet): run again later.

    python scripts/fetch_obs.py --day 2026-10-21
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

BASE = "https://opendata.chmi.cz/meteorology/climate"
ELEMENTS = {"T": "temperature_2m", "F": "wind_speed_10m", "SRA10M": "precipitation", "Fmax": "wind_gusts_10m"}
KEEP_QUALITY = (0, 3, 5)
PLAUSIBLE = {"temperature_2m": (-50, 50), "wind_speed_10m": (0, 75), "precipitation": (0, 120), "wind_gusts_10m": (0, 75)}
VARIABLES = ("temperature_2m", "wind_speed_10m", "precipitation", "wind_gusts_10m", "cloud_cover", "sunshine_duration")
UTC = timezone.utc


def get_json(url: str) -> dict | None:
    """One ČHMÚ JSON file, None if it does not exist or stays unreadable after retries."""
    for wait in (0, 3, 10, 30):
        time.sleep(wait)
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "mv-verify"}), timeout=60) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
        except (urllib.error.URLError, ValueError, TimeoutError):
            pass
    return None


def day_rows(wsi: str, day: date, today: date, checked_only: bool = False) -> tuple[list[list], str]:
    """The 10-minute rows of ``day`` and where they came from (``now`` or ``recent``);
    ``checked_only`` skips ``now/`` (unchecked) and takes the monthly ``recent/`` file only."""
    if (today - day).days <= 3 and not checked_only:
        doc = get_json(f"{BASE}/now/data/10m-{wsi}-{day:%Y%m%d}.json")
        if doc:
            return doc["data"]["data"]["values"], "now"
    doc = get_json(f"{BASE}/recent/data/10min/{day.month:02d}/10m-{wsi}-{day:%Y%m}.json")
    if doc:
        values = doc["data"]["data"]["values"]
        return [r for r in values if r[2].startswith(day.isoformat())], "recent"
    return [], "none"


def day_rows_1h(wsi: str, day: date, today: date, checked_only: bool = False) -> list[list]:
    """The hourly rows (elements N, SSV1H, ...) of ``day``: ``now/`` for the last days, else the monthly file."""
    if (today - day).days <= 3 and not checked_only:
        doc = get_json(f"{BASE}/now/data/1h-{wsi}-{day:%Y%m%d}.json")
        if doc:
            return doc["data"]["data"]["values"]
    doc = get_json(f"{BASE}/recent/data/1hour/{day.month:02d}/1h-{wsi}-{day:%Y%m}.json")
    if doc:
        return [r for r in doc["data"]["data"]["values"] if r[2].startswith(day.isoformat())]
    return []


def hourly_sky(rows: list[list], day: date) -> dict[datetime, dict[str, float]]:
    """Cloud cover (%) and sunshine (s of the hour ending at the time) of ``day`` from hourly rows."""
    out: dict[datetime, dict[str, float]] = {}
    for _station, element, dt, val, _flag, quality in rows:
        if element not in ("N", "SSV1H") or val in ("", None) or int(quality) not in KEEP_QUALITY:
            continue
        t = datetime.strptime(dt, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
        if t.date() != day:
            continue
        v = float(val)
        if element == "N":
            out.setdefault(t, {})["cloud_cover"] = 100.0 if v >= 8 else v * 12.5
        else:
            out.setdefault(t, {})["sunshine_duration"] = min(max(v / 10, 0.0), 1.0) * 3600
    return out


def hourly(rows: list[list], day: date) -> dict[datetime, dict[str, float]]:
    """Hourly values of ``day`` (UTC hours 00..23) from 10-minute rows of day-1 and day."""
    slots: dict[tuple[str, datetime], float] = {}
    for _station, element, dt, val, _flag, quality in rows:
        var = ELEMENTS.get(element)
        if var is None or val in ("", None) or int(quality) not in KEEP_QUALITY:
            continue
        slots[(var, datetime.strptime(dt, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC))] = float(val)
    out: dict[datetime, dict[str, float]] = {}
    start = datetime(day.year, day.month, day.day, tzinfo=UTC)
    for h in range(24):
        t = start + timedelta(hours=h)
        cell: dict[str, float] = {}
        for var in ("temperature_2m", "wind_speed_10m"):
            if (var, t) in slots:
                cell[var] = slots[(var, t)]
        for var, agg in (("precipitation", sum), ("wind_gusts_10m", max)):
            six = [slots.get((var, t - timedelta(minutes=10 * i))) for i in range(6)]
            if all(v is not None for v in six):
                cell[var] = agg(six)
        for var, v in list(cell.items()):
            low, high = PLAUSIBLE[var]
            if not low <= v <= high:
                del cell[var]
        out[t] = cell
    return out


def fetch_day(stations: list[str], day: date, today: date, checked_only: bool = False) -> tuple[dict[str, dict], dict[str, str]]:
    def one(wsi: str):
        before, s1 = day_rows(wsi, day - timedelta(days=1), today, checked_only)
        now, s2 = day_rows(wsi, day, today, checked_only)
        cells = hourly(before + now, day)
        for t, extra in hourly_sky(day_rows_1h(wsi, day, today, checked_only), day).items():
            cells.setdefault(t, {}).update(extra)
        return wsi, cells, s2

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(one, stations))
    return {w: h for w, h, _ in results}, {w: s for w, _, s in results}


def fmt(v: float | None) -> str:
    return "" if v is None else f"{round(v + 0.0, 1):.1f}"


def to_csv(stations: list[str], hours: dict[str, dict]) -> bytes:
    lines = ["station_id,time," + ",".join(VARIABLES)]
    for wsi in stations:
        for t, cell in sorted(hours[wsi].items()):
            lines.append(f"{wsi},{t:%Y-%m-%dT%H:%M:%SZ}," + ",".join(fmt(cell.get(v)) for v in VARIABLES))
    return ("\n".join(lines) + "\n").encode()


def update_sums(folder: Path) -> None:
    """``obs/SHA256SUMS`` (``sha256sum -c`` format) of every frozen day, sorted by name."""
    lines = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}" for p in sorted(folder.glob("obs-*.csv"))]  # incl. .final
    (folder / "SHA256SUMS").write_text("\n".join(lines) + "\n")


def final_versions(out: Path, day: date) -> list[Path]:
    """The checked files of ``day``, oldest first: ``.final``, ``.final2``, ``.final3``, ..."""
    stem = f"obs-{day:%Y%m%d}"
    found = [p for p in out.glob(f"{stem}.final*.csv")]
    return sorted(found, key=lambda p: (len(p.name), p.name))


def final_day(args, day: date, today: date) -> int:
    """Freeze the checked values of a day (rule R5: provisional until checked). The provisional file
    stays untouched (the first scores cite its hash); an existing checked file is never overwritten:
    if the checked values changed later, the next number is used (``.final2``). Exit 2: not published
    in ``recent/`` yet (try again later)."""
    if not (args.out / f"obs-{day:%Y%m%d}.csv").exists():
        print("no provisional file for this day: freeze it first (without --final)", file=sys.stderr)
        return 2
    ids = [s["id"] for s in json.loads(args.stations.read_text())["stations"]]
    probe, source = day_rows(ids[0], day, today, checked_only=True)
    if not probe:
        print("ČHMÚ recent/ does not have this day yet; nothing written", file=sys.stderr)
        return 2
    hours, source = fetch_day(ids, day, today, checked_only=True)
    good = sum(1 for w in ids if sum(1 for c in hours[w].values() if "temperature_2m" in c) >= 20)
    print(f"{good}/{len(ids)} stations with checked temperature in at least 20 hours")
    if good < 35:
        print("too few stations in recent/: nothing written", file=sys.stderr)
        return 2
    data = to_csv(ids, hours)
    have = final_versions(args.out, day)
    if have and have[-1].read_bytes() == data:
        print(f"{have[-1].name} is up to date")
        return 0
    name = f"obs-{day:%Y%m%d}.final.csv" if not have else f"obs-{day:%Y%m%d}.final{len(have) + 1}.csv"
    (args.out / name).write_bytes(data)
    update_sums(args.out)
    print(f"{args.out / name}: {len(data)} bytes, sha256 {hashlib.sha256(data).hexdigest()}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--day", required=True, help="UTC day YYYY-MM-DD (a complete, finished day)")
    ap.add_argument("--stations", type=Path, default=Path("register/stations40.json"))
    ap.add_argument("--out", type=Path, default=Path("obs"))
    ap.add_argument("--final", action="store_true", help="the checked version of a day already frozen provisionally: values from "
                    "ČHMÚ recent/ (checked) as obs-YYYYMMDD.final.csv (a later change of a checked value: .final2.csv, ...)")
    ap.add_argument("--partial", action="store_true", help="tests only: take a day still in progress (needs --out other than obs)")
    args = ap.parse_args()
    if args.partial and args.out.name == "obs":
        print("--partial never writes into obs/ (frozen days are complete days)", file=sys.stderr)
        return 2
    day = date.fromisoformat(args.day)
    today = datetime.now(UTC).date()
    if day >= today and not args.partial:
        print("the day is not finished yet", file=sys.stderr)
        return 2
    if args.final:
        return final_day(args, day, today)
    path = args.out / f"obs-{day:%Y%m%d}.csv"
    if path.exists():
        print(f"{path} is frozen already (sha256 {hashlib.sha256(path.read_bytes()).hexdigest()[:16]}…)")
        return 0
    ids = [s["id"] for s in json.loads(args.stations.read_text())["stations"]]
    hours, source = fetch_day(ids, day, today)
    good = sum(1 for w in ids if sum(1 for c in hours[w].values() if "temperature_2m" in c) >= (1 if args.partial else 20))
    print(f"{good}/{len(ids)} stations with temperature in at least 20 hours; sources: "
          + ", ".join(f"{k} {list(source.values()).count(k)}" for k in sorted(set(source.values()))))
    if good < 35 and not args.partial:
        print("too few stations: the data are not published yet; nothing written", file=sys.stderr)
        return 2
    args.out.mkdir(parents=True, exist_ok=True)
    data = to_csv(ids, hours)
    path.write_bytes(data)
    update_sums(args.out)
    print(f"{path}: {len(data)} bytes, sha256 {hashlib.sha256(data).hexdigest()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

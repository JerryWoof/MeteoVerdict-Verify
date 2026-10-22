#!/usr/bin/env python3
"""Verify the MeteoVerdict forecast log with nothing but the Python standard library.

What it checks, for every snapshot row of the ledger (``ledger/*.jsonl``):

1. the hash chain: ``chain_hash = sha256(prev_hash + canonical JSON of the row without
   chain_hash)``, every row starts from the previous row's ``chain_hash`` (genesis: 64 zeros),
   snapshot times only go forward. A changed, inserted or deleted row breaks the chain.
2. with ``--download``: the release asset of each snapshot (a tar) is downloaded and its sha256
   equals the ledger's; ``manifest.json`` inside hashes to ``content_sha256``; every file inside
   equals the hash in the ledger; the village register file in ``register/`` matches. Rows with
   ``policy`` ``public-v1`` publish only the files in ``public``: the files in ``withheld``
   (the villages, MeteoVerdict AI at the stations other than the ČHMÚ synoptic ones) are
   committed by their sha256 in ``files`` and in ``manifest.json`` but not published;
   ``derived`` gives the hash and rule of a public file cut from a withheld one. Every committed
   file is either public or withheld; ``public_stations`` (version, count, sha256 of the sorted
   ids) matches the fixed list ``register/stations<N>.json`` of that version, and, if pyarrow
   is installed, the public station files hold no other station. The village register is
   withheld too (its hashes are in ``register``).
3. with ``--remote``: GitHub's own digest of each release asset equals the ledger's.
4. it lists the missing snapshots (the outage log): a snapshot time without an issue at most
   6 hours old is written as ``missing`` and counted, never filled in afterwards.

It does not check the timestamps (OpenTimestamps) or the immutable-release attestation, see
README.md for the two commands. Exit code 0 only if everything checked is in order.

    python scripts/verify_public.py                      # chain of the ledger in this checkout
    python scripts/verify_public.py --download --days 3  # also the data of the last 3 days
    python scripts/verify_public.py --ledger test-ledger --prefix test-snap ...
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import tarfile
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = "JerryWoof/MeteoVerdict-Verify"
GENESIS = "0" * 64
CUTOFFS = (0, 6, 12, 18)


def canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def chain_hash(prev: str, row: dict) -> str:
    body = {k: v for k, v in row.items() if k != "chain_hash"}
    return sha256(prev.encode() + canonical(body))


def parse_time(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)


def read_ledger(folder: Path) -> list[dict]:
    rows = []
    for path in sorted(folder.glob("*.jsonl")):
        for n, line in enumerate(path.read_text().splitlines(), 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except ValueError as error:
                    raise SystemExit(f"FAIL {path.name}:{n} is not JSON: {error}") from None
    return rows


def check_chain(rows: list[dict]) -> list[str]:
    problems, prev, last = [], GENESIS, None
    for i, row in enumerate(rows):
        t = row.get("snapshot_time", "?")
        if row.get("prev_hash") != prev:
            problems.append(f"row {i} ({t}): prev_hash is not the chain hash of the row before")
        if row.get("chain_hash") != chain_hash(row.get("prev_hash", ""), row):
            problems.append(f"row {i} ({t}): chain_hash does not match the row's content")
        if last and parse_time(t) <= parse_time(last):
            problems.append(f"row {i} ({t}): not later than the row before ({last})")
        if parse_time(t).hour not in CUTOFFS or parse_time(t).minute or parse_time(t).second:
            problems.append(f"row {i} ({t}): not a 00/06/12/18 UTC snapshot time")
        prev, last = row.get("chain_hash", ""), t
    return problems


def fetch(url: str, token: str | None = None) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "mv-verify"})
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.read()


def check_asset(row: dict, repo: str, local_register: Path | None) -> list[str]:
    asset, t = row["asset"], row["snapshot_time"]
    url = f"https://github.com/{repo}/releases/download/{asset['tag']}/{asset['name']}"
    try:
        data = fetch(url)
    except urllib.error.URLError as error:
        return [f"{t}: asset not downloadable ({error})"]
    problems = []
    if sha256(data) != asset["sha256"]:
        return [f"{t}: asset sha256 differs from the ledger"]
    if len(data) != asset["bytes"]:
        problems.append(f"{t}: asset size differs from the ledger")
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        members = {m.name: tar.extractfile(m).read() for m in tar.getmembers() if m.isfile()}
    manifest = members.pop("manifest.json", None)
    if manifest is None or sha256(manifest) != row["content_sha256"]:
        problems.append(f"{t}: manifest.json does not hash to content_sha256")
    if "public" in row:
        # policy public-v1: the tar has the public files only; each hashes to the ledger
        # (the store's hash, or ``derived`` for a file cut from a withheld one); the withheld
        # files are committed by their hash in ``files`` and are not published
        want = {n: row["files"].get(n) or (row.get("derived", {}).get(n) or {}).get("sha256") for n in row["public"]}
        if {n: sha256(b) for n, b in members.items()} != want:
            problems.append(f"{t}: public files in the tar differ from the hashes in the ledger")
        if set(members) & set(row.get("withheld", [])):
            problems.append(f"{t}: a withheld file is in the tar")
        listed = json.loads(manifest)["files"] if manifest else {}
        if {n: i["sha256"] for n, i in listed.items()} != row["files"]:
            problems.append(f"{t}: manifest.json does not list the ledger's file hashes")
        unaccounted = set(row["files"]) - set(row["public"]) - set(row.get("withheld", []))
        if unaccounted:
            problems.append(f"{t}: committed files neither public nor withheld: {sorted(unaccounted)}")
        if row.get("public_stations"):
            problems += check_public_stations(row, members, repo, local_register)
    elif {n: sha256(b) for n, b in members.items()} != row["files"]:
        problems.append(f"{t}: files in the tar differ from the hashes in the ledger")
    reg = row.get("register")
    if reg and "public" not in row:  # before public-v1 the village register was published
        path = (local_register or Path("register")) / f"obce-{reg['version']}.parquet"
        try:
            blob = path.read_bytes() if path.exists() else fetch(
                f"https://raw.githubusercontent.com/{repo}/main/register/{path.name}")
            if sha256(blob) != reg["file_sha256"]:
                problems.append(f"{t}: village register {reg['version']} differs from the ledger")
        except urllib.error.URLError:
            problems.append(f"{t}: village register {reg['version']} not found")
    return problems


def station_list(version: int, repo: str, local_register: Path | None) -> list[str] | None:
    """The sorted ids of the fixed public station list ``version`` (``register/stations*.json``),
    from this checkout or the repository's main branch; None if there is none."""
    folder = local_register or Path("register")
    docs = [json.loads(p.read_text()) for p in sorted(folder.glob("stations*.json"))] if folder.is_dir() else []
    if not any(d.get("version") == version for d in docs):
        try:
            docs.append(json.loads(fetch(f"https://raw.githubusercontent.com/{repo}/main/register/stations40.json")))
        except (urllib.error.URLError, ValueError):
            pass
    doc = next((d for d in docs if d.get("version") == version), None)
    return sorted(s["id"] for s in doc["stations"]) if doc else None


def check_public_stations(row: dict, members: dict, repo: str, local_register: Path | None) -> list[str]:
    t, ref = row["snapshot_time"], row["public_stations"]
    ids = station_list(ref["version"], repo, local_register)
    if ids is None:
        return [f"{t}: public station list version {ref['version']} not found in register/"]
    problems = []
    if sha256(canonical(ids)) != ref["sha256"] or len(ids) != ref["count"]:
        problems.append(f"{t}: public_stations does not match the station list version {ref['version']}")
    try:
        import pyarrow.parquet as pq
    except ImportError:  # optional: the hashes above already fix the files
        return problems
    for name in row.get("derived", {}):
        if name in members:
            try:
                found = set(pq.read_table(io.BytesIO(members[name]), columns=["station_id"]).column(0).to_pylist())
            except Exception as error:  # noqa: BLE001
                problems.append(f"{t}: {name} is not a readable Parquet file with station_id ({error})")
                continue
            if found - set(ids):
                problems.append(f"{t}: {name} holds stations outside the public list: {sorted(found - set(ids))[:5]}")
    return problems


def check_remote_digest(row: dict, repo: str, token: str | None) -> list[str]:
    asset = row["asset"]
    try:
        rel = json.loads(fetch(f"https://api.github.com/repos/{repo}/releases/tags/{asset['tag']}", token))
    except urllib.error.URLError as error:
        return [f"{row['snapshot_time']}: release {asset['tag']} not found ({error})"]
    found = next((a for a in rel["assets"] if a["name"] == asset["name"]), None)
    if not found:
        return [f"{row['snapshot_time']}: release {asset['tag']} has no asset {asset['name']}"]
    problems = []
    if found.get("digest") != f"sha256:{asset['sha256']}":
        problems.append(f"{row['snapshot_time']}: GitHub's digest of the asset differs from the ledger")
    if rel.get("draft"):
        problems.append(f"{row['snapshot_time']}: release {asset['tag']} is still a draft")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ledger", type=Path, default=Path("ledger"), help="ledger folder (default: ledger)")
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--download", action="store_true", help="also download and check the snapshot files")
    ap.add_argument("--remote", action="store_true", help="also compare with GitHub's asset digests")
    ap.add_argument("--days", type=int, default=0, help="only the last N days for --download/--remote (0: all)")
    ap.add_argument("--token", help="GitHub token for --remote (only to avoid rate limits)")
    args = ap.parse_args()

    rows = read_ledger(args.ledger)
    problems = check_chain(rows)
    ok = [r for r in rows if r["status"] == "ok"]
    missing = [r["snapshot_time"] for r in rows if r["status"] != "ok"]
    print(f"{len(rows)} snapshot rows ({len(ok)} ok, {len(missing)} missing); chain "
          + ("BROKEN" if problems else "intact") + (f"; head {rows[-1]['chain_hash'][:16]}…" if rows else ""))
    if missing:
        print("missing snapshots (outage log): " + ", ".join(missing))
    if rows:  # a gap in the times without a row is a deleted row or a failed run that left no trace
        t, seen = parse_time(rows[0]["snapshot_time"]), {r["snapshot_time"] for r in rows}
        while t <= parse_time(rows[-1]["snapshot_time"]):
            if t.strftime("%Y-%m-%dT%H:%M:%SZ") not in seen:
                problems.append(f"no row for the snapshot time {t:%Y-%m-%dT%H:%MZ}")
            t += timedelta(hours=6)
    recent = ok
    if args.days:
        since = parse_time(rows[-1]["snapshot_time"]) - timedelta(days=args.days) if rows else None
        recent = [r for r in ok if parse_time(r["snapshot_time"]) > since]
    if args.remote:
        for r in recent:
            problems += check_remote_digest(r, args.repo, args.token)
    if args.download:
        for r in recent:
            found = check_asset(r, args.repo, args.ledger.parent / "register")
            print(f"  {r['snapshot_time']}  {'ok' if not found else 'PROBLEM'}  {r['asset']['bytes'] / 1e6:.1f} MB")
            problems += found
    for p in problems:
        print("FAIL", p)
    print("RESULT:", "everything checked is in order" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())

# MeteoVerdict-Verify

Public, timestamped log of the forecasts of [MeteoVerdict](https://meteoverdict.cz) (its own AI model, our blend and the public weather models) for Czechia, with hashes, third-party timestamps and a script anyone can run to check that the log was not changed after the fact.

**Status:** set-up phase. The first official snapshot will appear as the first row of `ledger/`. Until then this repository only holds the tooling.

**Only the four daily snapshots are verified.** MeteoVerdict issues new forecasts more often than that (whenever a weather model has a new run) and the website shows the newest one. The in-between issues are *not* part of this log, are not scored here and cannot be verified from here. They are kept in our own backups only.

## Licence

- Forecast data and other published outputs: **CC BY-SA 4.0** (`LICENSE`). They derive from forecasts of several providers, including UK Met Office data, which requires ShareAlike. Credit the data providers (ČHMÚ, Open-Meteo, DWD, ECMWF and others, see the [licensing page](https://github.com/JerryWoof/MeteoVerdict/blob/main/docs/licensing.md)).
- Code and scripts: **MIT** (`LICENSE-CODE`).

## How to verify

1. **What is claimed, and the rules.** The AI forecast is "verified" only by the rules written down *before* the live series started, in [`preregistration/`](preregistration/) (date and sha256 are in its header; its git commit and OpenTimestamps proof date it independently of us). Until 60 days of live snapshots exist and the rules' conditions are met, nothing here claims that the AI works.
2. **What is in the repository, and where the data are.**
   - `ledger/YYYY-MM-DD.jsonl`: one line per official snapshot (00, 06, 12, 18 UTC) with the issue used, its age, `model_version`, the sha256 of every file, the release asset and the chain hash. A snapshot with no issue at most 6 hours old is written as `status: "missing"`; no older issue is substituted.
   - Releases `snap-YYYYMMDDTHHMMZ`: one **immutable** release per snapshot with one asset, a tar (`snap-…tar`) holding `manifest.json` and the **public** Parquet files: `okresy` (77 districts: AI, our blend `combined` and the public models, hourly to about +5 days), `stations-public` (the AI at the 40 ČHMÚ synoptic stations, WMO ids `0-20000-0-…`), the public models at the stations, `hourly` (AI precipitation, sunshine, cloud cover), `rain`, `sky`.
   - **Committed, not published** (since policy `public-v1`, 22 Oct 2026): the villages (`obce`: about 6,300 villages, AI, blend and every public model, hourly) and the AI at the other stations (`stations`). Their sha256 is in the ledger row (`files`, listed under `withheld`) and in `manifest.json`, so they are fixed at the snapshot time and cannot be changed afterwards, but the data are not released: they can be scored only where there are measurements, and the full history would make it easy to imitate the model. The village register (`register/obce-<version>.parquet`: names, positions, heights) stays public.
   - Releases `ledger-YYYY-MM-DD`: the day's ledger closed as an immutable release.
   - `stamps/*.head.ots`: OpenTimestamps proofs of the chain head after each snapshot.
3. **Observations and station list.** The reference is the ČHMÚ station network (open data); the district value is the mean of its measuring stations. *Planned:* the fixed station and district-pool list (versioned, with a hash) and the score script's cleaning rules will be added here before the scoring starts.
4. **The public models' runs.** The log stores, per snapshot, the model values we used. The same runs can be fetched from [Open-Meteo](https://open-meteo.com) (model run archive / Previous Runs) and the providers' open data. *Planned:* the run times and availability times per snapshot (`release_inputs`) will be added to the ledger.
5. **Timeline and pairing rule.** The time axis is the snapshot time, not the model run. For a snapshot only the newest issue with `issue_time` ≤ snapshot time counts (`staleness_h` in the ledger); a public model contributes only runs that were available at that time. Lead bands follow `valid_time − snapshot time` (D+1 = 24–47 h and so on). Only hours where every compared series has a value count; no gaps are filled and no older run replaces a missing one.
6. **The check script.** `python3 scripts/verify_public.py` (standard library only) recomputes the chain of the whole ledger and lists the missing snapshots. With `--download --days N` it also downloads the data of the last N days and checks every file against the ledger; with `--remote` it compares GitHub's own digest of each asset. *Planned:* a scoring mode reproducing the numbers of the website from the downloaded snapshots and the ČHMÚ data (deterministic bootstrap seed).
7. **Hashes and timestamps.**
   - Hash: `sha256sum snap-….tar` must equal `asset.sha256` of its ledger row; `chain_hash = sha256(prev_hash + canonical JSON of the row without chain_hash)` (canonical = sorted keys, no spaces, UTF-8), the first `prev_hash` is 64 zeros.
   - Immutable release: `gh release verify snap-YYYYMMDDTHHMMZ -R JerryWoof/MeteoVerdict-Verify` (GitHub's attestation) and `gh release verify-asset snap-… snap-….tar`.
   - OpenTimestamps (Bitcoin): `ots verify stamps/<time>.head.ots` (needs a Bitcoin node) or drop the `.head` file and its `.ots` on <https://opentimestamps.org>. A proof is *pending* for some hours after the stamp and complete once `stamp.yml` upgraded it. The chain hash in a `.head` file covers every earlier row, so it dates the whole history before it.
   - Commits of the ledger are signed with an SSH key of the publisher (shown as "Verified" on GitHub); `main` cannot be force-pushed or deleted.
   - A weekly job asks the Internet Archive to save the repository page and ledger (`wayback.yml`).
8. **Limits and weaknesses, openly.**
   - Only the four snapshots a day are verified; intermediate issues, shown on the website between snapshots, are not (see above).
   - The ledger proves *when* a snapshot existed and that it was not altered afterwards. It cannot prove that the AI had no knowledge of the future: that rests on the model's training rules and the rules in the pre-registration.
   - The AI's code, weights and features are private; this log holds outputs only.
   - Our blend and the public models' values at villages and stations are derived by our own (private) code, so the log is the only record of them.
   - Failed or late runs show up as `missing` snapshots, counted and never repaired afterwards. Forecast issues from 19–22 October 2026 are anchored late (planned): their timestamps date the anchoring, not the creation.
   - Large changes of the model are a new `model_version`; scores are reported per version and the day count starts again.
9. **Report a discrepancy.** Open an issue in this repository with the snapshot time, what you recomputed and what differs, or write to info@meteoverdict.cz.

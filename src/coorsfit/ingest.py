"""Pull raw data into data/raw: MLB schedules, Statcast pitches one day at a time, sprint speed.

Every response is cached to parquet, so rerunning only fetches what is missing. Statcast days
are checked against the schedule: a day that comes back without any of its scheduled games is
treated as a failed request and retried rather than cached as empty.

Run with `python -m coorsfit.ingest`.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from pybaseball import statcast, statcast_sprint_speed

from coorsfit.config import data_path, load_config

log = logging.getLogger(__name__)

SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
PEOPLE_URL = "https://statsapi.mlb.com/api/v1/people"
PLAYED_STATE = "F"  # codedGameState for Final and Completed Early; D postponed, C cancelled


def with_retries[T](fetch: Callable[[], T], what: str) -> T:
    cfg = load_config()["ingest"]
    for attempt in range(cfg["max_retries"]):
        try:
            return fetch()
        except Exception as e:  # network drops surface as several exception types
            wait = min(cfg["backoff_max_seconds"], cfg["backoff_base_seconds"] * 2**attempt)
            log.warning(
                "%s failed (%s: %s); retry %d in %ds", what, type(e).__name__, e, attempt + 1, wait
            )
            time.sleep(wait)
    raise RuntimeError(f"{what}: gave up after {cfg['max_retries']} attempts")


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write via a temp file so an interrupted run never leaves a partial cache entry."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def fetch_schedule(season: int) -> pd.DataFrame:
    """Every regular-season schedule entry, one row per entry (postponed games appear twice)."""
    path = data_path("raw") / "schedule" / f"{season}.parquet"
    if path.exists():
        return pd.read_parquet(path)

    def get() -> dict:
        params = {"sportId": 1, "season": season, "gameType": "R"}
        r = requests.get(SCHEDULE_URL, params=params, timeout=60)
        r.raise_for_status()
        return r.json()

    games = [g for d in with_retries(get, f"schedule {season}")["dates"] for g in d["games"]]
    df = pd.DataFrame(
        {
            "season": season,
            "game_pk": g["gamePk"],
            "official_date": g["officialDate"],
            "coded_state": g["status"]["codedGameState"],
            "detailed_state": g["status"]["detailedState"],
            "venue_id": g["venue"]["id"],
            "venue_name": g["venue"]["name"],
            "home_team_id": g["teams"]["home"]["team"]["id"],
            "away_team_id": g["teams"]["away"]["team"]["id"],
        }
        for g in games
    )
    write_parquet(df, path)
    return df


def played_entries(schedule: pd.DataFrame) -> pd.DataFrame:
    """Schedule entries for games that were played. A suspended game can have one per day it ran."""
    return schedule[schedule["coded_state"] == PLAYED_STATE]


def fetch_statcast_day(season: int, day: str, expected: set[int]) -> bool:
    """Cache one day of Statcast pitches. Returns True if it hit the network."""
    path = data_path("raw") / "statcast" / str(season) / f"{day}.parquet"
    if path.exists():
        return False

    def get() -> pd.DataFrame:
        df = statcast(day, day, verbose=False, parallel=False)
        got = set(df["game_pk"]) if len(df) else set()
        if expected and not got & expected:
            raise ValueError(f"none of {len(expected)} scheduled games returned")
        if missing := expected - got:
            log.warning(
                "%s: %d scheduled games missing from Statcast: %s",
                day,
                len(missing),
                sorted(missing),
            )
        return df

    write_parquet(with_retries(get, f"statcast {day}"), path)
    return True


def fetch_sprint_speed(season: int) -> None:
    path = data_path("raw") / "sprint_speed" / f"{season}.parquet"
    if not path.exists():
        # min_opp=1 keeps every runner; the qualifying threshold is applied in features.
        write_parquet(
            with_retries(lambda: statcast_sprint_speed(season, 1), f"sprint speed {season}"), path
        )


def fetch_names(ids: list[int]) -> pd.Series:
    """Player names by MLBAM id from the MLB Stats API, cached; only missing ids are fetched."""
    path = data_path("raw") / "people.parquet"
    cached = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["id", "name"])
    missing = sorted(set(ids) - set(cached["id"]))
    for start in range(0, len(missing), 100):
        batch = ",".join(str(i) for i in missing[start : start + 100])

        def get(batch: str = batch) -> list[dict]:
            r = requests.get(PEOPLE_URL, params={"personIds": batch}, timeout=60)
            r.raise_for_status()
            return r.json()["people"]

        people = with_retries(get, "people")
        found = pd.DataFrame(
            {"id": [p["id"] for p in people], "name": [p["fullName"] for p in people]}
        )
        cached = pd.concat([cached, found], ignore_index=True)
    if missing:
        write_parquet(cached.astype({"id": "int64"}), path)
    return cached.set_index("id")["name"]


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )
    first, last = load_config()["seasons"]["pull"]
    failures = []
    for season in range(first, last + 1):
        played = played_entries(fetch_schedule(season))
        expected = played.groupby("official_date")["game_pk"].agg(set)
        start, fetched = time.time(), 0
        # Savant takes about 5 s per uncached day, so a few requests run at once.
        with ThreadPoolExecutor(max_workers=load_config()["ingest"]["workers"]) as pool:
            jobs = [
                pool.submit(fetch_statcast_day, season, day, games)
                for day, games in expected.items()
            ]
            for job in as_completed(jobs):
                try:
                    fetched += job.result()
                except RuntimeError as e:
                    failures.append(str(e))
                    log.error("%s", e)
        days = expected.index
        fetch_sprint_speed(season)
        log.info(
            "%d: %d game days, %d fetched, %.0fs", season, len(days), fetched, time.time() - start
        )
    if failures:
        raise SystemExit(f"{len(failures)} days failed; rerun `make data` to retry them")


if __name__ == "__main__":
    main()

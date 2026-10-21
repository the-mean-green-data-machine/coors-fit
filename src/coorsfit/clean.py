"""Build cleaned pitch tables and the plate-appearance table from the raw pulls.

Outputs:
- data/interim/pitches_{season}.parquet: regular-season pitches, deduplicated, with venue flags
- data/processed/pa.parquet: one row per plate appearance (the pitch that ends it)
- reports/tables/data_summary.md: per-season counts, also printed

Run with `python -m coorsfit.clean`.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from coorsfit.config import ROOT, data_path, load_config
from coorsfit.ingest import played_entries

log = logging.getLogger(__name__)

INT_COLUMNS = [
    "game_pk", "game_year", "at_bat_number", "pitch_number", "inning",
    "batter", "pitcher", "balls", "strikes", "outs_when_up",
]  # fmt: skip
FLOAT_COLUMNS = [
    "release_speed", "pfx_x", "pfx_z", "plate_x", "plate_z",
    "launch_speed", "launch_angle", "hit_distance_sc", "hc_x", "hc_y",
    "estimated_woba_using_speedangle", "woba_value", "woba_denom", "babip_value", "iso_value",
    "delta_run_exp",
]  # fmt: skip
STR_COLUMNS = [
    "game_type", "inning_topbot", "home_team", "away_team", "stand", "p_throws",
    "pitch_type", "description", "type", "events", "bb_type",
]  # fmt: skip
PITCH_COLUMNS = ["game_date", *INT_COLUMNS, *FLOAT_COLUMNS, *STR_COLUMNS]
PITCH_KEY = ["game_pk", "at_bat_number", "pitch_number"]
PA_KEY = ["game_pk", "at_bat_number"]

STRIKEOUTS = {"strikeout", "strikeout_double_play"}
HITS = {"single", "double", "triple", "home_run"}
# Events Savant leaves out of wOBA's denominator. Unlike the textbook formula, Savant counts
# catcher's interference (denominator 1, value 0.7), and woba_value comes from Savant.
NOT_IN_WOBA = {"sac_bunt", "sac_bunt_double_play", "intent_walk", "truncated_pa"}


def load_raw_season(season: int) -> pd.DataFrame:
    files = sorted((data_path("raw") / "statcast" / str(season)).glob("*.parquet"))
    frames = [pd.read_parquet(f) for f in files]
    return pd.concat(
        [f.reindex(columns=PITCH_COLUMNS) for f in frames if len(f)], ignore_index=True
    )


def venues(schedule: pd.DataFrame) -> pd.DataFrame:
    """One row per played game with its venue. A suspended game keeps its first venue."""
    played = played_entries(schedule).sort_values("official_date")
    return played.drop_duplicates("game_pk")[["game_pk", "venue_id"]]


def clean_pitches(raw: pd.DataFrame, game_venues: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Regular-season pitches with plain dtypes, no duplicate pitches, and Coors/team flags."""
    cfg = load_config()["coors"]
    df = raw[raw["game_type"] == "R"].copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    # Raw days can disagree on dtype (an all-null column reads as object), so cast via the
    # nullable types, which accept both NaN and pd.NA.
    for c in INT_COLUMNS:
        col = df[c].astype("Int64")
        if col.isna().any():
            raise ValueError(f"{c} is missing on {col.isna().sum()} pitches")
        df[c] = col.astype("int64")
    for c in FLOAT_COLUMNS:
        df[c] = df[c].astype("Float64").to_numpy(dtype="float64", na_value=np.nan)
    for c in STR_COLUMNS:
        df[c] = df[c].astype("str")

    dup = df.duplicated(PITCH_KEY, keep="first")
    stats = {
        "dup_pitches_exact": int(df.duplicated(keep="first").sum()),
        "dup_pitches_dropped": int(dup.sum()),
    }
    df = df[~dup]

    df = df.merge(game_venues, on="game_pk", how="left", validate="many_to_one")
    df["season"] = df["game_year"]
    df["is_coors"] = df["venue_id"] == cfg["venue_id"]
    bat_home = df["inning_topbot"] == "Bot"
    df["bat_team"] = df["home_team"].where(bat_home, df["away_team"])
    df["fld_team"] = df["away_team"].where(bat_home, df["home_team"])
    # At Coors, a Rockies hitter is the home player; everyone else at Coors is a visitor.
    df["home_player"] = df["is_coors"] & (df["bat_team"] == cfg["team"])
    return df.sort_values(PITCH_KEY, ignore_index=True), stats


def build_pa_table(pitches: pd.DataFrame) -> pd.DataFrame:
    """One row per plate appearance, taken from the pitch that ends it (the row with an event)."""
    n_pitches = pitches.groupby(PA_KEY).size().rename("n_pitches")
    ends = pitches[pitches["events"].notna()]
    # If a PA ever carried two event rows, the later pitch is the one that ended it.
    pa = ends.sort_values(PITCH_KEY).drop_duplicates(PA_KEY, keep="last")
    pa = pa.join(n_pitches, on=PA_KEY)
    pa = pa.drop(columns=["pitch_number", "balls", "strikes", "description", "game_type"])
    # Savant leaves woba_denom blank on some ordinary PAs (3.4% in 2015); this rule matches it
    # everywhere it is filled. Savant's column is kept for the audit test.
    pa = pa.rename(columns={"woba_denom": "woba_denom_savant"})
    pa["woba_denom"] = (~pa["events"].isin(NOT_IN_WOBA)).astype("float64")
    pa["is_woba_pa"] = pa["woba_denom"] == 1
    pa["is_bip"] = pa["type"] == "X"
    pa["is_k"] = pa["events"].isin(STRIKEOUTS)
    pa["is_bb"] = pa["events"] == "walk"
    pa["is_hr"] = pa["events"] == "home_run"
    pa["is_hit"] = pa["events"].isin(HITS)
    # Savant also fills expected wOBA on strikeouts and walks; keep it only for batted balls.
    pa["xwoba_bip"] = pa["estimated_woba_using_speedangle"].where(pa["is_bip"])
    return pa.drop(columns="estimated_woba_using_speedangle").reset_index(drop=True)


def season_summary(pa: pd.DataFrame, stats: dict) -> dict:
    w = pa[pa["is_woba_pa"]]
    coors = w[w["is_coors"]]
    visitors_coors = coors[~coors["home_player"]]
    road_elsewhere = w[~w["is_coors"] & (w["inning_topbot"] == "Top")]
    bip = pa[pa["is_bip"]]
    return {
        **stats,
        "pas": len(pa),
        "woba_pas": len(w),
        "denom_filled_pct": 100 * pa["woba_denom_savant"].isna().mean(),
        "lg_woba": w["woba_value"].mean(),
        "coors_games": coors["game_pk"].nunique(),
        "coors_pas_visitors": len(visitors_coors),
        "coors_pas_rockies": int(coors["home_player"].sum()),
        "woba_visitors_coors": visitors_coors["woba_value"].mean(),
        "woba_visitors_elsewhere": road_elsewhere["woba_value"].mean(),
        "bip_no_ev_pct": 100 * bip["launch_speed"].isna().mean(),
        "bip_no_xwoba_pct": 100 * bip["xwoba_bip"].isna().mean(),
    }


def to_markdown(df: pd.DataFrame) -> str:
    def fmt(col: str, v: object) -> str:
        if col == "season":
            return str(v)
        if isinstance(v, int | np.integer):
            return f"{v:,}"
        return f"{v:.1f}" if col.endswith("pct") else f"{v:.3f}"

    header = "| " + " | ".join(df.columns) + " |"
    rule = "|" + "|".join("---:" for _ in df.columns) + "|"
    rows = [
        "| " + " | ".join(fmt(c, v) for c, v in zip(df.columns, row, strict=True)) + " |"
        for row in df.itertuples(index=False)
    ]
    return "\n".join([header, rule, *rows])


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )
    first, last = load_config()["seasons"]["pull"]
    pas, summaries = [], []
    for season in range(first, last + 1):
        schedule = pd.read_parquet(data_path("raw") / "schedule" / f"{season}.parquet")
        game_venues = venues(schedule)
        pitches, stats = clean_pitches(load_raw_season(season), game_venues)
        pitches.to_parquet(data_path("interim") / f"pitches_{season}.parquet", index=False)
        pa = build_pa_table(pitches)
        pas.append(pa)
        stats = {
            "season": season,
            "games_played": len(game_venues),
            "games_statcast": pitches["game_pk"].nunique(),
            "games_no_venue": int(pitches.loc[pitches["venue_id"].isna(), "game_pk"].nunique()),
            "pitches": len(pitches),
            **stats,
        }
        summaries.append(season_summary(pa, stats))
        log.info("%d: %d pitches, %d PAs", season, len(pitches), len(pa))

    pd.concat(pas, ignore_index=True).to_parquet(data_path("processed") / "pa.parquet", index=False)
    summary = pd.DataFrame(summaries)
    out = ROOT / "reports" / "tables" / "data_summary.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "# Data summary\n\nWritten by `make data` (coorsfit.clean). Regular season only. "
        "wOBA columns use PAs with `woba_denom == 1`; *visitors elsewhere* are road batters "
        "at every park except Coors.\n\n" + to_markdown(summary) + "\n"
    )
    print(to_markdown(summary))


if __name__ == "__main__":
    main()

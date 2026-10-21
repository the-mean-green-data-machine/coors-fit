"""Sanity checks on the built data (Phase 1). Skipped until `make data` has run."""

import pandas as pd
import pytest

from coorsfit.clean import NOT_IN_WOBA, PITCH_KEY
from coorsfit.config import data_path, load_config
from coorsfit.ingest import played_entries

PA_PATH = data_path("processed") / "pa.parquet"
pytestmark = pytest.mark.skipif(not PA_PATH.exists(), reason="run `make data` first")

CFG = load_config()
SEASONS = range(CFG["seasons"]["pull"][0], CFG["seasons"]["pull"][1] + 1)
SHORTENED = set(CFG["seasons"]["shortened"])
MEXICO_CITY_2024 = [746560, 746561]  # Rockies home games at Estadio Alfredo Harp Helú


@pytest.fixture(scope="module")
def pa() -> pd.DataFrame:
    return pd.read_parquet(PA_PATH)


@pytest.fixture(scope="module")
def played() -> dict[int, pd.DataFrame]:
    return {
        s: played_entries(pd.read_parquet(data_path("raw") / "schedule" / f"{s}.parquet"))
        for s in SEASONS
    }


def test_every_season_is_present(pa):
    assert sorted(pa["season"].unique()) == list(SEASONS)


def test_pa_counts_per_season_are_plausible(pa):
    # A full season is 2,430 games at roughly 38 PAs per team per game; 2020 had 60 games a team.
    for season, n in pa.groupby("season").size().items():
        low, high = (62_000, 70_000) if season in SHORTENED else (178_000, 192_000)
        assert low <= n <= high, (season, n)


def test_statcast_games_match_the_played_schedule(pa, played):
    for season in SEASONS:
        scheduled = set(played[season]["game_pk"])
        pulled = set(pa.loc[pa["season"] == season, "game_pk"])
        assert scheduled - pulled == set(), f"{season}: played games missing from Statcast"
        assert pulled - scheduled == set(), f"{season}: Statcast games not in the played schedule"


def test_no_duplicate_pas(pa):
    assert not pa.duplicated(["game_pk", "at_bat_number"]).any()


def test_no_duplicate_pitches():
    for season in SEASONS:
        keys = pd.read_parquet(
            data_path("interim") / f"pitches_{season}.parquet", columns=PITCH_KEY
        )
        assert not keys.duplicated().any(), season


def test_every_pa_has_batter_pitcher_home_team_event_and_venue(pa):
    for col in ["batter", "pitcher", "home_team", "events", "venue_id"]:
        assert pa[col].notna().all(), col


def test_coors_games_and_pas_per_season_are_plausible(pa, played):
    coors = pa[pa["is_coors"]]
    for season in SEASONS:
        games = played[season].drop_duplicates("game_pk")
        expected = int((games["venue_id"] == CFG["coors"]["venue_id"]).sum())
        in_season = coors[coors["season"] == season]
        assert in_season["game_pk"].nunique() == expected, season
        low, high = (25, 35) if season in SHORTENED else (78, 81)
        assert low <= expected <= high, (season, expected)
        # Both teams together bat 70 to 90 times in a typical Coors game.
        assert 70 <= len(in_season) / expected <= 90, season


def test_neutral_site_rockies_home_games_are_not_coors(pa):
    mexico = pa[pa["game_pk"].isin(MEXICO_CITY_2024)]
    assert mexico["game_pk"].nunique() == 2
    assert (mexico["home_team"] == "COL").all()
    assert not mexico["is_coors"].any()


def test_home_players_are_rockies_batting_at_coors(pa):
    coors = pa[pa["is_coors"]]
    assert (coors["home_player"] == (coors["inning_topbot"] == "Bot")).all()
    assert (coors.loc[coors["home_player"], "bat_team"] == "COL").all()
    assert not pa.loc[~pa["is_coors"], "home_player"].any()


def test_woba_denominator_rule_matches_savant_wherever_savant_fills_it(pa):
    filled = pa["woba_denom_savant"].notna()
    assert (pa.loc[filled, "woba_denom"] == pa.loc[filled, "woba_denom_savant"]).all()
    assert not pa.loc[pa["is_woba_pa"], "events"].isin(NOT_IN_WOBA).any()


def test_woba_values_are_present_and_in_range(pa):
    counted = pa.loc[pa["is_woba_pa"], "woba_value"]
    assert counted.notna().all()
    assert counted.between(0, 2.2).all()


def test_pas_with_blank_savant_denominator_carry_normal_woba_values(pa):
    # Those PAs are kept, so their woba_value must look like every other PA with the same event.
    counted = pa[pa["is_woba_pa"]]
    blank = counted["woba_denom_savant"].isna()
    means = counted.groupby([blank, "season", "events"])["woba_value"].agg(["mean", "size"])
    both = means.loc[True].join(means.loc[False], lsuffix="_blank", rsuffix="_filled").dropna()
    both = both[(both["size_blank"] >= 30) & (both["size_filled"] >= 30)]
    assert len(both) > 0
    assert (both["mean_blank"] - both["mean_filled"]).abs().max() < 0.01


def test_run_value_is_from_the_batters_side():
    # The H1 trait reads a positive delta_run_exp as good for the hitter.
    p = pd.read_parquet(
        data_path("interim") / f"pitches_{SEASONS[-1]}.parquet", columns=["events", "delta_run_exp"]
    )
    assert p.loc[p["events"] == "home_run", "delta_run_exp"].mean() > 0
    assert p.loc[p["events"] == "strikeout", "delta_run_exp"].mean() < 0

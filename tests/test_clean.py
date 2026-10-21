"""Unit tests for the cleaning logic on small synthetic inputs (no data needed)."""

import pandas as pd

from coorsfit.clean import PITCH_COLUMNS, build_pa_table, clean_pitches, venues


def pitch(**overrides) -> dict:
    row = dict.fromkeys(PITCH_COLUMNS)
    row.update(
        game_date="2025-06-06", game_pk=1, game_year=2025, at_bat_number=1, pitch_number=1,
        inning=1, inning_topbot="Top", home_team="COL", away_team="SF", batter=10, pitcher=20,
        balls=0, strikes=0, outs_when_up=0, game_type="R", stand="R", p_throws="R",
        pitch_type="FF", description="ball", type="B",
    )  # fmt: skip
    row.update(overrides)
    return row


def raw(*rows: dict) -> pd.DataFrame:
    return pd.DataFrame(list(rows), columns=PITCH_COLUMNS)


GAME_VENUES = pd.DataFrame({"game_pk": [1, 2], "venue_id": [19, 5325]})


def test_venues_keeps_one_row_per_played_game():
    schedule = pd.DataFrame(
        {
            "game_pk": [1, 1, 2, 3],
            "official_date": ["2025-06-01", "2025-06-02", "2025-06-02", "2025-06-03"],
            "coded_state": ["D", "F", "F", "C"],  # 1 postponed then played; 3 cancelled
            "venue_id": [19, 19, 5325, 19],
        }
    )
    assert venues(schedule).to_dict("list") == {"game_pk": [1, 2], "venue_id": [19, 5325]}


def test_clean_pitches_drops_duplicates_and_non_regular_season():
    df, stats = clean_pitches(
        raw(pitch(), pitch(), pitch(pitch_number=2), pitch(game_type="S", pitch_number=3)),
        GAME_VENUES,
    )
    assert len(df) == 2
    assert stats == {"dup_pitches_exact": 1, "dup_pitches_dropped": 1}


def test_clean_pitches_flags_coors_and_home_player():
    df, _ = clean_pitches(
        raw(
            pitch(),  # SF batting at Coors
            pitch(at_bat_number=2, inning_topbot="Bot", batter=11),  # COL batting at Coors
            pitch(
                game_pk=2, inning_topbot="Bot", home_team="COL", batter=11
            ),  # COL at a neutral site
        ),
        GAME_VENUES,
    )
    assert df["is_coors"].tolist() == [True, True, False]
    assert df["bat_team"].tolist() == ["SF", "COL", "COL"]
    assert df["home_player"].tolist() == [False, True, False]


def test_build_pa_table_takes_the_pitch_that_ends_each_pa():
    pitches, _ = clean_pitches(
        raw(
            pitch(),
            pitch(pitch_number=2, type="S", description="swinging_strike"),
            pitch(
                pitch_number=3,
                type="S",
                events="strikeout",
                woba_value=0.0,
                woba_denom=1,
                estimated_woba_using_speedangle=0.0,
            ),
            pitch(
                at_bat_number=2,
                type="X",
                events="home_run",
                bb_type="fly_ball",
                woba_value=2.0,
                woba_denom=1,
                estimated_woba_using_speedangle=1.6,
            ),
        ),  # fmt: skip
        GAME_VENUES,
    )
    pa = build_pa_table(pitches)
    assert pa["n_pitches"].tolist() == [3, 1]
    assert pa["is_k"].tolist() == [True, False]
    assert pa["is_hr"].tolist() == [False, True]
    # Expected wOBA is kept only for batted balls, not the strikeout.
    assert pa["xwoba_bip"].isna().tolist() == [True, False]


def test_woba_denominator_comes_from_the_event_when_savant_leaves_it_blank():
    pitches, _ = clean_pitches(
        raw(
            pitch(type="X", events="single", woba_value=0.9),  # Savant denominator blank
            pitch(at_bat_number=2, type="X", events="sac_bunt", woba_value=0.0, woba_denom=0),
            pitch(at_bat_number=3, type="B", events="catcher_interf", woba_value=0.7, woba_denom=1),
        ),
        GAME_VENUES,
    )
    pa = build_pa_table(pitches)
    assert pa["woba_denom"].tolist() == [1.0, 0.0, 1.0]
    assert pa["woba_denom_savant"].isna().tolist() == [True, False, False]

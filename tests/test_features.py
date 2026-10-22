"""Leakage tests: no Coors PA and no season at or after the target season may move a trait."""

import numpy as np
import pandas as pd
import pytest

from coorsfit.config import data_path
from coorsfit.features import SUMS, TRAITS, build_features, load_inputs, window_counts

RNG = np.random.default_rng(11)


def synthetic(n_batters=120, seasons=range(2015, 2019)) -> tuple[pd.DataFrame, pd.DataFrame]:
    pa_rows, pitch_rows = [], []
    for b in range(n_batters):
        for s in seasons:
            n = 300
            pa_rows.append(
                pd.DataFrame(
                    {
                        "batter": b,
                        "season": s,
                        "is_coors": RNG.random(n) < 0.05,
                        "events": RNG.choice(["single", "field_out", "strikeout"], n),
                        "bb_type": RNG.choice(["line_drive", "fly_ball", "ground_ball", None], n),
                        "launch_speed": RNG.normal(90, 8, n),
                        "is_woba_pa": True,
                    }
                )  # fmt: skip
            )
            m = 1200
            pitch_rows.append(
                pd.DataFrame(
                    {
                        "batter": b,
                        "season": s,
                        "is_coors": RNG.random(m) < 0.05,
                        "pitch_type": RNG.choice(["FF", "SL", "CH", "CU"], m),
                        "delta_run_exp": RNG.normal(0, 0.2, m),
                    }
                )  # fmt: skip
            )
    pa = pd.concat(pa_rows, ignore_index=True)
    pa["is_k"] = pa["events"] == "strikeout"
    return pa, pd.concat(pitch_rows, ignore_index=True)


def test_window_counts_never_include_the_target_season():
    counts = pd.DataFrame(0.0, index=range(3), columns=SUMS)
    counts["batter"], counts["season"], counts["pa_n"] = 1, [2015, 2016, 2017], [10, 20, 40]
    w = window_counts(counts, 2).set_index("season")["pa_n"]
    assert w[2017] == 10 + 20
    assert w[2018] == 20 + 40
    assert w[2016] == 10


def test_traits_ignore_coors_rows_and_the_target_and_later_seasons():
    pa, pitches = synthetic()
    population = pd.DataFrame({"batter": range(120), "season": 2017})
    before = build_features(pa, pitches, population)

    pa2, pitches2 = pa.copy(), pitches.copy()
    coors_or_later = pa2["is_coors"] | (pa2["season"] >= 2017)
    pa2.loc[coors_or_later, ["events", "bb_type"]] = ["strikeout", "fly_ball"]
    pa2.loc[coors_or_later, "is_k"] = True
    pa2.loc[coors_or_later, "launch_speed"] = 120.0
    pitches2.loc[pitches2["is_coors"] | (pitches2["season"] >= 2017), "delta_run_exp"] = 5.0
    after = build_features(pa2, pitches2, population)

    pd.testing.assert_frame_equal(before, after)


DATA = data_path("processed") / "features.parquet"


@pytest.mark.skipif(not DATA.exists(), reason="run `make features` first")
def test_stored_traits_are_unchanged_without_coors_rows_or_future_seasons():
    season = 2022
    stored = pd.read_parquet(DATA)
    stored = stored[stored["season"] == season].reset_index(drop=True)
    pa, pitches = load_inputs()
    keep_pa = ~pa["is_coors"] & (pa["season"] < season)
    keep_pitch = ~pitches["is_coors"] & (pitches["season"] < season)
    rebuilt = build_features(pa[keep_pa], pitches[keep_pitch], stored[["batter", "season"]])
    pd.testing.assert_frame_equal(stored[rebuilt.columns], rebuilt, check_dtype=False)
    assert stored[TRAITS].notna().all().all()

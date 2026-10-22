"""Phase 4 tests: model fitting, scoring and the multiple-comparison correction."""

import numpy as np
import pandas as pd

from coorsfit.evaluate import holm, lift, splits
from coorsfit.models import Linear, batter_folds, design, wls

RNG = np.random.default_rng(5)


def test_holm_matches_a_worked_example():
    p = np.array([0.01, 0.04, 0.03, 0.005])
    assert np.allclose(holm(p), [0.03, 0.06, 0.06, 0.02])


def test_weighted_fit_on_hitter_seasons_equals_least_squares_on_pas():
    rows = pd.DataFrame({"x": RNG.normal(size=40), "n": RNG.integers(1, 30, 40)})
    pas = rows.loc[rows.index.repeat(rows["n"])].reset_index()
    pas["r"] = 0.04 + 0.01 * pas["x"] + RNG.normal(0, 0.5, len(pas))
    rows["y"] = pas.groupby("index")["r"].mean()
    by_rows = wls(design(rows, ["x"]), rows["y"].to_numpy(), rows["n"].to_numpy(float))
    by_pas = np.linalg.lstsq(design(pas, ["x"]), pas["r"].to_numpy(), rcond=None)[0]
    assert np.allclose(by_rows, by_pas)


def test_a_model_identical_to_its_baseline_has_zero_lift():
    pred = pd.DataFrame(
        {"batter": np.arange(50) % 20, "n": 10, "y": RNG.normal(0.04, 0.1, 50), "ss_within": 2.0}
    )
    pred["pred_A"] = pred["pred_B"] = 0.04
    res = lift(pred, "A", "B", stream=0)
    assert res["lift"] == res["lift_lo"] == res["lift_hi"] == 0
    assert res["rmse_ratio"] == 1


def test_m1_with_traits_fully_penalized_reduces_to_b0q():
    df = pd.DataFrame(
        {
            "quality": RNG.normal(0, 0.02, 300),
            "t": RNG.normal(size=300),
            "n": RNG.integers(1, 30, 300),
        }
    )
    df["y"] = 0.04 - 0.3 * df["quality"] + 0.01 * df["t"] + RNG.normal(0, 0.1, 300)
    X, y, w = design(df, ["quality", "t"]), df["y"].to_numpy(), df["n"].to_numpy(float)
    m1 = Linear("M1", controls=["quality"], traits=["t"])
    b_m1 = wls(X, y, w, m1._penalty(1e12))
    b_b0q = wls(design(df, ["quality"]), y, w)
    assert abs(b_m1[2]) < 1e-6  # the trait is shrunk to zero...
    assert np.allclose(b_m1[:2], b_b0q, atol=1e-6)  # ...while intercept and quality are not


def test_no_batter_is_split_across_folds():
    batters = pd.Series(RNG.integers(0, 200, 2000))
    folds = batter_folds(batters, 5, stream=1)
    assert (pd.Series(folds).groupby(batters).nunique() == 1).all()


def test_rolling_origin_trains_only_on_earlier_seasons():
    df = pd.DataFrame({"season": np.repeat(range(2016, 2027), 10), "batter": np.arange(110)})
    for train, test in splits(df, "rolling_origin"):
        assert df.loc[train, "season"].max() < df.loc[test, "season"].min()
        assert df.loc[test, "season"].nunique() == 1

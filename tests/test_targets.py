"""Phase 2 tests: shrinkage helpers, the expectation model, and leakage rules on synthetic data."""

import numpy as np
import pandas as pd

from coorsfit.expectation import fit_expectation, flag_pitchers_batting
from coorsfit.shrinkage import design_effect, shrink, tau2_anova
from coorsfit.targets import eligibility_counts

RNG = np.random.default_rng(7)


def test_tau2_anova_recovers_the_between_group_variance():
    n = RNG.integers(5, 200, size=3000)
    truth = RNG.normal(0, 0.05, size=n.size)
    sums = np.array([RNG.normal(t, 0.5, size=k).sum() for t, k in zip(truth, n, strict=True)])
    assert abs(tau2_anova(sums, n, 0.25) ** 0.5 - 0.05) < 0.005


def test_tau2_anova_is_near_zero_for_pure_noise():
    n = np.full(2000, 10)
    sums = RNG.normal(0, 0.5 * np.sqrt(10), size=n.size)
    # Group means vary by 0.025 from noise alone; the estimate's SE is about 0.0008.
    assert tau2_anova(sums, n, 0.25) < 0.0025


def test_shrink_returns_zeros_when_groups_do_not_differ():
    assert not shrink(np.array([3.0, -2.0]), np.array([10, 10]), 0.25, 0.0).any()


def test_design_effect_is_one_without_clustering_and_larger_with_game_shocks():
    groups = np.repeat(np.arange(200), 100)
    games = np.tile(np.repeat(np.arange(4), 25), 200)
    noise = RNG.normal(0, 0.5, size=groups.size)
    assert design_effect(noise, groups, games, 0.25) < 1.1
    shocks = RNG.normal(0, 0.15, size=800)[groups * 4 + games]
    assert design_effect(noise + shocks, groups, games, 0.25) > 1.5


def synthetic_pa(n=60_000, seed=0) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    batter_effect = pd.Series(rng.normal(0, 0.04, 300), index=np.arange(1000, 1300))
    pitcher_effect = pd.Series(rng.normal(0, 0.03, 200), index=np.arange(5000, 5200))
    df = pd.DataFrame(
        {
            "season": 2025,
            "game_pk": rng.integers(0, 2000, n),
            "batter": rng.choice(batter_effect.index, n),
            "pitcher": rng.choice(pitcher_effect.index, n),
            "venue_id": rng.integers(1, 30, n),
            "stand": rng.choice(["L", "R"], n),
            "p_throws": rng.choice(["L", "R"], n),
            "inning_topbot": rng.choice(["Top", "Bot"], n),
            "is_woba_pa": True,
            "is_coors": rng.random(n) < 0.03,
            "pitcher_batting": False,
            "n_pitches": 4,
        }
    )
    df["woba_value"] = (
        0.32
        + df["batter"].map(batter_effect)
        + df["pitcher"].map(pitcher_effect)
        + rng.normal(0, 0.5, n)
    )
    return df, batter_effect, pitcher_effect


def test_expectation_recovers_batter_and_pitcher_effects_without_overfitting():
    df, batter_effect, pitcher_effect = synthetic_pa()
    model = fit_expectation(df)
    for est, truth in [
        (model.effects["batter"], batter_effect),
        (model.effects["pitcher"], pitcher_effect),
    ]:
        est = est.reindex(truth.index)
        # Regressing the truth on a well-shrunk estimate gives a slope near 1.
        assert 0.8 < np.polyfit(est, truth, 1)[0] < 1.25
        assert np.corrcoef(est, truth)[0, 1] > 0.6


def test_expectation_never_uses_coors_outcomes():
    df, _, _ = synthetic_pa()
    before = fit_expectation(df).predict(df)
    df.loc[df["is_coors"], "woba_value"] += 5.0
    assert np.allclose(fit_expectation(df).predict(df), before)


def test_expectation_never_uses_pitchers_batting():
    df, _, _ = synthetic_pa()
    df["pitcher_batting"] = df.index % 50 == 0
    before = fit_expectation(df).predict(df)
    df.loc[df["pitcher_batting"], "woba_value"] -= 5.0
    assert np.allclose(fit_expectation(df).predict(df), before)


def test_flag_pitchers_batting_spares_two_way_players():
    pa = pd.DataFrame(
        {
            "season": 2019,
            # Player 1 pitches a lot and bats rarely; player 2 pitches a lot and bats every day.
            "batter": [1] * 20 + [2] * 150 + [3] * 50,
            "pitcher": [1] * 60 + [2] * 60 + [9] * 100,
            "n_pitches": 4,
        }
    )
    flagged = flag_pitchers_batting(pa)
    assert flagged[pa["batter"] == 1].all()
    assert not flagged[pa["batter"] != 1].any()


def test_eligibility_counts_use_only_prior_non_coors_seasons():
    rows = []
    for season, n, coors in [(2015, 50, False), (2016, 70, False), (2017, 90, False),
                             (2018, 110, False), (2017, 500, True)]:  # fmt: skip
        rows += [{"batter": 1, "season": season, "is_woba_pa": True, "is_coors": coors}] * n
    counts = eligibility_counts(pd.DataFrame(rows)).set_index("season")
    # 2018's prior window is 2016 and 2017 non-Coors PAs only; the Coors PAs never count.
    assert counts.loc[2018, "n_noncoors_prior"] == 70 + 90
    assert counts.loc[2018, "n_noncoors_same"] == 110
    assert counts.loc[2016, "n_noncoors_prior"] == 50
    assert counts.loc[2017, "n_noncoors_same"] == 90

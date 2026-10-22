"""Phase 3 tests: the gate's tau estimator on synthetic hitters with known spread."""

import numpy as np
import pandas as pd

from coorsfit.gate import gate_estimate, hitter_table, split_half, tau2_from_hitters


def synthetic_hitters(tau: float, post_var: float = 0.0004, n_hitters: int = 800, seed: int = 0):
    """Coors PAs with a true hitter effect, per-season expectation error, and uneven noise."""
    rng = np.random.default_rng(seed)
    rows = []
    for batter in range(n_hitters):
        theta = rng.normal(0, tau)
        sd = rng.uniform(0.45, 0.6)  # outcome volatility differs by hitter
        for season in range(2016, 2016 + rng.integers(1, 5)):
            err = rng.normal(0, np.sqrt(post_var))  # shared by all his PAs that season
            n = rng.integers(8, 16)
            rows.append(
                pd.DataFrame(
                    {
                        "batter": batter,
                        "season": season,
                        "game_pk": season * 100_000 + batter * 10 + np.arange(n) // 4,
                        "dev": theta + err + rng.normal(0, sd, n),
                        "batter_post_var": post_var,
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)


def test_gate_recovers_a_real_spread_between_hitters():
    pas = synthetic_hitters(tau=0.04)
    est = gate_estimate(pas)
    assert abs(est["tau"] - 0.04) < 0.012
    assert est["stable_skill"]


def test_gate_does_not_pass_on_noise_alone():
    est = gate_estimate(synthetic_hitters(tau=0.0, seed=1))
    assert est["tau2_lo"] < 0 < est["tau2_hi"]
    assert not est["stable_skill"]


def test_expectation_error_must_be_removed_or_it_looks_like_skill():
    pas = synthetic_hitters(tau=0.0, post_var=0.0016, seed=2)
    corrected = tau2_from_hitters(hitter_table(pas), deff=1.0)
    ignored = tau2_from_hitters(hitter_table(pas.assign(batter_post_var=0.0)), deff=1.0)
    assert ignored > corrected
    assert abs(corrected) < ignored


def test_split_half_is_near_zero_without_a_stable_effect():
    rng = np.random.default_rng(3)
    n_hitters, seasons, per_season = 400, range(2016, 2020), 20  # 40 PAs in each half
    pas = pd.DataFrame(
        {
            "batter": np.repeat(np.arange(n_hitters), len(seasons) * per_season),
            "season": np.tile(np.repeat(list(seasons), per_season), n_hitters),
            "dev": rng.normal(0, 0.5, n_hitters * len(seasons) * per_season),
        }
    )
    res = split_half(pas)
    assert res["hitters"] == n_hitters
    assert res["r_lo"] < 0 < res["r_hi"]

"""Phase 2 checks on the built targets. Skipped until `make targets` has run."""

import numpy as np
import pandas as pd
import pytest

from coorsfit.config import data_path, load_config

TARGETS = data_path("processed") / "targets.parquet"
pytestmark = pytest.mark.skipif(not TARGETS.exists(), reason="run `make targets` first")
CFG = load_config()


@pytest.fixture(scope="module")
def targets() -> pd.DataFrame:
    return pd.read_parquet(TARGETS)


@pytest.fixture(scope="module")
def coors_pa() -> pd.DataFrame:
    return pd.read_parquet(data_path("processed") / "coors_pa.parquet")


@pytest.fixture(scope="module")
def fits() -> pd.DataFrame:
    return pd.read_parquet(data_path("processed") / "expectation_fits.parquet")


def test_every_coors_woba_pa_lands_in_exactly_one_hitter_season(targets, coors_pa):
    pa = pd.read_parquet(data_path("processed") / "pa.parquet", columns=["is_coors", "is_woba_pa"])
    assert not targets.duplicated(["season", "batter", "group"]).any()
    assert targets["n_coors"].sum() == len(coors_pa) == (pa["is_coors"] & pa["is_woba_pa"]).sum()


def test_expected_values_are_finite_and_plausible(coors_pa):
    assert np.isfinite(coors_pa["expected"]).all()
    assert coors_pa["expected"].between(0.1, 0.6).all()


def test_eligibility_follows_the_configured_thresholds(targets):
    first, last = CFG["seasons"]["targets"]
    expected = (
        targets["season"].between(first, last)
        & (targets["n_noncoors_same"] >= CFG["target"]["min_same_season_noncoors_pa"])
        & (targets["n_noncoors_prior"] >= CFG["traits"]["min_prior_noncoors_pa"])
    )
    assert (targets["eligible"] == expected).all()


def test_visitors_gain_at_coors_in_every_target_season(targets):
    v = targets[targets["eligible"] & (targets["group"] == "visitor")]
    effect = v.groupby("season")["sum_resid"].sum() / v.groupby("season")["n_coors"].sum()
    assert len(effect) == CFG["seasons"]["targets"][1] - CFG["seasons"]["targets"][0] + 1
    assert effect.between(0, 0.15).all(), effect


def test_expectation_is_centered_and_roughly_calibrated_out_of_sample(fits):
    # Fit on odd-numbered games, scored on even-numbered non-Coors PAs (DECISIONS.md, D-19).
    assert (fits["mean_resid"].abs() < 0.01).all()
    assert 0.85 <= fits["slope"].mean() <= 1.10


def test_shrinkage_only_pulls_toward_the_season_effect(targets):
    e = targets[targets["eligible"]]
    assert e["dev_shrunk"].notna().all()
    assert targets.loc[~targets["eligible"], "dev_shrunk"].isna().all()
    assert (e["dev_shrunk"].abs() <= e["dev"].abs() + 1e-12).all()

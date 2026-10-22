"""Phase 2: Coors residuals for every Coors PA and every hitter-season, with shrinkage.

Outputs:
- data/processed/coors_pa.parquet: each Coors wOBA PA with its expected value and residual
- data/processed/targets.parquet: one row per season, batter and group (visitor or Rockies)
- data/processed/expectation_fits.parquet: per-season fit and held-out calibration numbers

The league-average Coors effect is only described here (by season and pooled). Phase 4 refits
it inside each training fold before it is used for any prediction.

Run with `python -m coorsfit.targets`.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from coorsfit.config import data_path, load_config
from coorsfit.expectation import fit_expectation, flag_pitchers_batting, heldout_calibration
from coorsfit.shrinkage import shrink, tau2_anova

log = logging.getLogger(__name__)

TARGET_KEY = ["season", "batter", "group"]


def coors_residuals(pa: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Expected wOBA and residual for every Coors wOBA PA, using a model fit per season."""
    coors, fits = [], []
    for season, season_pa in pa.groupby("season"):
        model = fit_expectation(season_pa)
        at_coors = season_pa[season_pa["is_coors"] & season_pa["is_woba_pa"]].copy()
        at_coors["expected"] = model.predict(at_coors)
        coors.append(at_coors)
        fits.append(
            {
                "season": season,
                "fit_pas": model.n_fit,
                "pitcher_batting_pas": int(season_pa["pitcher_batting"].sum()),
                "mu": model.mu,
                "sigma": model.sigma2**0.5,
                **{f"tau_{n}": t**0.5 for n, t in model.tau2.items()},
                **{f"deff_{n}": d for n, d in model.design_effect.items()},
                "home_advantage": model.home_advantage,
                "iterations": model.iterations,
                **heldout_calibration(season_pa),
            }
        )
    out = pd.concat(coors, ignore_index=True)
    out["group"] = np.where(out["home_player"], "rockies", "visitor")
    out["resid"] = out["woba_value"] - out["expected"]
    keep = ["game_pk", "at_bat_number", "season", "batter", "pitcher", "group",
            "woba_value", "expected", "resid"]  # fmt: skip
    return out[keep], pd.DataFrame(fits)


def eligibility_counts(pa: pd.DataFrame) -> pd.DataFrame:
    """Non-Coors wOBA PAs per batter: in the same season, and in the prior trait window."""
    window = load_config()["traits"]["window_seasons"]
    counts = (
        pa[pa["is_woba_pa"] & ~pa["is_coors"]]
        .groupby(["batter", "season"])
        .size()
        .unstack("season", fill_value=0)
    )
    seasons = sorted(pa["season"].unique())
    counts = counts.reindex(columns=seasons, fill_value=0)
    prior = sum(counts.shift(lag, axis=1, fill_value=0) for lag in range(1, window + 1))
    return pd.DataFrame(
        {"n_noncoors_same": counts.stack(), "n_noncoors_prior": prior.stack()}
    ).reset_index()


def build_targets(coors_pa: pd.DataFrame, counts: pd.DataFrame) -> pd.DataFrame:
    cfg = load_config()
    t = (
        coors_pa.groupby(TARGET_KEY)
        .agg(
            n_coors=("resid", "size"),
            sum_resid=("resid", "sum"),
            mean_actual=("woba_value", "mean"),
            mean_expected=("expected", "mean"),
        )
        .reset_index()
    )
    t["mean_resid"] = t["sum_resid"] / t["n_coors"]
    t = t.merge(counts, on=["batter", "season"], how="left", validate="many_to_one")
    first, last = cfg["seasons"]["targets"]
    t["eligible"] = (
        t["season"].between(first, last)
        & (t["n_noncoors_same"] >= cfg["target"]["min_same_season_noncoors_pa"])
        & (t["n_noncoors_prior"] >= cfg["traits"]["min_prior_noncoors_pa"])
    )
    return t


def shrink_targets(t: pd.DataFrame, coors_pa: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Shrink each eligible hitter-season's deviation from its season's Coors effect.

    Deviations are taken from the group's Coors effect in the same season, so season-to-season
    swings (such as the Rockies staff getting better or worse) are not counted as hitter skill.
    """
    t = t.copy()
    t[["season_effect", "dev", "shrink_factor", "dev_shrunk"]] = np.nan
    diagnostics = []
    for group in ("visitor", "rockies"):
        rows = t["eligible"] & (t["group"] == group)
        e = t[rows]
        season_effect = (
            e.groupby("season")["sum_resid"].sum() / e.groupby("season")["n_coors"].sum()
        )
        effect = e["season"].map(season_effect)
        dev_sum = e["sum_resid"] - e["n_coors"] * effect
        pas = coors_pa.merge(e[TARGET_KEY], on=TARGET_KEY)
        sigma2 = float((pas["resid"] - pas["season"].map(season_effect)).var())
        tau2 = tau2_anova(dev_sum.to_numpy(), e["n_coors"].to_numpy(), sigma2)
        k = sigma2 / tau2 if tau2 > 0 else np.inf
        t.loc[rows, "season_effect"] = effect
        t.loc[rows, "dev"] = dev_sum / e["n_coors"]
        t.loc[rows, "shrink_factor"] = e["n_coors"] / (e["n_coors"] + k)
        t.loc[rows, "dev_shrunk"] = shrink(
            dev_sum.to_numpy(), e["n_coors"].to_numpy(), sigma2, tau2
        )
        diagnostics.append(
            {
                "group": group,
                "hitter_seasons": int(rows.sum()),
                "coors_pas": int(e["n_coors"].sum()),
                "pooled_effect": float(e["sum_resid"].sum() / e["n_coors"].sum()),
                "sigma": sigma2**0.5,
                "tau_hitter_season": tau2**0.5,
                "median_coors_pas": float(e["n_coors"].median()),
                "median_shrink_factor": float(t.loc[rows, "shrink_factor"].median()),
                "sd_dev_raw": float(
                    np.sqrt(np.average(t.loc[rows, "dev"] ** 2, weights=e["n_coors"]))
                ),
                "sd_dev_shrunk": float(t.loc[rows, "dev_shrunk"].std()),
            }
        )
    return t, diagnostics


def power(n_pas: int, sigma: float = 0.514) -> dict:
    """Minimum detectable trait effect (wOBA per trait SD) from a PA count alone.

    `sigma` is the pre-registered planning value of the per-PA wOBA SD, so no outcome enters.
    """
    se = sigma / np.sqrt(n_pas)
    return {
        "eligible_pas": n_pas,
        "se": se,
        "mde_80_alpha05": 2.80 * se,
        "mde_80_alpha01": 3.42 * se,
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )
    pa = pd.read_parquet(data_path("processed") / "pa.parquet")
    pa["pitcher_batting"] = flag_pitchers_batting(pa)
    coors_pa, fits = coors_residuals(pa)
    targets, diagnostics = shrink_targets(build_targets(coors_pa, eligibility_counts(pa)), coors_pa)

    out = data_path("processed")
    coors_pa.to_parquet(out / "coors_pa.parquet", index=False)
    targets.to_parquet(out / "targets.parquet", index=False)
    fits.to_parquet(out / "expectation_fits.parquet", index=False)
    pd.DataFrame(diagnostics).to_parquet(out / "shrinkage_diagnostics.parquet", index=False)

    visitors = targets[targets["eligible"] & (targets["group"] == "visitor")]
    log.info("power from eligible visitor PAs: %s", power(int(visitors["n_coors"].sum())))
    for d in diagnostics:
        log.info("%s", d)


if __name__ == "__main__":
    main()

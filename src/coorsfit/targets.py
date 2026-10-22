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
from coorsfit.shrinkage import design_effect, shrink, tau2_anova

log = logging.getLogger(__name__)

TARGET_KEY = ["season", "batter", "group"]


def coors_residuals(pa: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Expected wOBA and residual for every Coors wOBA PA, using a model fit per season."""
    coors, fits = [], []
    for season, season_pa in pa.groupby("season"):
        model = fit_expectation(season_pa)
        at_coors = season_pa[season_pa["is_coors"] & season_pa["is_woba_pa"]].copy()
        at_coors["expected"] = model.predict(at_coors)
        # The batter-effect estimate's own error is shared by all his Coors PAs that season,
        # so it would look like Coors skill; its variance is removed when estimating tau.
        at_coors["batter_post_var"] = model.posterior_var("batter", at_coors["batter"])
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
            "woba_value", "expected", "batter_post_var", "resid"]  # fmt: skip
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
            batter_post_var=("batter_post_var", "mean"),
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


def within_noise(dev: np.ndarray, units: np.ndarray, games: np.ndarray) -> tuple[float, float]:
    """Per-PA variance within units (hitter-seasons or hitters) and its game design effect.

    A hitter's PAs in one Coors game share that day's weather and pitchers, so the noise in his
    mean is the within-unit variance times the design effect.
    """
    codes = pd.factorize(units)[0]
    n = np.bincount(codes)
    within = dev - (np.bincount(codes, weights=dev) / n)[codes]
    sigma2 = float(np.sum(within**2) / (len(dev) - len(n)))
    return sigma2, design_effect(dev, codes, pd.factorize(games)[0], sigma2)


def shrink_targets(t: pd.DataFrame, coors_pa: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Shrink each eligible hitter-season's deviation from its season's Coors effect.

    Deviations are taken from the group's Coors effect in the same season, so season-to-season
    swings (such as the Rockies staff getting better or worse) are not counted as hitter skill.
    Also adds `quality` (amendment A-3): the hitter-season's mean expected wOBA at Coors,
    centered on the PA-weighted mean of its group and season.
    """
    t = t.copy()
    cols = ["season_effect", "dev", "quality", "shrink_factor", "dev_shrunk"]
    t[cols] = np.nan
    diagnostics = []
    for group in ("visitor", "rockies"):
        rows = t["eligible"] & (t["group"] == group)
        e = t[rows]
        n = e["n_coors"]
        season_effect = e.groupby("season")["sum_resid"].sum() / n.groupby(e["season"]).sum()
        season_expected = (e["mean_expected"] * n).groupby(e["season"]).sum() / n.groupby(
            e["season"]
        ).sum()
        effect = e["season"].map(season_effect)
        dev_sum = e["sum_resid"] - n * effect
        pas = coors_pa.merge(e[TARGET_KEY], on=TARGET_KEY)
        sigma2, deff = within_noise(
            (pas["resid"] - pas["season"].map(season_effect)).to_numpy(),
            (pas["batter"] * 10_000 + pas["season"]).to_numpy(),
            pas["game_pk"].to_numpy(),
        )
        noise = sigma2 * deff
        extra = e["batter_post_var"].to_numpy()
        tau2 = tau2_anova(dev_sum.to_numpy(), n.to_numpy(), noise, extra_var=extra)
        t.loc[rows, "season_effect"] = effect
        t.loc[rows, "dev"] = dev_sum / n
        t.loc[rows, "quality"] = e["mean_expected"] - e["season"].map(season_expected)
        t.loc[rows, "shrink_factor"] = tau2 / (tau2 + extra + noise / n) if tau2 > 0 else 0.0
        t.loc[rows, "dev_shrunk"] = shrink(dev_sum.to_numpy(), n.to_numpy(), noise, tau2, extra)
        diagnostics.append(
            {
                "group": group,
                "hitter_seasons": int(rows.sum()),
                "coors_pas": int(n.sum()),
                "pooled_effect": float(e["sum_resid"].sum() / n.sum()),
                "sigma": sigma2**0.5,
                "design_effect": deff,
                "sd_expectation_error": float(np.sqrt(np.average(extra, weights=n))),
                "tau2_unclipped": tau2_anova(
                    dev_sum.to_numpy(), n.to_numpy(), noise, extra_var=extra, clip=False
                ),
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

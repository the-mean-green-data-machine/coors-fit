"""Expected wOBA for a plate appearance, learned from one season's non-Coors PAs.

The model is additive: league mean + batter + pitcher + park + platoon cell + home/road.
Batter, pitcher and park effects are empirical-Bayes shrunk toward zero (league average);
platoon and home/road are a handful of cells estimated directly. Coors PAs and pitchers'
own PAs never enter the fit. A Coors PA is predicted with a park effect of zero, so its
residual measures Coors against the average non-Coors park.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from coorsfit.config import load_config
from coorsfit.shrinkage import cluster_slope, design_effect, shrink, tau2_anova

SHRUNK_TERMS = ("batter", "pitcher", "park")
CELL_TERMS = ("platoon", "home")
TERMS = SHRUNK_TERMS + CELL_TERMS


def term_keys(pa: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "batter": pa["batter"],
        "pitcher": pa["pitcher"],
        "park": pa["venue_id"],
        "platoon": pa["stand"] + pa["p_throws"],
        "home": pa["inning_topbot"] == "Bot",  # the batting team is the home team
    }


def flag_pitchers_batting(pa: pd.DataFrame) -> pd.Series:
    """PAs by a player who pitched a lot that season and batted little (pre-2022 pitchers)."""
    cfg = load_config()["expectation"]
    thrown = pa.groupby(["season", "pitcher"])["n_pitches"].sum()
    batted = pa.groupby(["season", "batter"]).size()
    thrown.index.names = batted.index.names = ["season", "player"]
    pitchers = thrown[thrown >= cfg["pitcher_batting_min_pitches"]].index
    regulars = batted[batted >= cfg["pitcher_batting_max_pa"]].index
    flagged = pitchers.difference(regulars)
    player_seasons = pd.MultiIndex.from_arrays([pa["season"], pa["batter"]])
    return pd.Series(player_seasons.isin(flagged), index=pa.index)


@dataclass
class Expectation:
    mu: float
    effects: dict[str, pd.Series]
    sigma2: float
    tau2: dict[str, float] = field(default_factory=dict)
    design_effect: dict[str, float] = field(default_factory=dict)
    n_fit: int = 0
    iterations: int = 0

    def predict(self, pa: pd.DataFrame) -> np.ndarray:
        out = np.full(len(pa), self.mu)
        for name, key in term_keys(pa).items():
            # A batter, pitcher or park never seen in the fit gets the league average (zero).
            out += key.map(self.effects[name]).fillna(0.0).to_numpy(dtype=float)
        return out

    @property
    def home_advantage(self) -> float:
        return float(self.effects["home"].get(True, 0.0) - self.effects["home"].get(False, 0.0))


def fit_expectation(pa: pd.DataFrame) -> Expectation:
    """Fit on one season's PAs, dropping Coors PAs, non-wOBA PAs and pitchers batting."""
    cfg = load_config()["expectation"]
    rows = pa[pa["is_woba_pa"] & ~pa["is_coors"] & ~pa["pitcher_batting"]]
    y = rows["woba_value"].to_numpy(dtype=float)
    codes, levels, counts = {}, {}, {}
    for name, key in term_keys(rows).items():
        codes[name], levels[name] = pd.factorize(key)
        counts[name] = np.bincount(codes[name], minlength=len(levels[name]))
    games = pd.factorize(rows["game_pk"])[0]
    eff = {name: np.zeros(len(levels[name])) for name in TERMS}
    mu, sigma2, tau2, deff = y.mean(), y.var(), {}, {}

    # Backfitting: update each term against the others until nothing moves.
    iterations = 0
    while iterations < cfg["max_iter"]:
        iterations += 1
        largest_change = 0.0
        for name in TERMS:
            others = sum(eff[o][codes[o]] for o in TERMS if o != name)
            partial = y - mu - others
            sums = np.bincount(codes[name], weights=partial, minlength=len(levels[name]))
            if name in SHRUNK_TERMS:
                # PAs in one game share shocks (a pitcher's bad day, the weather), so a
                # group's mean is noisier than independent PAs would make it.
                deff[name] = design_effect(partial, codes[name], games, sigma2)
                noise = sigma2 * deff[name]
                tau2[name] = tau2_anova(sums, counts[name], noise)
                new = shrink(sums, counts[name], noise, tau2[name])
            else:
                new = sums / counts[name]
                new -= np.average(new, weights=counts[name])  # the level lives in mu
            largest_change = max(largest_change, float(np.abs(new - eff[name]).max()))
            eff[name] = new
        fitted = sum(eff[n][codes[n]] for n in TERMS)
        mu = float((y - fitted).mean())
        sigma2 = float((y - mu - fitted).var())
        if largest_change < cfg["tol"]:
            break

    effects = {name: pd.Series(eff[name], index=levels[name]) for name in TERMS}
    return Expectation(mu, effects, sigma2, tau2, deff, n_fit=len(y), iterations=iterations)


def heldout_calibration(season_pa: pd.DataFrame) -> dict:
    """Fit on odd-numbered games, predict even-numbered non-Coors PAs, and check calibration.

    `slope` regresses the held-out wOBA on the prediction (1 means well calibrated). The
    per-term coefficients come from regressing it on each term's contribution together.
    """
    odd = season_pa["game_pk"] % 2 == 1
    test = season_pa[
        ~odd & season_pa["is_woba_pa"] & ~season_pa["is_coors"] & ~season_pa["pitcher_batting"]
    ]
    model = fit_expectation(season_pa[odd])
    pred = model.predict(test)
    y = test["woba_value"].to_numpy(dtype=float)
    slope, se = cluster_slope(pred - pred.mean(), y, test["batter"].to_numpy())
    parts = [
        key.map(model.effects[n]).fillna(0.0).to_numpy(float) for n, key in term_keys(test).items()
    ]
    X = np.column_stack([np.ones(len(test)), *parts])
    coef = np.linalg.lstsq(X, y, rcond=None)[0][1:]
    return {
        "heldout_pas": len(test),
        "mean_resid": float(np.mean(y - pred)),
        "slope": slope,
        "slope_se": se,
        **{f"coef_{n}": float(c) for n, c in zip(TERMS, coef, strict=True)},
    }

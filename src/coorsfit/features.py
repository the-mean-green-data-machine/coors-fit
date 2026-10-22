"""Phase 4 traits: computed only from non-Coors data in the seasons before the target season.

For a target season s, each trait pools the batter's non-Coors PAs (or pitches) from seasons
s-1 .. s-W (W = traits.window_seasons), is shrunk toward the league mean by empirical Bayes,
then standardized within s among eligible hitters (at least 200 prior and 100 same-season
non-Coors PAs). Neither step looks at a Coors PA or at season s itself, apart from the
same-season PA count that defines the population.

Traits (docs/hypotheses.md): brk_deficit (H1), ld_rate (H2), fb_rate (H3a), fb_ev (H3b),
k_rate (H4), sprint_speed (H5, exploratory).

Writes data/processed/features.parquet. Run with `python -m coorsfit.features`.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from coorsfit.config import data_path, load_config

log = logging.getLogger(__name__)

TRAITS = ["brk_deficit", "ld_rate", "fb_rate", "fb_ev", "k_rate", "sprint_speed"]
CONFIRMATORY = ["brk_deficit", "ld_rate", "fb_rate", "fb_ev", "k_rate"]
SUMS = ["pa_n", "k_n", "bip_n", "ld_n", "fb_n", "fb_ev_n", "fb_ev_sum", "fb_ev_sumsq",
        "fast_n", "fast_sum", "fast_sumsq", "brk_n", "brk_sum", "brk_sumsq"]  # fmt: skip


def season_counts(pa: pd.DataFrame, pitches: pd.DataFrame) -> pd.DataFrame:
    """Per batter and season, sums over non-Coors PAs and pitches only."""
    groups = load_config()["traits"]["pitch_groups"]
    pa = pa[~pa["is_coors"] & (pa["events"] != "truncated_pa")]
    bbt = pa["bb_type"]
    fb_ev = pa["launch_speed"].where(bbt == "fly_ball")
    by_pa = (
        pd.DataFrame(
            {
                "batter": pa["batter"],
                "season": pa["season"],
                "pa_n": 1,
                "k_n": pa["is_k"].astype(int),
                "bip_n": bbt.notna().astype(int),
                "ld_n": (bbt == "line_drive").astype(int),
                "fb_n": (bbt == "fly_ball").astype(int),
                "fb_ev_n": fb_ev.notna().astype(int),
                "fb_ev_sum": fb_ev.fillna(0.0),
                "fb_ev_sumsq": (fb_ev**2).fillna(0.0),
            }
        )
        .groupby(["batter", "season"])
        .sum()
    )

    p = pitches[~pitches["is_coors"] & pitches["delta_run_exp"].notna()]
    rv = p["delta_run_exp"]  # positive is good for the batter (checked in tests/test_data.py)
    parts = {}
    for name, codes in [("fast", groups["fastball"]), ("brk", groups["breaking"])]:
        x = rv.where(p["pitch_type"].isin(codes))
        parts[f"{name}_n"] = x.notna().astype(int)
        parts[f"{name}_sum"] = x.fillna(0.0)
        parts[f"{name}_sumsq"] = (x**2).fillna(0.0)
    by_pitch = (
        pd.DataFrame({"batter": p["batter"], "season": p["season"], **parts})
        .groupby(["batter", "season"])
        .sum()
    )
    return by_pa.join(by_pitch, how="outer").fillna(0.0).reset_index()


def window_counts(counts: pd.DataFrame, window: int) -> pd.DataFrame:
    """Sums over the `window` seasons before each target season (never the season itself)."""
    shifted = [counts.assign(season=counts["season"] + lag) for lag in range(1, window + 1)]
    return pd.concat(shifted).groupby(["batter", "season"])[SUMS].sum().reset_index()


def raw_traits(w: pd.DataFrame) -> pd.DataFrame:
    """Trait estimates and their sampling variances from one target season's window sums.

    League rates and pooled variances come from these rows only, so another target season's
    window (which holds later seasons' data) can never touch this one.
    """
    assert w["season"].nunique() == 1, "call once per target season"
    out = w[["batter", "season"]].copy()

    def rate(num: str, den: str, name: str) -> None:
        n = w[den].to_numpy(dtype=float)
        p0 = w[num].sum() / w[den].sum()
        with np.errstate(invalid="ignore", divide="ignore"):
            out[name] = w[num] / n
            out[f"{name}_var"] = p0 * (1 - p0) / n

    def mean(prefix: str) -> tuple[pd.Series, pd.Series]:
        n, s, ss = (w[f"{prefix}_{k}"].to_numpy(dtype=float) for k in ("n", "sum", "sumsq"))
        within = np.sum(ss - np.divide(s**2, n, where=n > 0, out=np.zeros_like(s)))
        pooled = within / max(np.sum(np.maximum(n - 1, 0)), 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return pd.Series(s / n, index=w.index), pd.Series(pooled / n, index=w.index)

    rate("k_n", "pa_n", "k_rate")
    rate("ld_n", "bip_n", "ld_rate")
    rate("fb_n", "bip_n", "fb_rate")
    out["fb_ev"], out["fb_ev_var"] = mean("fb_ev")
    fast, fast_var = mean("fast")
    brk, brk_var = mean("brk")
    # Run value per 100 pitches against fastballs minus against breaking balls (H1).
    out["brk_deficit"] = 100 * (fast - brk)
    out["brk_deficit_var"] = 100**2 * (fast_var + brk_var)
    return out


def eb_shrink(est: pd.Series, var: pd.Series) -> pd.Series:
    """Shrink estimates toward their mean by the share of spread that is not sampling noise."""
    ok = est.notna() & np.isfinite(var)
    m = est[ok].mean()
    tau2 = max(0.0, est[ok].var() - var[ok].mean())
    factor = (tau2 / (tau2 + var)).where(ok, 0.0)
    return m + (est.fillna(m) - m) * factor


def sprint_trait(seasons: range, window: int) -> pd.DataFrame:
    """Most recent qualifying sprint speed from the `window` seasons before each target season."""
    min_runs = load_config()["traits"]["sprint_min_runs"]
    rows = []
    for s in range(seasons.start - window, seasons.stop):
        path = data_path("raw") / "sprint_speed" / f"{s}.parquet"
        if path.exists():
            d = pd.read_parquet(path)
            d = d[d["competitive_runs"] >= min_runs]
            rows.append(pd.DataFrame({"batter": d["player_id"], "measured": s,
                                      "sprint_speed": d["sprint_speed"]}))  # fmt: skip
    speeds = pd.concat(rows)
    out = []
    for target in seasons:
        prior = speeds[speeds["measured"].between(target - window, target - 1)]
        latest = prior.sort_values("measured").drop_duplicates("batter", keep="last")
        out.append(latest.assign(season=target)[["batter", "season", "sprint_speed"]])
    return pd.concat(out, ignore_index=True)


def build_features(
    pa: pd.DataFrame, pitches: pd.DataFrame, population: pd.DataFrame
) -> pd.DataFrame:
    """Shrunk, standardized traits for every (batter, season) in `population`."""
    window = load_config()["traits"]["window_seasons"]
    counts = season_counts(pa, pitches)
    windows = window_counts(counts, window)
    raw = pd.concat(raw_traits(g) for _, g in windows.groupby("season"))
    seasons = range(int(population["season"].min()), int(population["season"].max()) + 1)
    raw = raw.merge(sprint_trait(seasons, window), on=["batter", "season"], how="left")
    feats = population[["batter", "season"]].merge(raw, on=["batter", "season"], how="left")

    out = feats[["batter", "season"]].copy()
    out["sprint_missing"] = feats["sprint_speed"].isna()
    for season, idx in feats.groupby("season").groups.items():
        f = feats.loc[idx]
        for trait in TRAITS:
            if trait == "sprint_speed":  # a measurement, not a noisy rate: no shrinkage
                shrunk = f[trait].fillna(f[trait].mean())
            else:
                shrunk = eb_shrink(f[trait], f[f"{trait}_var"])
            out.loc[idx, f"{trait}_raw"] = f[trait]
            out.loc[idx, trait] = (shrunk - shrunk.mean()) / shrunk.std()
        log.debug("season %s: %d hitters", season, len(idx))
    return out


def trait_population(pa: pd.DataFrame) -> pd.DataFrame:
    """Eligible hitters per target season: enough prior and same-season non-Coors PAs."""
    from coorsfit.targets import eligibility_counts  # avoid a cycle at import time

    cfg = load_config()
    counts = eligibility_counts(pa)
    first, last = cfg["seasons"]["targets"]
    keep = (
        counts["season"].between(first, last)
        & (counts["n_noncoors_prior"] >= cfg["traits"]["min_prior_noncoors_pa"])
        & (counts["n_noncoors_same"] >= cfg["target"]["min_same_season_noncoors_pa"])
    )
    return counts[keep]


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = load_config()
    cols = ["batter", "season", "is_coors", "events", "bb_type", "is_k", "launch_speed",
            "is_woba_pa"]  # fmt: skip
    pa = pd.read_parquet(data_path("processed") / "pa.parquet", columns=cols)
    first, last = cfg["seasons"]["pull"]
    pitch_cols = ["batter", "season", "is_coors", "pitch_type", "delta_run_exp"]
    pitches = pd.concat(
        pd.read_parquet(data_path("interim") / f"pitches_{s}.parquet", columns=pitch_cols)
        for s in range(first, last + 1)
    )
    return pa, pitches


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")  # fmt: skip
    pa, pitches = load_inputs()
    feats = build_features(pa, pitches, trait_population(pa))
    feats.to_parquet(data_path("processed") / "features.parquet", index=False)
    log.info("%d hitter-seasons; sprint speed missing for %.1f%%", len(feats),
             100 * feats["sprint_missing"].mean())  # fmt: skip
    log.info("trait correlations:\n%s", feats[TRAITS].corr().round(2))


if __name__ == "__main__":
    main()

"""Phase 4 evaluation: out-of-sample lift, confirmatory tests, and the Rockies transfer test.

- Lift: per-PA reduction in squared error against a baseline on held-out visiting Coors PAs,
  under five-fold CV grouped by batter and under rolling origin (test 2021-2026, train on
  earlier target seasons). Intervals resample batters.
- Confirmatory: PA-weighted least squares of the hitter-season residual on the five
  confirmatory traits and `quality` (A-3), with a batter bootstrap and Holm correction.
- Transfer: M1 fit on visitors predicts Rockies hitters, scored by calibration slope.

Writes data/processed/predictions.parquet, reports/tables/phase4_results.md and
reports/figures/phase4_results.png. Run with `python -m coorsfit.evaluate`.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats

from coorsfit.config import data_path, load_config, rng
from coorsfit.features import CONFIRMATORY, TRAITS
from coorsfit.models import Linear, batter_folds, design, model_set, wls
from coorsfit.targets import TARGET_KEY

log = logging.getLogger(__name__)

PREDICTED_SIGN = {"brk_deficit": +1, "ld_rate": +1, "fb_rate": +1, "fb_ev": +1, "k_rate": -1}
HYPOTHESIS = {"brk_deficit": "H1", "ld_rate": "H2", "fb_rate": "H3a", "fb_ev": "H3b",
              "k_rate": "H4", "sprint_speed": "H5"}  # fmt: skip
COMPARISONS = [  # (model, baseline, role)
    ("M1", "B0q", "headline: traits beyond quality"),
    ("M2", "B0q", "nonlinear check"),
    ("B0q", "B0", "quality slope alone"),
    ("M1u", "B0", "sensitivity: no quality control"),
]


def modeling_table(
    group: str, features: pd.DataFrame | None = None, min_prior: int | None = None
) -> pd.DataFrame:
    """Eligible hitter-seasons of one group with outcome, weights, quality, traits and noise.

    `features` and `min_prior` let the Phase 6 check swap in traits rebuilt for a stricter
    population and keep only hitter-seasons that meet it.
    """
    P = data_path("processed")
    t = pd.read_parquet(P / "targets.parquet")
    t = t[t["eligible"] & (t["group"] == group)]
    if min_prior is not None:
        t = t[t["n_noncoors_prior"] >= min_prior]
    cp = pd.read_parquet(P / "coors_pa.parquet").merge(t[TARGET_KEY], on=TARGET_KEY)
    cp["dev2"] = (cp["resid"] - cp.groupby(TARGET_KEY)["resid"].transform("mean")) ** 2
    ss = cp.groupby(TARGET_KEY)["dev2"].sum().rename("ss_within").reset_index()
    feats = pd.read_parquet(P / "features.parquet") if features is None else features
    df = t.merge(ss, on=TARGET_KEY).merge(feats, on=["batter", "season"], how="left")
    assert df[TRAITS].notna().all().all(), "every eligible hitter-season needs traits"
    df = df.rename(columns={"mean_resid": "y", "n_coors": "n"})
    keep = [*TARGET_KEY, "n", "y", "ss_within", "batter_post_var", "quality", *TRAITS]
    return df[keep].reset_index(drop=True)


def splits(df: pd.DataFrame, scheme: str) -> list[tuple[np.ndarray, np.ndarray]]:
    cfg = load_config()["validation"]
    if scheme == "grouped_cv":
        folds = batter_folds(df["batter"], cfg["grouped_cv_folds"], stream=7)
        return [(folds != f, folds == f) for f in range(cfg["grouped_cv_folds"])]
    last = load_config()["seasons"]["targets"][1]
    return [
        ((df["season"] < y).to_numpy(), (df["season"] == y).to_numpy())
        for y in range(cfg["rolling_origin_first_test"], last + 1)
    ]


def cross_predict(df: pd.DataFrame, scheme: str) -> tuple[pd.DataFrame, list[dict]]:
    """Out-of-sample predictions from every model; each test row is predicted once."""
    out, fits = [], []
    for i, (train, test) in enumerate(splits(df, scheme)):
        rows = df[test].copy()
        sigma2 = df[train]["ss_within"].sum() / (df[train]["n"] - 1).clip(lower=0).sum()
        rows["noise_var"] = sigma2 / rows["n"] + rows["batter_post_var"]
        rows["scheme"], rows["fold"] = scheme, i
        for model in model_set():
            model.fit(df[train])
            rows[f"pred_{model.name}"] = model.predict(rows)
            if isinstance(model, Linear) and model.traits:
                fits.append(
                    {"scheme": scheme, "fold": i, "model": model.name, "alpha": model.alpha}
                )
        out.append(rows)
    return pd.concat(out, ignore_index=True), fits


def lift(pred: pd.DataFrame, model: str, baseline: str, stream: int) -> dict:
    """Per-PA squared-error reduction (wOBA^2), RMSE ratio, and a batter bootstrap interval."""
    n, y = pred["n"].to_numpy(float), pred["y"].to_numpy()
    se_b = n * (y - pred[f"pred_{baseline}"].to_numpy()) ** 2
    se_m = n * (y - pred[f"pred_{model}"].to_numpy()) ** 2
    codes = pd.factorize(pred["batter"])[0]
    d_b = np.bincount(codes, weights=se_b - se_m)
    n_b = np.bincount(codes, weights=n)
    gen = rng(stream)
    reps = load_config()["validation"]["bootstrap_reps"]
    boot = np.empty(reps)
    for r in range(reps):
        k = np.bincount(gen.integers(0, len(d_b), len(d_b)), minlength=len(d_b))
        boot[r] = (k @ d_b) / (k @ n_b)
    alpha = load_config()["validation"]["alpha"]
    within = pred["ss_within"].sum()
    return {
        "model": model,
        "baseline": baseline,
        "hitters": len(d_b),
        "coors_pas": int(n.sum()),
        "lift": d_b.sum() / n.sum(),
        "lift_lo": float(np.quantile(boot, alpha / 2)),
        "lift_hi": float(np.quantile(boot, 1 - alpha / 2)),
        "rmse_ratio": float(np.sqrt((within + se_m.sum()) / (within + se_b.sum()))),
    }


def coverage(pred: pd.DataFrame, model: str) -> dict:
    """Share of held-out hitter-seasons inside 50% and 90% predictive intervals."""
    sd = np.sqrt(pred["noise_var"])
    err = np.abs(pred["y"] - pred[f"pred_{model}"])
    return {
        "model": model,
        "hitter_seasons": len(pred),
        "cover_50": float((err <= stats.norm.ppf(0.75) * sd).mean()),
        "cover_90": float((err <= stats.norm.ppf(0.95) * sd).mean()),
    }


def holm(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adjusted[i] = min(1.0, running)
    return adjusted


def coefficient_tests(df: pd.DataFrame, cols: list[str], stream: int) -> pd.DataFrame:
    """PA-weighted least squares with a batter bootstrap; coefficients in wOBA points."""
    X, y, w = design(df, cols), df["y"].to_numpy(), df["n"].to_numpy(float)
    beta = wls(X, y, w)
    rows_of = df.groupby("batter").indices
    batters = list(rows_of)
    gen = rng(stream)
    boot = []
    for _ in range(load_config()["validation"]["bootstrap_reps"]):
        idx = np.concatenate(
            [rows_of[batters[i]] for i in gen.integers(0, len(batters), len(batters))]
        )
        boot.append(wls(X[idx], y[idx], w[idx]))
    boot = np.array(boot)
    se = boot.std(axis=0, ddof=1)
    alpha = load_config()["validation"]["alpha"]
    out = pd.DataFrame(
        {
            "term": ["intercept", *cols],
            "coef_points": 1000 * beta,
            "lo_points": 1000 * np.quantile(boot, alpha / 2, axis=0),
            "hi_points": 1000 * np.quantile(boot, 1 - alpha / 2, axis=0),
            "se_points": 1000 * se,
            "p": 2 * stats.norm.sf(np.abs(beta / se)),
        }
    )
    return out


def confirmatory(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    results = {}
    for name, cols, stream in [
        ("primary (A-3: with quality)", [*CONFIRMATORY, "quality"], 201),
        ("sensitivity: no quality", CONFIRMATORY, 202),
        ("exploratory: with sprint speed (H5)", [*CONFIRMATORY, "sprint_speed", "quality"], 203),
    ]:
        res = coefficient_tests(df, cols, stream)
        conf = res["term"].isin(CONFIRMATORY)
        res.loc[conf, "p_holm"] = holm(res.loc[conf, "p"].to_numpy())
        res["hypothesis"] = res["term"].map(HYPOTHESIS)
        res["predicted_sign"] = res["term"].map(PREDICTED_SIGN)
        res["sign_matches"] = np.sign(res["coef_points"]) == res["predicted_sign"]
        res["survives_holm"] = conf & (res["p_holm"] < load_config()["validation"]["alpha"])
        results[name] = res
    return results


def transfer(visitors: pd.DataFrame, rockies: pd.DataFrame, stream: int = 301) -> dict:
    """Fit M1 on visitors, predict Rockies hitters, and score by calibration slope."""
    m1 = next(m for m in model_set() if m.name == "M1").fit(visitors)
    r = rockies.copy()
    r["pred"] = m1.predict(r)
    n = r["n"].to_numpy(float)

    def loo_center(col: str) -> np.ndarray:
        """Each row minus the PA-weighted mean of the other batters' rows."""
        tot, wt = (r[col] * r["n"]).sum(), r["n"].sum()
        by_b = (r[col] * r["n"]).groupby(r["batter"]).sum()
        n_b = r["n"].groupby(r["batter"]).sum()
        others = (tot - r["batter"].map(by_b)) / (wt - r["batter"].map(n_b))
        return (r[col] - others).to_numpy()

    yc, pc = loo_center("y"), loo_center("pred")

    def slope(idx: np.ndarray) -> float:
        X = np.column_stack([np.ones(len(idx)), pc[idx]])
        return wls(X, yc[idx], n[idx])[1]

    rows_of = r.groupby("batter").indices
    batters = list(rows_of)
    gen = rng(stream)
    boot = [
        slope(
            np.concatenate(
                [rows_of[batters[i]] for i in gen.integers(0, len(batters), len(batters))]
            )
        )
        for _ in range(load_config()["validation"]["bootstrap_reps"])
    ]
    alpha = load_config()["validation"]["alpha"]
    lift_per_pa = np.sum(n * (yc**2 - (yc - pc) ** 2)) / n.sum()
    return {
        "rockies_hitter_seasons": len(r),
        "rockies_hitters": len(batters),
        "coors_pas": int(n.sum()),
        "sd_predicted_deviation_points": 1000 * float(np.sqrt(np.average(pc**2, weights=n))),
        "slope": slope(np.arange(len(r))),
        "slope_lo": float(np.quantile(boot, alpha / 2)),
        "slope_hi": float(np.quantile(boot, 1 - alpha / 2)),
        "lift_vs_rockies_mean": float(lift_per_pa),
        "m1_alpha": m1.alpha,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")  # fmt: skip
    from coorsfit.evaluate_report import write_report  # figures and tables live there

    visitors, rockies = modeling_table("visitor"), modeling_table("rockies")
    preds, fits = [], []
    for scheme in ("grouped_cv", "rolling_origin"):
        p, f = cross_predict(visitors, scheme)
        preds.append(p)
        fits += f
    preds = pd.concat(preds, ignore_index=True)
    preds.to_parquet(data_path("processed") / "predictions.parquet", index=False)

    lifts = pd.DataFrame(
        [
            {
                "scheme": scheme,
                "role": role,
                **lift(preds[preds["scheme"] == scheme], m, b, 400 + i),
            }
            for scheme in ("grouped_cv", "rolling_origin")
            for i, (m, b, role) in enumerate(COMPARISONS)
        ]
    )
    cover = pd.DataFrame(
        [
            {"scheme": s, **coverage(preds[preds["scheme"] == s], m)}
            for s in ("grouped_cv", "rolling_origin")
            for m in ("B0", "B0q", "M1", "M2")
        ]
    )
    write_report(visitors, preds, lifts, cover, confirmatory(visitors),
                 transfer(visitors, rockies), pd.DataFrame(fits))  # fmt: skip


if __name__ == "__main__":
    main()

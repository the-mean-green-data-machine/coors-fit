"""Phase 3 gate: do hitters differ in their Coors effect once their seasons are pooled?

Primary (pre-registered): tau, the between-hitter SD of the true Coors effect. Each hitter's
Coors residuals are pooled across his eligible seasons, net of each season's group effect, and
tau is estimated by method of moments. The noise term removes three things that are not skill:
- sampling noise, from each hitter's own per-PA variance (outcome volatility differs by hitter),
- PAs within one game sharing shocks (a design effect),
- error in his estimated batter level, which every Coors PA in that season shares.
The interval is a bootstrap over hitters. Secondary: the correlation between a hitter's
odd-season and even-season residuals.

Writes reports/tables/gate.md and reports/figures/phase3_gate.png.
Run with `python -m coorsfit.gate`.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from coorsfit.clean import to_markdown  # noqa: E402
from coorsfit.config import ROOT, data_path, load_config, rng  # noqa: E402
from coorsfit.shrinkage import cluster_slope, design_effect, tau2_anova  # noqa: E402
from coorsfit.targets import TARGET_KEY  # noqa: E402
from coorsfit.targets_report import COLORS, INK, INK_2, MUTED, SURFACE, style  # noqa: E402

MIN_SPLIT_PAS = 30  # Coors PAs a hitter needs in each half for the split-half check


def eligible_pas(group: str) -> pd.DataFrame:
    """Eligible Coors PAs of one group, with `dev` (net of the season effect) and `quality`."""
    P = data_path("processed")
    t = pd.read_parquet(P / "targets.parquet")
    t = t[t["eligible"] & (t["group"] == group)]
    cp = pd.read_parquet(P / "coors_pa.parquet")
    cp = cp.merge(t[[*TARGET_KEY, "season_effect", "quality"]], on=TARGET_KEY)
    cp["dev"] = cp["resid"] - cp["season_effect"]
    return cp


def hitter_table(pas: pd.DataFrame, value: str = "dev") -> pd.DataFrame:
    """One row per hitter: PAs, summed residual, own per-PA variance, expectation-error variance."""
    by_season = pas.groupby(["batter", "season"]).agg(
        n=(value, "size"), post_var=("batter_post_var", "mean")
    )
    # Expectation error is shared within a season and independent across seasons.
    by_season["n2v"] = by_season["n"] ** 2 * by_season["post_var"]
    h = pas.groupby("batter").agg(n=(value, "size"), sum=(value, "sum"), s2=(value, "var"))
    h["v_expect"] = by_season.groupby("batter")["n2v"].sum() / h["n"] ** 2
    within = pas[value] - pas.groupby("batter")[value].transform("mean")
    pooled_s2 = float(np.sum(within**2) / (len(pas) - len(h)))
    h["s2"] = h["s2"].fillna(pooled_s2)  # hitters with a single Coors PA
    h.attrs["pooled_s2"] = pooled_s2
    return h


def tau2_from_hitters(h: pd.DataFrame, deff: float, idx: np.ndarray | None = None) -> float:
    """Unclipped tau^2; `idx` resamples hitters (with repeats) for the bootstrap."""
    d = h if idx is None else h.iloc[idx]
    noise = deff * d["s2"].to_numpy() / d["n"].to_numpy() + d["v_expect"].to_numpy()
    return tau2_anova(d["sum"].to_numpy(), d["n"].to_numpy(), 0.0, extra_var=noise, clip=False)


def gate_estimate(pas: pd.DataFrame, value: str = "dev", stream: int = 0) -> dict:
    cfg = load_config()["validation"]
    h = hitter_table(pas, value)
    codes = pd.factorize(pas["batter"])[0]
    deff = design_effect(
        pas[value].to_numpy(), codes, pd.factorize(pas["game_pk"])[0], h.attrs["pooled_s2"]
    )
    point = tau2_from_hitters(h, deff)
    gen = rng(stream)
    boot = np.array(
        [
            tau2_from_hitters(h, deff, gen.integers(0, len(h), len(h)))
            for _ in range(cfg["bootstrap_reps"])
        ]
    )
    lo, hi = np.quantile(boot, [cfg["alpha"] / 2, 1 - cfg["alpha"] / 2])
    noise_sd = np.sqrt(deff * h["s2"] / h["n"] + h["v_expect"])
    return {
        "hitters": len(h),
        "coors_pas": int(h["n"].sum()),
        "median_pas_per_hitter": float(h["n"].median()),
        "design_effect": deff,
        "median_noise_sd": float(noise_sd.median()),
        "tau2": point,
        "tau2_lo": float(lo),
        "tau2_hi": float(hi),
        "tau": float(np.sqrt(max(point, 0.0))),
        "tau_hi": float(np.sqrt(max(hi, 0.0))),
        "stable_skill": bool(lo > 0),
    }


def split_half(pas: pd.DataFrame, stream: int = 1) -> dict:
    """Correlation of each hitter's odd-season and even-season mean residual."""
    cfg = load_config()["validation"]
    half = np.where(pas["season"] % 2 == 1, "odd", "even")
    m = pas.groupby(["batter", half])["dev"].agg(["mean", "size"]).unstack()
    m = m[(m["size"] >= MIN_SPLIT_PAS).all(axis=1)]
    odd, even = m[("mean", "odd")].to_numpy(), m[("mean", "even")].to_numpy()
    r = float(np.corrcoef(odd, even)[0, 1])
    gen = rng(stream)
    boot = []
    for _ in range(cfg["bootstrap_reps"]):
        i = gen.integers(0, len(m), len(m))
        boot.append(np.corrcoef(odd[i], even[i])[0, 1])
    lo, hi = np.quantile(boot, [cfg["alpha"] / 2, 1 - cfg["alpha"] / 2])
    return {"hitters": len(m), "r": r, "r_lo": float(lo), "r_hi": float(hi),
            "odd": odd, "even": even}  # fmt: skip


def figure(rows: pd.DataFrame, split: dict) -> plt.Figure:
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=SURFACE,
                               gridspec_kw={"width_ratios": [1.15, 1]})  # fmt: skip
    style(a, "How much do hitters truly differ at Coors?", "")
    a.grid(axis="y", visible=False)
    a.grid(axis="x", color="#e6e5e1", linewidth=0.8)
    y = np.arange(len(rows))[::-1]
    for yi, (_, r) in zip(y, rows.iterrows(), strict=True):
        color = COLORS[r["group"]]
        lo = np.sqrt(max(r["tau2_lo"], 0)) * 1000
        a.plot([lo, r["tau_hi"] * 1000], [yi, yi], color=color, linewidth=2.5,
               solid_capstyle="round")  # fmt: skip
        a.plot(r["tau"] * 1000, yi, "o", color=color, markersize=7)
        a.text(r["tau_hi"] * 1000 + 0.6, yi, f"< {r['tau_hi'] * 1000:.0f}", va="center",
               fontsize=8.5, color=INK_2)  # fmt: skip
    a.set_yticks(y, rows["label"], fontsize=8.5, color=INK)
    a.set_xlim(-0.8, max(rows["tau_hi"]) * 1000 + 6)
    a.set_xlabel("Between-hitter SD of the Coors effect (wOBA points); 95% interval",
                 fontsize=8.5, color=INK_2)  # fmt: skip
    a.spines["bottom"].set_visible(True)

    style(b, "Odd seasons don't predict even seasons", "Even-season residual (wOBA)")
    b.axhline(0, color=MUTED, linewidth=1)
    b.axvline(0, color=MUTED, linewidth=1)
    b.scatter(split["odd"], split["even"], s=14, color=COLORS["visitor"], alpha=0.55,
              linewidths=0)  # fmt: skip
    b.set_xlabel(f"Odd-season residual (wOBA); visitors with ≥{MIN_SPLIT_PAS} Coors PAs in each",
                 fontsize=8.5, color=INK_2)  # fmt: skip
    b.text(0.02, 0.96, f"r = {split['r']:+.2f}  [{split['r_lo']:+.2f}, {split['r_hi']:+.2f}]"
           f"   n = {split['hitters']}", transform=b.transAxes, va="top", fontsize=8.5,
           color=INK)  # fmt: skip
    lim = np.quantile(np.abs(np.concatenate([split["odd"], split["even"]])), 0.995) * 1.05
    b.set_xlim(-lim, lim)
    b.set_ylim(-lim, lim)
    fig.tight_layout(w_pad=3)
    return fig


def main() -> None:
    visitors, rockies = eligible_pas("visitor"), eligible_pas("rockies")
    # Supplementary (amendment A-3): residual net of the in-sample slope on hitter quality.
    slope, slope_se = cluster_slope(
        visitors["quality"].to_numpy(), visitors["dev"].to_numpy(), visitors["batter"].to_numpy()
    )
    visitors["dev_q"] = visitors["dev"] - slope * visitors["quality"]

    rows = pd.DataFrame(
        [
            {"group": "visitor", "label": "Visitors (primary)", **gate_estimate(visitors)},
            {"group": "visitor", "label": "Visitors, net of quality",
             **gate_estimate(visitors, "dev_q", stream=2)},
            {"group": "rockies", "label": "Rockies hitters", **gate_estimate(rockies, stream=3)},
        ]
    )  # fmt: skip
    split = split_half(visitors)
    rows.to_parquet(data_path("processed") / "gate.parquet", index=False)
    pd.DataFrame([{k: v for k, v in split.items() if k not in ("odd", "even")}]).to_parquet(
        data_path("processed") / "gate_split.parquet", index=False
    )

    fig = figure(rows, split)
    fig.savefig(ROOT / "reports" / "figures" / "phase3_gate.png", dpi=160, facecolor=SURFACE)

    primary = rows.iloc[0]
    verdict = (
        "**Gate: a stable Coors skill is detected** (the interval for tau-squared excludes zero)."
        if primary["stable_skill"]
        else f"**Gate: no stable Coors skill detected.** The interval for tau-squared includes "
        f"zero; hitters' true Coors effects differ by an SD of less than "
        f"{primary['tau_hi'] * 1000:.0f} wOBA points (upper 95% bound)."
    )
    split_row = pd.DataFrame([{k: v for k, v in split.items() if k not in ("odd", "even")}])
    sections = [
        "# Phase 3 gate",
        "Written by `make gate` (`coorsfit.gate`). Figure: `reports/figures/phase3_gate.png`.",
        verdict,
        "Phase 4 runs either way (pre-registered).",
        "## Between-hitter SD of the Coors effect",
        "Each hitter's Coors residuals are pooled across his eligible seasons, net of the "
        "season's group effect. `tau2` is unclipped (it can be negative when the spread is "
        "smaller than noise predicts); `tau` clips it at zero. Intervals are 95% percentile "
        "bootstraps over hitters. `median_noise_sd` is the typical SD of a hitter's pooled mean "
        "from noise alone.",
        to_markdown(rows.drop(columns="group")),
        f"Quality slope removed for the supplementary row: {slope:+.3f} (SE {slope_se:.3f}).",
        "## Secondary: odd-season vs even-season residuals (visitors)",
        f"Hitters with at least {MIN_SPLIT_PAS} visiting Coors PAs in both halves.",
        to_markdown(split_row),
        "## What this means for Phase 4",
        "A perfect forecast of each hitter's true Coors effect could cut the per-PA squared "
        f"error by at most tau-squared: {max(primary['tau2'], 0):.5f} at the point estimate and "
        f"{max(primary['tau2_hi'], 0):.5f} at the upper bound (wOBA squared per PA).",
    ]
    out = ROOT / "reports" / "tables" / "gate.md"
    out.write_text("\n\n".join(sections) + "\n")
    print(out.read_text())


if __name__ == "__main__":
    main()

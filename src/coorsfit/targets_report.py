"""Phase 2 diagnostics: expectation calibration, residual centering, shrinkage, power.

Writes reports/tables/targets_diagnostics.md and reports/figures/phase2_targets.png from the
files `coorsfit.targets` saves. Run with `python -m coorsfit.targets_report`.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from coorsfit.clean import to_markdown  # noqa: E402
from coorsfit.config import ROOT, data_path  # noqa: E402
from coorsfit.shrinkage import cluster_slope  # noqa: E402
from coorsfit.targets import TARGET_KEY, power  # noqa: E402

# Reference palette, light mode (dataviz skill): slots 1 and 2, text and surface tokens.
COLORS = {"visitor": "#2a78d6", "rockies": "#eb6834"}
LABELS = {"visitor": "Visitors", "rockies": "Rockies hitters"}
INK, INK_2, MUTED, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#fcfcfb"


def cluster_mean_se(values: pd.Series, clusters: pd.Series) -> tuple[float, float]:
    """Mean and its standard error, clustered so one hitter's PAs count as one unit."""
    mean = values.mean()
    totals = (values - mean).groupby(clusters.to_numpy()).sum()
    return float(mean), float(np.sqrt((totals**2).sum()) / len(values))


def eligible_coors_pas() -> pd.DataFrame:
    P = data_path("processed")
    targets = pd.read_parquet(P / "targets.parquet")
    side = pd.read_parquet(P / "pa.parquet", columns=["game_pk", "at_bat_number", "stand"])
    cp = pd.read_parquet(P / "coors_pa.parquet")
    cp = cp.merge(targets[[*TARGET_KEY, "eligible", "season_effect"]], on=TARGET_KEY)
    cp = cp[cp["eligible"]].merge(side, on=["game_pk", "at_bat_number"])
    cp["dev"] = cp["resid"] - cp["season_effect"]
    cp["expected_c"] = cp["expected"] - cp.groupby(["season", "group"])["expected"].transform(
        "mean"
    )
    return cp


def season_effects(cp: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (group, season), d in cp.groupby(["group", "season"]):
        mean, se = cluster_mean_se(d["resid"], d["batter"])
        rows.append({"group": group, "season": season, "pas": len(d), "effect": mean, "se": se})
    return pd.DataFrame(rows)


def centering(cp: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Residual (net of the season effect) by expected-wOBA quintile, and its slope."""
    quintiles, slopes = [], []
    for group, d in cp.groupby("group"):
        q = pd.qcut(d["expected_c"], 5, labels=False)
        for k, dq in d.groupby(q):
            mean, se = cluster_mean_se(dq["dev"], dq["batter"])
            quintiles.append(
                {"group": group, "quintile": k + 1, "pas": len(dq),
                 "expected": dq["expected"].mean(), "dev": mean, "se": se}
            )  # fmt: skip
        slope, se = cluster_slope(
            d["expected_c"].to_numpy(), d["dev"].to_numpy(), d["batter"].to_numpy()
        )
        by_side = {s: cluster_mean_se(ds["dev"], ds["batter"]) for s, ds in d.groupby("stand")}
        slopes.append(
            {"group": group, "pas": len(d), "slope_on_expected": slope, "slope_se": se,
             **{f"dev_{s}": v[0] for s, v in by_side.items()},
             **{f"dev_{s}_se": v[1] for s, v in by_side.items()}}
        )  # fmt: skip
    return pd.DataFrame(quintiles), pd.DataFrame(slopes)


def style(ax: plt.Axes, title: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=10.5, color=INK, pad=8)
    ax.set_ylabel(ylabel, fontsize=9, color=INK_2)
    ax.tick_params(colors=INK_2, labelsize=8.5, length=0)
    ax.grid(axis="y", color="#e6e5e1", linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color("#c9c8c3")


def figure(
    fits: pd.DataFrame,
    effects: pd.DataFrame,
    quint: pd.DataFrame,
    targets: pd.DataFrame,
    sigma: float,
):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.6), facecolor=SURFACE)
    (a, b), (c, d) = axes

    style(a, "Expectation is close to calibrated out of sample", "Held-out slope (1 = calibrated)")
    a.axhline(1, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    a.errorbar(fits["season"], fits["slope"], yerr=1.96 * fits["slope_se"], fmt="o",
               color=COLORS["visitor"], markersize=6, elinewidth=2, capsize=0)  # fmt: skip
    a.set_xticks(fits["season"], [f"'{s % 100:02d}" for s in fits["season"]])
    a.set_xlabel("Fit on odd games, scored on even-game non-Coors PAs; 95% intervals",
                 fontsize=8.5, color=INK_2)  # fmt: skip

    style(b, "Coors residual by hitter quality", "Residual vs season effect (wOBA)")
    b.axhline(0, color=MUTED, linewidth=1)
    for i, (group, g) in enumerate(quint.groupby("group", sort=False)):
        x = g["quintile"] + (i - 0.5) * 0.18
        b.errorbar(x, g["dev"], yerr=1.96 * g["se"], fmt="o", color=COLORS[group],
                   markersize=6, elinewidth=2, capsize=0, label=LABELS[group])  # fmt: skip
    b.set_xticks(range(1, 6), ["1\nweakest", "2", "3", "4", "5\nstrongest"])
    b.set_xlabel("Quintile of expected wOBA at Coors", fontsize=9, color=INK_2)
    b.legend(frameon=False, fontsize=8.5, loc="upper right", labelcolor=INK)

    style(c, "Visitors' Coors effect by season", "Mean residual at Coors (wOBA)")
    v = effects[effects["group"] == "visitor"]
    pooled = (v["effect"] * v["pas"]).sum() / v["pas"].sum()
    c.axhline(pooled, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    c.set_xlabel(
        f"Dashed line: pooled effect {pooled:+.3f}; 95% intervals", fontsize=8.5, color=INK_2
    )
    c.errorbar(v["season"], v["effect"], yerr=1.96 * v["se"], fmt="o", color=COLORS["visitor"],
               markersize=6, elinewidth=2, capsize=0)  # fmt: skip
    c.set_xticks(v["season"], [f"'{s % 100:02d}" for s in v["season"]])
    c.set_ylim(bottom=min(0, c.get_ylim()[0]))

    style(d, "Visitor hitter-seasons spread no wider than noise", "Raw residual vs season effect")
    e = targets[targets["eligible"] & (targets["group"] == "visitor")]
    n = np.arange(1, e["n_coors"].max() + 1)
    d.fill_between(n, -1.96 * sigma / np.sqrt(n), 1.96 * sigma / np.sqrt(n),
                   color="#e6e5e1", linewidth=0, label="95% band from noise alone")  # fmt: skip
    d.scatter(e["n_coors"], e["dev"], s=9, color=COLORS["visitor"], alpha=0.35, linewidths=0,
              label="Hitter-season")  # fmt: skip
    inside = (e["dev"].abs() <= 1.96 * sigma / np.sqrt(e["n_coors"])).mean()
    d.set_xlabel("Coors PAs in the season", fontsize=9, color=INK_2)
    d.set_ylim(-1.0, 1.0)
    d.text(0.99, 0.04, f"{inside:.1%} inside the band (95% expected)", transform=d.transAxes,
           ha="right", fontsize=8, color=INK_2)  # fmt: skip
    d.legend(frameon=False, fontsize=8.5, loc="upper right", labelcolor=INK)

    fig.tight_layout(h_pad=2.5, w_pad=2.5)
    return fig, inside


def main() -> None:
    P = data_path("processed")
    fits = pd.read_parquet(P / "expectation_fits.parquet")
    targets = pd.read_parquet(P / "targets.parquet")
    shrinkage = pd.read_parquet(P / "shrinkage_diagnostics.parquet")
    cp = eligible_coors_pas()
    effects = season_effects(cp)
    quint, slopes = centering(cp)

    sigma = float(shrinkage.set_index("group").loc["visitor", "sigma"])  # per-PA residual SD
    fig, inside = figure(fits, effects, quint, targets, sigma)
    out_fig = ROOT / "reports" / "figures" / "phase2_targets.png"
    fig.savefig(out_fig, dpi=160, facecolor=SURFACE)

    visitors = targets[targets["eligible"] & (targets["group"] == "visitor")]
    pw = pd.DataFrame([power(int(visitors["n_coors"].sum()))])
    fit_cols = ["season", "fit_pas", "pitcher_batting_pas", "sigma", "tau_batter", "tau_pitcher",
                "tau_park", "deff_batter", "deff_pitcher", "home_advantage", "mean_resid",
                "slope", "slope_se"]  # fmt: skip
    sections = [
        "# Phase 2 diagnostics",
        "Written by `make targets` (`coorsfit.targets_report`). Residuals are wOBA points per PA."
        " Standard errors are clustered by batter. Figure: `reports/figures/phase2_targets.png`.",
        "## Expectation fit by season (non-Coors PAs)",
        "`slope` and `mean_resid` are out of sample: fit on odd-numbered games, scored on "
        "even-numbered games. A slope of 1 means the expected values are calibrated.",
        to_markdown(fits[fit_cols]),
        "## Coors effect by season (eligible hitter-seasons)",
        to_markdown(effects),
        "## Residual centering",
        "Residual net of the season's Coors effect, by quintile of expected wOBA at Coors.",
        to_markdown(quint),
        to_markdown(slopes),
        "## Shrinkage (hitter-season level)",
        f"Share of visitor hitter-seasons inside the 95% noise-only band: {inside:.3f}.",
        to_markdown(shrinkage),
        "## Power from eligible visitor Coors PAs",
        "Uses the pre-registered per-PA SD of 0.514, so no outcome enters.",
        to_markdown(pw),
    ]
    out = ROOT / "reports" / "tables" / "targets_diagnostics.md"
    out.write_text("\n\n".join(sections) + "\n")
    print(out.read_text())


if __name__ == "__main__":
    main()

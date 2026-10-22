"""Phase 6 robustness: does the Phase 4 picture hold without 2020, or with a 400-PA trait minimum?

Each variant reruns the primary confirmatory fit (amendment A-3) and the headline lift (M1 vs
B0q) under both validation schemes. The 400-PA variant rebuilds the traits, because the trait
population sets their shrinkage and standardization. Neither variant is confirmatory.

Writes reports/tables/phase6_robustness.md and reports/figures/phase6_robustness.png.
Run with `python -m coorsfit.robustness`.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from coorsfit.clean import to_markdown  # noqa: E402
from coorsfit.config import ROOT, load_config  # noqa: E402
from coorsfit.evaluate import confirmatory, cross_predict, lift, modeling_table  # noqa: E402
from coorsfit.evaluate_report import POINTS2, SCHEME_LABEL  # noqa: E402
from coorsfit.features import CONFIRMATORY, build_features, load_inputs, trait_population  # noqa: E402
from coorsfit.targets_report import INK, INK_2, MUTED, SURFACE, style  # noqa: E402

# Reference palette slots 1-3, which stay distinguishable in every pairing.
VARIANT_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
SCHEMES = ("grouped_cv", "rolling_origin")
HYP = {"brk_deficit": "H1", "ld_rate": "H2", "fb_rate": "H3a", "fb_ev": "H3b", "k_rate": "H4"}


def variants() -> dict[str, pd.DataFrame]:
    cfg = load_config()
    base = modeling_table("visitor")
    strict = cfg["traits"]["sensitivity_min_prior_noncoors_pa"]
    pa, pitches = load_inputs()
    strict_traits = build_features(pa, pitches, trait_population(pa, strict))
    return {
        "Phase 4 (registered)": base,
        "Without 2020": base[~base["season"].isin(cfg["seasons"]["shortened"])].reset_index(
            drop=True
        ),
        f"Trait minimum {strict} PAs": modeling_table("visitor", strict_traits, strict),
    }


def run() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sizes, lifts, coefs = [], [], []
    for name, df in variants().items():
        sizes.append({"variant": name, "hitter_seasons": len(df), "hitters": df["batter"].nunique(),
                      "coors_pas": int(df["n"].sum())})  # fmt: skip
        for scheme in SCHEMES:
            preds, _ = cross_predict(df, scheme)
            # The same bootstrap stream as Phase 4's M1-vs-B0q lift, so the registered row
            # reproduces it exactly; interval endpoints move by about ±15 across streams.
            lifts.append({"variant": name, "scheme": scheme,
                          **lift(preds, "M1", "B0q", stream=400)})  # fmt: skip
        primary = confirmatory(df)["primary (A-3: with quality)"]
        coefs.append(primary[primary["term"].isin(CONFIRMATORY)].assign(variant=name))
    return pd.DataFrame(sizes), pd.DataFrame(lifts), pd.concat(coefs, ignore_index=True)


def figure(lifts: pd.DataFrame, coefs: pd.DataFrame) -> plt.Figure:
    names = list(dict.fromkeys(coefs["variant"]))
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.8), facecolor=SURFACE,
                               gridspec_kw={"width_ratios": [1.15, 1]})  # fmt: skip

    style(a, "Confirmatory trait effects by variant", "")
    a.grid(axis="y", visible=False)
    a.grid(axis="x", color="#e6e5e1", linewidth=0.8)
    a.axvline(0, color=MUTED, linewidth=1)
    for k, trait in enumerate(CONFIRMATORY):
        base_y = len(CONFIRMATORY) - 1 - k
        for v, name in enumerate(names):
            r = coefs[(coefs["variant"] == name) & (coefs["term"] == trait)].iloc[0]
            y = base_y + (1 - v) * 0.22
            a.plot([r["lo_points"], r["hi_points"]], [y, y], color=VARIANT_COLORS[v],
                   linewidth=2.2, solid_capstyle="round")  # fmt: skip
            a.plot(r["coef_points"], y, "o", color=VARIANT_COLORS[v], markersize=5,
                   label=name if k == 0 else None)  # fmt: skip
    a.set_yticks(range(len(CONFIRMATORY))[::-1], [f"{HYP[t]} {t}" for t in CONFIRMATORY],
                 fontsize=8.5, color=INK)  # fmt: skip
    a.set_xlabel("wOBA points per trait SD; 95% intervals", fontsize=8.5, color=INK_2)
    a.legend(frameon=False, fontsize=8, loc="lower right", labelcolor=INK)
    a.spines["bottom"].set_visible(True)

    style(b, "Headline lift: M1 vs B0q", "")
    b.grid(axis="y", visible=False)
    b.grid(axis="x", color="#e6e5e1", linewidth=0.8)
    b.axvline(0, color=MUTED, linewidth=1)
    ticks, labels = [], []
    for s, scheme in enumerate(SCHEMES):
        for v, name in enumerate(names):
            r = lifts[(lifts["variant"] == name) & (lifts["scheme"] == scheme)].iloc[0]
            y = -(s * (len(names) + 1) + v)
            b.plot([r["lift_lo"] * POINTS2, r["lift_hi"] * POINTS2], [y, y],
                   color=VARIANT_COLORS[v], linewidth=2.2, solid_capstyle="round")  # fmt: skip
            b.plot(r["lift"] * POINTS2, y, "o", color=VARIANT_COLORS[v], markersize=5)
            ticks.append(y)
            labels.append(f"{SCHEME_LABEL[scheme].split(' (')[0]} · {name}")
    b.set_yticks(ticks, labels, fontsize=8, color=INK)
    b.set_xlabel("(wOBA points)² per PA; 95% intervals", fontsize=8.5, color=INK_2)
    b.spines["bottom"].set_visible(True)
    fig.tight_layout(w_pad=3)
    return fig


def main() -> None:
    sizes, lifts, coefs = run()
    fig = figure(lifts, coefs)
    fig.savefig(ROOT / "reports" / "figures" / "phase6_robustness.png", dpi=160, facecolor=SURFACE)

    lt = lifts[["variant", "scheme", "hitters", "coors_pas", "lift", "lift_lo", "lift_hi"]].copy()
    for c in ("lift", "lift_lo", "lift_hi"):
        lt[c] = (lt[c] * POINTS2).map("{:+.1f}".format)
    ct = coefs.assign(hypothesis=coefs["term"].map(HYP))[
        ["variant", "hypothesis", "term", "coef_points", "lo_points", "hi_points", "p_holm",
         "sign_matches", "survives_holm"]
    ]  # fmt: skip

    lines = []
    for name, d in coefs.groupby("variant", sort=False):
        passed = d[d["survives_holm"]]
        desc = ", ".join(
            f"{HYP[t]} {t} ({'predicted' if m else 'opposite'} sign)"
            for t, m in zip(passed["term"], passed["sign_matches"], strict=True)
        )
        head = lifts[lifts["variant"] == name].set_index("scheme")
        lift_txt = "; ".join(
            f"{SCHEME_LABEL[s].split(' (')[0].lower()} {head.loc[s, 'lift'] * POINTS2:+.0f} "
            f"[{head.loc[s, 'lift_lo'] * POINTS2:+.0f}, {head.loc[s, 'lift_hi'] * POINTS2:+.0f}]"
            for s in SCHEMES
        )
        lines.append(f"- **{name}:** survive Holm: {desc or 'none'}. Headline lift: {lift_txt}.")

    sections = [
        "# Phase 6 robustness",
        "Written by `make robustness` (`coorsfit.robustness`). Figure: "
        "`reports/figures/phase6_robustness.png`. Neither variant is confirmatory.",
        "## Summary",
        "\n".join(lines),
        "## Samples",
        to_markdown(sizes),
        "## Headline lift (M1 vs B0q), (wOBA points)² per PA",
        to_markdown(lt),
        "## Confirmatory coefficients (primary specification), wOBA points per trait SD",
        to_markdown(ct),
    ]
    out = ROOT / "reports" / "tables" / "phase6_robustness.md"
    out.write_text("\n\n".join(sections) + "\n")
    print(out.read_text())


if __name__ == "__main__":
    main()

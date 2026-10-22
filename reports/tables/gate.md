# Phase 3 gate

Written by `make gate` (`coorsfit.gate`). Figure: `reports/figures/phase3_gate.png`.

**Gate: no stable Coors skill detected.** The interval for tau-squared includes zero; hitters' true Coors effects differ by an SD of less than 27 wOBA points (upper 95% bound).

Phase 4 runs either way (pre-registered).

## Between-hitter SD of the Coors effect

Each hitter's Coors residuals are pooled across his eligible seasons, net of the season's group effect. `tau2` is unclipped (it can be negative when the spread is smaller than noise predicts); `tau` clips it at zero. Intervals are 95% percentile bootstraps over hitters. `median_noise_sd` is the typical SD of a hitter's pooled mean from noise alone.

| label | hitters | coors_pas | median_pas_per_hitter | design_effect | median_noise_sd | tau2 | tau2_lo | tau2_hi | tau | tau_hi | stable_skill |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Visitors (primary) | 708 | 25,275 | 23.000 | 1.001 | 0.104 | -0.000129 | -0.000954 | +0.000704 | 0.000 | 0.027 | False |
| Visitors, net of quality | 708 | 25,275 | 23.000 | 1.001 | 0.104 | -0.000124 | -0.000947 | +0.000736 | 0.000 | 0.027 | False |
| Rockies hitters | 45 | 19,527 | 248.000 | 1.004 | 0.039 | +0.000373 | -0.000092 | +0.000806 | 0.019 | 0.028 | False |

Quality slope removed for the supplementary row: -0.180 (SE 0.162).

## Secondary: odd-season vs even-season residuals (visitors)

Hitters with at least 30 visiting Coors PAs in both halves.

| hitters | r | r_lo | r_hi |
|---:|---:|---:|---:|
| 77 | -0.021 | -0.248 | 0.219 |

## What this means for Phase 4

A perfect forecast of each hitter's true Coors effect could cut the per-PA squared error by at most tau-squared: 0.00000 at the point estimate and 0.00070 at the upper bound (wOBA squared per PA).

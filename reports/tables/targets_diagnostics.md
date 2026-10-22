# Phase 2 diagnostics

Written by `make targets` (`coorsfit.targets_report`). Residuals are wOBA points per PA. Standard errors are clustered by batter. Figure: `reports/figures/phase2_targets.png`.

## Expectation fit by season (non-Coors PAs)

`slope` and `mean_resid` are out of sample: fit on odd-numbered games, scored on even-numbered games. A slope of 1 means the expected values are calibrated.

| season | fit_pas | pitcher_batting_pas | sigma | tau_batter | tau_pitcher | tau_park | deff_batter | deff_pitcher | home_advantage | mean_resid | slope | slope_se |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2015 | 170,642 | 5,366 | 0.509 | 0.031 | 0.027 | 0.007 | 1.006 | 1.079 | 0.015 | -0.003 | 0.973 | 0.077 |
| 2016 | 171,709 | 5,338 | 0.517 | 0.029 | 0.032 | 0.012 | 1.007 | 1.036 | 0.009 | -0.002 | 1.034 | 0.068 |
| 2017 | 172,758 | 5,244 | 0.523 | 0.031 | 0.033 | 0.009 | 1.018 | 1.042 | 0.014 | -0.005 | 1.031 | 0.073 |
| 2018 | 172,845 | 5,103 | 0.515 | 0.029 | 0.030 | 0.010 | 1.012 | 1.051 | 0.007 | 0.000 | 1.005 | 0.079 |
| 2019 | 174,115 | 5,146 | 0.530 | 0.034 | 0.028 | 0.009 | 1.014 | 1.051 | 0.008 | 0.003 | 0.861 | 0.065 |
| 2020 | 63,843 | 3 | 0.519 | 0.026 | 0.038 | 0.013 | 1.035 | 1.003 | 0.018 | -0.005 | 0.694 | 0.101 |
| 2021 | 170,228 | 4,726 | 0.518 | 0.033 | 0.030 | 0.013 | 1.000 | 1.026 | 0.011 | 0.004 | 0.922 | 0.057 |
| 2022 | 174,858 | 6 | 0.505 | 0.032 | 0.031 | 0.006 | 1.013 | 1.016 | 0.009 | 0.001 | 1.001 | 0.074 |
| 2023 | 176,837 | 5 | 0.516 | 0.029 | 0.026 | 0.009 | 1.017 | 1.038 | 0.010 | 0.001 | 0.953 | 0.089 |
| 2024 | 175,376 | 2 | 0.509 | 0.030 | 0.027 | 0.009 | 1.013 | 1.053 | 0.006 | -0.000 | 0.823 | 0.090 |
| 2025 | 175,438 | 7 | 0.510 | 0.029 | 0.031 | 0.008 | 1.020 | 1.057 | 0.010 | -0.006 | 0.912 | 0.076 |
| 2026 | 176,297 | 3 | 0.508 | 0.025 | 0.030 | 0.012 | 1.015 | 1.015 | 0.010 | -0.001 | 0.865 | 0.071 |

## Coors effect by season (eligible hitter-seasons)

| group | season | pas | effect | se |
|---:|---:|---:|---:|---:|
| rockies | 2016 | 1,717 | 0.073 | 0.013 |
| rockies | 2017 | 2,349 | 0.065 | 0.020 |
| rockies | 2018 | 2,340 | 0.064 | 0.016 |
| rockies | 2019 | 1,707 | 0.082 | 0.020 |
| rockies | 2020 | 386 | 0.049 | 0.014 |
| rockies | 2021 | 1,913 | 0.062 | 0.012 |
| rockies | 2022 | 2,009 | 0.061 | 0.013 |
| rockies | 2023 | 1,652 | 0.032 | 0.010 |
| rockies | 2024 | 2,158 | 0.045 | 0.015 |
| rockies | 2025 | 1,529 | 0.054 | 0.015 |
| rockies | 2026 | 1,767 | 0.016 | 0.016 |
| visitor | 2016 | 2,285 | 0.048 | 0.011 |
| visitor | 2017 | 2,235 | 0.033 | 0.013 |
| visitor | 2018 | 2,479 | 0.039 | 0.012 |
| visitor | 2019 | 2,336 | 0.056 | 0.011 |
| visitor | 2020 | 761 | 0.061 | 0.020 |
| visitor | 2021 | 2,231 | 0.012 | 0.012 |
| visitor | 2022 | 2,481 | 0.038 | 0.010 |
| visitor | 2023 | 2,671 | 0.047 | 0.010 |
| visitor | 2024 | 2,557 | 0.030 | 0.010 |
| visitor | 2025 | 2,632 | 0.065 | 0.009 |
| visitor | 2026 | 2,607 | 0.036 | 0.011 |

## Residual centering

Residual net of the season's Coors effect, by quintile of expected wOBA at Coors.

| group | quintile | pas | expected | dev | se |
|---:|---:|---:|---:|---:|---:|
| rockies | 1 | 3,907 | 0.287 | 0.000 | 0.011 |
| rockies | 2 | 3,904 | 0.309 | -0.007 | 0.012 |
| rockies | 3 | 3,907 | 0.323 | 0.005 | 0.009 |
| rockies | 4 | 3,903 | 0.337 | 0.002 | 0.007 |
| rockies | 5 | 3,906 | 0.359 | 0.000 | 0.013 |
| visitor | 1 | 5,055 | 0.294 | 0.016 | 0.007 |
| visitor | 2 | 5,057 | 0.318 | 0.008 | 0.007 |
| visitor | 3 | 5,053 | 0.333 | -0.015 | 0.008 |
| visitor | 4 | 5,058 | 0.348 | 0.001 | 0.008 |
| visitor | 5 | 5,052 | 0.376 | -0.009 | 0.009 |

| group | pas | slope_on_expected | slope_se | dev_L | dev_R | dev_L_se | dev_R_se |
|---:|---:|---:|---:|---:|---:|---:|---:|
| rockies | 19,527 | -0.027 | 0.203 | -0.001 | 0.001 | 0.010 | 0.007 |
| visitor | 25,275 | -0.277 | 0.120 | -0.008 | 0.006 | 0.005 | 0.004 |

## Shrinkage (hitter-season level)

Share of visitor hitter-seasons inside the 95% noise-only band: 0.963.

| group | hitter_seasons | coors_pas | pooled_effect | sigma | design_effect | sd_expectation_error | tau2_unclipped | tau_hitter_season | median_coors_pas | median_shrink_factor | sd_dev_raw | sd_dev_shrunk |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| visitor | 1,877 | 25,275 | 0.041 | 0.546 | 1.017 | 0.019 | -0.002822 | 0.000 | 12.000 | 0.000 | 0.142 | 0.000 |
| rockies | 89 | 19,527 | 0.056 | 0.548 | 1.001 | 0.022 | +0.000241 | 0.016 | 236.000 | 0.120 | 0.045 | 0.005 |

## Power from eligible visitor Coors PAs

Uses the pre-registered per-PA SD of 0.514, so no outcome enters.

| eligible_pas | se | mde_80_alpha05 | mde_80_alpha01 |
|---:|---:|---:|---:|
| 25,275 | 0.003 | 0.009 | 0.011 |

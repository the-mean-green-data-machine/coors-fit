# Phase 4 results

Written by `make models` (`coorsfit.evaluate`). Figure: `reports/figures/phase4_results.png`.

Lift is the reduction in squared error per held-out visiting Coors PA against the baseline, in (wOBA points)²; positive means the model is better. The ceiling is the gate's upper bound on τ², 704 (wOBA points)² per PA: no forecast of hitters' true Coors effects could do better. Intervals resample batters (95%).

## Summary

- **Unseen hitters (grouped CV):** M1 vs B0q lift +154.0 [-47.5, +351.8] (wOBA points)² per PA; RMSE ratio 0.99974.
- **Future seasons (rolling origin):** M1 vs B0q lift -12.4 [-329.0, +301.3] (wOBA points)² per PA; RMSE ratio 1.00002.
- **Confirmatory tests:** 1 of 5 traits have the predicted sign and survive the Holm correction (H3b fb_ev). A trait counts as a finding only if its Phase 5 mechanism check also agrees.
- **Rockies transfer:** calibration slope +0.45 [-0.21, +1.24] across 45 hitters.

## Out-of-sample lift

| scheme | comparison | role | hitters | coors_pas | lift | lift_lo | lift_hi | rmse_ratio | share_of_ceiling |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| grouped_cv | M1 vs B0q | headline: traits beyond quality | 708 | 25,275 | +154.0 | -47.5 | +351.8 | 0.99974 | +0.22 |
| grouped_cv | M2 vs B0q | nonlinear check | 708 | 25,275 | -367.7 | -745.1 | +0.8 | 1.00062 | -0.52 |
| grouped_cv | B0q vs B0 | quality slope alone | 708 | 25,275 | -34.3 | -103.4 | +35.1 | 1.00006 | -0.05 |
| grouped_cv | M1u vs B0 | sensitivity: no quality control | 708 | 25,275 | +99.2 | -61.2 | +265.5 | 0.99983 | +0.14 |
| rolling_origin | M1 vs B0q | headline: traits beyond quality | 508 | 15,179 | -12.4 | -329.0 | +301.3 | 1.00002 | -0.02 |
| rolling_origin | M2 vs B0q | nonlinear check | 508 | 15,179 | -700.0 | -1255.3 | -163.1 | 1.00119 | -0.99 |
| rolling_origin | B0q vs B0 | quality slope alone | 508 | 15,179 | -30.7 | -129.7 | +70.5 | 1.00005 | -0.04 |
| rolling_origin | M1u vs B0 | sensitivity: no quality control | 508 | 15,179 | +30.3 | -218.4 | +287.7 | 0.99995 | +0.04 |

## Confirmatory tests (primary, amendment A-3)

PA-weighted least squares on 1,877 eligible visitor hitter-seasons (25,275 Coors PAs). Coefficients are wOBA points per trait SD (quality: per unit of expected wOBA, ×1000); the intercept is the league Coors effect. Holm correction across the five confirmatory traits.

| hypothesis | term | predicted_sign | coef_points | lo_points | hi_points | p | p_holm | sign_matches | survives_holm |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|  | intercept |  | 40.498 | 33.582 | 46.907 | 0.000 |  |  |  |
| H1 | brk_deficit | + | -9.598 | -15.559 | -3.362 | 0.002 | 0.008 | False | True |
| H2 | ld_rate | + | -3.641 | -10.339 | 3.073 | 0.296 | 0.889 | False | False |
| H3a | fb_rate | + | 1.798 | -4.238 | 8.170 | 0.567 | 1.000 | True | False |
| H3b | fb_ev | + | 12.309 | 2.804 | 21.636 | 0.010 | 0.040 | True | True |
| H4 | k_rate | - | -2.152 | -11.272 | 6.702 | 0.635 | 1.000 | True | False |
|  | quality |  | -359.858 | -704.982 | -14.521 | 0.042 |  |  |  |

## Sensitivity: without the quality control

| hypothesis | term | predicted_sign | coef_points | lo_points | hi_points | p | p_holm | sign_matches | survives_holm |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|  | intercept |  | 41.163 | 34.091 | 47.604 | 0.000 |  |  |  |
| H1 | brk_deficit | + | -9.744 | -15.487 | -3.375 | 0.002 | 0.010 | False | True |
| H2 | ld_rate | + | -4.461 | -11.253 | 1.842 | 0.178 | 0.535 | False | False |
| H3a | fb_rate | + | 1.193 | -4.825 | 7.784 | 0.709 | 1.000 | True | False |
| H3b | fb_ev | + | 8.103 | -0.275 | 16.727 | 0.065 | 0.258 | True | False |
| H4 | k_rate | - | 0.358 | -8.629 | 8.469 | 0.933 | 1.000 | False | False |

## Exploratory: adding sprint speed (H5)

| hypothesis | term | predicted_sign | coef_points | lo_points | hi_points | p | p_holm | sign_matches | survives_holm |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|  | intercept |  | 39.212 | 32.325 | 45.531 | 0.000 |  |  |  |
| H1 | brk_deficit | + | -9.206 | -15.437 | -3.273 | 0.003 | 0.012 | False | True |
| H2 | ld_rate | + | -2.932 | -9.518 | 3.261 | 0.373 | 0.597 | False | False |
| H3a | fb_rate | + | 3.667 | -2.450 | 10.150 | 0.258 | 0.597 | True | False |
| H3b | fb_ev | + | 15.105 | 4.664 | 24.427 | 0.002 | 0.012 | True | True |
| H4 | k_rate | - | -6.009 | -15.655 | 2.965 | 0.199 | 0.597 | True | False |
| H5 | sprint_speed |  | 10.593 | 3.204 | 17.596 | 0.003 |  |  |  |
|  | quality |  | -426.849 | -780.231 | -81.964 | 0.018 |  |  |  |

## Predictive interval coverage (held out)

Intervals are the prediction ± the hitter-season's noise SD (per-PA variance over its Coors PAs plus expectation error) at the 50% and 90% levels.

| scheme | model | hitter_seasons | cover_50 | cover_90 |
|---:|---:|---:|---:|---:|
| grouped_cv | B0 | 1,877 | 0.515 | 0.929 |
| grouped_cv | B0q | 1,877 | 0.517 | 0.928 |
| grouped_cv | M1 | 1,877 | 0.524 | 0.927 |
| grouped_cv | M2 | 1,877 | 0.507 | 0.923 |
| rolling_origin | B0 | 1,158 | 0.519 | 0.938 |
| rolling_origin | B0q | 1,158 | 0.514 | 0.938 |
| rolling_origin | M1 | 1,158 | 0.519 | 0.940 |
| rolling_origin | M2 | 1,158 | 0.502 | 0.935 |

## Rockies transfer test

M1 fit on all visitors predicts Rockies hitters' hitter-season residuals. Both predictions and outcomes are centered on the Rockies average without the hitter's own rows; the slope regresses outcome on prediction, weighted by Coors PAs. Lift is per PA against that Rockies average, in (wOBA points)².

| rockies_hitter_seasons | rockies_hitters | coors_pas | sd_predicted_deviation_points | slope | slope_lo | slope_hi | lift_vs_rockies_mean | m1_alpha |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 89 | 45 | 19,527 | 13.300 | 0.447 | -0.214 | 1.240 | -18.6 | 3000.000 |

## Ridge penalties chosen inside training

| scheme | model | min | 50% | max |
|---:|---:|---:|---:|---:|
| grouped_cv | M1 | 1000.000 | 3000.000 | 10000.000 |
| grouped_cv | M1u | 1000.000 | 3000.000 | 10000.000 |
| rolling_origin | M1 | 1000.000 | 3000.000 | 10000.000 |
| rolling_origin | M1u | 3000.000 | 3000.000 | 10000.000 |

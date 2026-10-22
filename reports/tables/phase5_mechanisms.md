# Phase 5 mechanism checks

Written by `make mechanisms` (`coorsfit.mechanisms`). Rules fixed before running (DECISIONS.md, D-30). Figure: `reports/figures/phase5_mechanisms.png`.

Each outcome is the hitter-season's mean at Coors (visiting PAs) minus his mean at other parks that season, regressed on the five confirmatory traits plus quality, weighted by inverse variance, with 95% batter-bootstrap intervals. `coors_minus_elsewhere` is the weighted average difference (the Coors effect on that outcome). Coefficients are per SD of the trait, in the outcome's unit.

## Verdicts

| hypothesis | trait | confirmatory_coef | confirmatory_p_holm | mechanism_agrees | verdict |
|---:|---:|---:|---:|---:|---:|
| H1 | brk_deficit | -9.598 | 0.008 | False | contradicts the hypothesis (significant, opposite sign) |
| H2 | ld_rate | -3.641 | 0.889 | False | not supported (confirmatory test not passed) |
| H3a | fb_rate | 1.798 | 1.000 | False | not supported (confirmatory test not passed) |
| H3b | fb_ev | 12.309 | 0.040 | False | likely noise: passes the confirmatory test without mechanism evidence |

## H1: does the observed negative effect run through breaking balls?

`brk_deficit` on breaking-ball run value: -0.12 [-0.41, +0.15]; on fastball run value: -0.30 [-0.49, -0.09] (runs per 100 pitches per SD). The observed negative H1 effect does **not** clearly run through breaking balls.

## Each hypothesis's own trait

| hypothesis | metric | role | term | unit | hitter_seasons | coors_events | coors_minus_elsewhere | coef | lo | hi | p |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| H1 | rv100_breaking | channel | brk_deficit | runs per 100 pitches | 1,841 | 28,125 | 0.380 | -0.120 | -0.409 | 0.152 | 0.390 |
| H1 | rv100_fastball | comparison | brk_deficit | runs per 100 pitches | 1,876 | 56,629 | 1.082 | -0.299 | -0.490 | -0.093 | 0.004 |
| H2 | inpark_ld_fb_hit_rate | channel | ld_rate | percentage points | 1,755 | 8,329 | 5.851 | -0.173 | -1.233 | 0.887 | 0.746 |
| H2 | hr_per_pa | control | ld_rate | percentage points | 1,877 | 25,275 | 0.360 | -0.189 | -0.424 | 0.043 | 0.103 |
| H3 | hr_per_fb | channel | fb_rate | percentage points | 1,538 | 4,419 | 2.240 | -0.456 | -1.638 | 0.899 | 0.469 |
| H3 | hr_per_fb | channel | fb_ev | percentage points | 1,538 | 4,419 | 2.240 | 0.027 | -1.580 | 1.648 | 0.974 |
| H3 | fb_distance | channel | fb_rate | feet | 1,530 | 4,357 | 16.034 | -0.457 | -2.658 | 1.777 | 0.689 |
| H3 | fb_distance | channel | fb_ev | feet | 1,530 | 4,357 | 16.034 | 0.566 | -2.389 | 3.498 | 0.704 |
| H3 | fb_woba_minus_x | channel | fb_rate | wOBA points | 1,538 | 4,413 | 86.994 | -2.031 | -21.096 | 17.802 | 0.840 |
| H3 | fb_woba_minus_x | channel | fb_ev | wOBA points | 1,538 | 4,413 | 86.994 | 13.377 | -7.493 | 34.770 | 0.223 |

## All coefficients

| metric | term | coef | lo | hi | p |
|---:|---:|---:|---:|---:|---:|
| rv100_breaking | brk_deficit | -0.120 | -0.409 | 0.152 | 0.390 |
| rv100_breaking | ld_rate | -0.146 | -0.461 | 0.140 | 0.337 |
| rv100_breaking | fb_rate | -0.108 | -0.398 | 0.235 | 0.494 |
| rv100_breaking | fb_ev | 0.385 | -0.021 | 0.741 | 0.051 |
| rv100_breaking | k_rate | -0.251 | -0.598 | 0.116 | 0.167 |
| rv100_breaking | quality | -18.316 | -33.619 | -3.808 | 0.016 |
| rv100_fastball | brk_deficit | -0.299 | -0.490 | -0.093 | 0.004 |
| rv100_fastball | ld_rate | -0.121 | -0.326 | 0.073 | 0.248 |
| rv100_fastball | fb_rate | 0.075 | -0.108 | 0.267 | 0.443 |
| rv100_fastball | fb_ev | 0.239 | -0.023 | 0.498 | 0.073 |
| rv100_fastball | k_rate | 0.027 | -0.234 | 0.277 | 0.834 |
| rv100_fastball | quality | -20.601 | -30.657 | -10.802 | 0.000 |
| inpark_ld_fb_hit_rate | brk_deficit | -0.456 | -1.371 | 0.579 | 0.361 |
| inpark_ld_fb_hit_rate | ld_rate | -0.173 | -1.233 | 0.887 | 0.746 |
| inpark_ld_fb_hit_rate | fb_rate | 0.167 | -0.890 | 1.197 | 0.753 |
| inpark_ld_fb_hit_rate | fb_ev | 0.715 | -0.772 | 2.248 | 0.357 |
| inpark_ld_fb_hit_rate | k_rate | -0.936 | -2.427 | 0.460 | 0.194 |
| inpark_ld_fb_hit_rate | quality | -95.205 | -150.880 | -45.204 | 0.000 |
| hr_per_pa | brk_deficit | -0.242 | -0.498 | -0.004 | 0.055 |
| hr_per_pa | ld_rate | -0.189 | -0.424 | 0.043 | 0.103 |
| hr_per_pa | fb_rate | 0.038 | -0.203 | 0.296 | 0.757 |
| hr_per_pa | fb_ev | 0.083 | -0.210 | 0.361 | 0.571 |
| hr_per_pa | k_rate | 0.254 | -0.040 | 0.556 | 0.089 |
| hr_per_pa | quality | -1.765 | -13.278 | 9.396 | 0.756 |
| hr_per_fb | brk_deficit | -1.136 | -2.274 | 0.055 | 0.052 |
| hr_per_fb | ld_rate | -0.171 | -1.306 | 0.971 | 0.770 |
| hr_per_fb | fb_rate | -0.456 | -1.638 | 0.899 | 0.469 |
| hr_per_fb | fb_ev | 0.027 | -1.580 | 1.648 | 0.974 |
| hr_per_fb | k_rate | 0.912 | -0.565 | 2.303 | 0.224 |
| hr_per_fb | quality | -14.900 | -71.784 | 41.950 | 0.612 |
| fb_distance | brk_deficit | 0.055 | -1.921 | 2.012 | 0.955 |
| fb_distance | ld_rate | -0.355 | -2.518 | 1.870 | 0.752 |
| fb_distance | fb_rate | -0.457 | -2.658 | 1.777 | 0.689 |
| fb_distance | fb_ev | 0.566 | -2.389 | 3.498 | 0.704 |
| fb_distance | k_rate | 1.590 | -1.171 | 4.210 | 0.249 |
| fb_distance | quality | -14.894 | -115.527 | 81.389 | 0.766 |
| fb_woba_minus_x | brk_deficit | -9.141 | -22.419 | 5.536 | 0.204 |
| fb_woba_minus_x | ld_rate | 7.118 | -11.322 | 25.223 | 0.442 |
| fb_woba_minus_x | fb_rate | -2.031 | -21.096 | 17.802 | 0.840 |
| fb_woba_minus_x | fb_ev | 13.377 | -7.493 | 34.770 | 0.223 |
| fb_woba_minus_x | k_rate | -2.086 | -24.431 | 17.077 | 0.845 |
| fb_woba_minus_x | quality | -344.236 | -1120.021 | 441.797 | 0.392 |

# Phase 6 robustness

Written by `make robustness` (`coorsfit.robustness`). Figure: `reports/figures/phase6_robustness.png`. Neither variant is confirmatory.

## Summary

- **Phase 4 (registered):** survive Holm: H1 brk_deficit (opposite sign), H3b fb_ev (predicted sign). Headline lift: unseen hitters +154 [-48, +352]; future seasons -12 [-329, +301].
- **Without 2020:** survive Holm: H1 brk_deficit (opposite sign). Headline lift: unseen hitters +128 [-54, +304]; future seasons -44 [-391, +299].
- **Trait minimum 400 PAs:** survive Holm: H1 brk_deficit (opposite sign), H3b fb_ev (predicted sign). Headline lift: unseen hitters +44 [-94, +185]; future seasons -57 [-415, +282].

## Samples

| variant | hitter_seasons | hitters | coors_pas |
|---:|---:|---:|---:|
| Phase 4 (registered) | 1,877 | 708 | 25,275 |
| Without 2020 | 1,819 | 707 | 24,514 |
| Trait minimum 400 PAs | 1,536 | 570 | 21,476 |

## Headline lift (M1 vs B0q), (wOBA points)² per PA

| variant | scheme | hitters | coors_pas | lift | lift_lo | lift_hi |
|---:|---:|---:|---:|---:|---:|---:|
| Phase 4 (registered) | grouped_cv | 708 | 25,275 | +154.0 | -47.5 | +351.8 |
| Phase 4 (registered) | rolling_origin | 508 | 15,179 | -12.4 | -329.0 | +301.3 |
| Without 2020 | grouped_cv | 707 | 24,514 | +127.6 | -54.0 | +304.3 |
| Without 2020 | rolling_origin | 508 | 15,179 | -43.6 | -390.9 | +298.6 |
| Trait minimum 400 PAs | grouped_cv | 570 | 21,476 | +44.1 | -94.2 | +185.4 |
| Trait minimum 400 PAs | rolling_origin | 412 | 13,095 | -56.9 | -414.8 | +282.0 |

## Confirmatory coefficients (primary specification), wOBA points per trait SD

| variant | hypothesis | term | coef_points | lo_points | hi_points | p_holm | sign_matches | survives_holm |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Phase 4 (registered) | H1 | brk_deficit | -9.598 | -15.559 | -3.362 | 0.008 | False | True |
| Phase 4 (registered) | H2 | ld_rate | -3.641 | -10.339 | 3.073 | 0.889 | False | False |
| Phase 4 (registered) | H3a | fb_rate | 1.798 | -4.238 | 8.170 | 1.000 | True | False |
| Phase 4 (registered) | H3b | fb_ev | 12.309 | 2.804 | 21.636 | 0.040 | True | True |
| Phase 4 (registered) | H4 | k_rate | -2.152 | -11.272 | 6.702 | 1.000 | True | False |
| Without 2020 | H1 | brk_deficit | -9.301 | -15.101 | -3.264 | 0.012 | False | True |
| Without 2020 | H2 | ld_rate | -3.453 | -10.454 | 3.583 | 0.993 | False | False |
| Without 2020 | H3a | fb_rate | 2.470 | -3.935 | 9.424 | 0.993 | True | False |
| Without 2020 | H3b | fb_ev | 11.558 | 1.711 | 21.380 | 0.074 | True | False |
| Without 2020 | H4 | k_rate | -2.911 | -11.907 | 5.792 | 0.993 | True | False |
| Trait minimum 400 PAs | H1 | brk_deficit | -9.862 | -17.042 | -2.791 | 0.028 | False | True |
| Trait minimum 400 PAs | H2 | ld_rate | -1.870 | -8.899 | 5.345 | 1.000 | False | False |
| Trait minimum 400 PAs | H3a | fb_rate | -0.159 | -7.364 | 7.059 | 1.000 | False | False |
| Trait minimum 400 PAs | H3b | fb_ev | 14.929 | 4.146 | 25.005 | 0.024 | True | True |
| Trait minimum 400 PAs | H4 | k_rate | -3.860 | -13.690 | 5.222 | 1.000 | True | False |

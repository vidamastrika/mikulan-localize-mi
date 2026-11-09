# Primary no-registration diagnostic comparison

**Primary no-warp/template values are unregistered HEAD-coordinate distances, not validated anatomical localization errors.** Historical baseline within-subject errors appear separately as context.

Planned jobs: 5,124. Paired PASS: 5,109; template-only PASS: 10; no-warp-only PASS: 0; both FAIL: 5.

All 7 participants, including sub-03, and all 61 EEG runs are included. Only original no-warp primary cross-subject rows enter this comparison. No-warp/template frozen configurations match. Checked EEG preprocessing metadata agrees on paired successes. No MRI mapping, inverse fits, parameter changes or sensitivity substitution.

Difference = template distance minus no-warp distance on identical successful jobs. Negative means smaller **coordinate separation**, not established anatomical improvement. Median paired difference is not the difference between condition medians.

| Strategy | Method | Paired jobs | No-warp paired median (mm) | Template paired median (mm) | Median paired delta (mm) | Failures no-warp / template |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| LOSO | MNE | 61 | 32.17 | 38.16 | +2.72 | 0 / 0 |
| LOSO | dSPM | 61 | 24.06 | 25.55 | +8.16 | 0 / 0 |
| LOSO | sLORETA | 61 | 26.73 | 34.40 | +2.72 | 0 / 0 |
| LOSO | eLORETA | 61 | 25.44 | 35.35 | +11.29 | 0 / 0 |
| LOSO | Continuous-ECD | 61 | 24.70 | 32.08 | +5.16 | 0 / 0 |
| LOSO | MxNE | 61 | 28.82 | 36.46 | +5.47 | 0 / 0 |
| LOSO | RAP-MUSIC | 61 | 26.02 | 30.99 | +7.86 | 0 / 0 |
| LOSO | LCMV | 61 | 30.08 | 37.79 | +3.60 | 0 / 0 |
| LOSO | irMxNE | 61 | 30.05 | 36.49 | +6.42 | 0 / 0 |
| LOSO | Champagne | 61 | 33.64 | 33.15 | +1.01 | 0 / 0 |
| LOSO | Gamma-MAP | 60 | 29.80 | 35.54 | +7.09 | 1 / 0 |
| LOSO | SESAME | 61 | 27.30 | 25.55 | +2.53 | 0 / 0 |
| Single-reference | MNE | 366 | 39.69 | 38.16 | -1.94 | 0 / 0 |
| Single-reference | dSPM | 366 | 31.38 | 27.43 | -1.39 | 0 / 0 |
| Single-reference | sLORETA | 366 | 36.02 | 35.07 | -2.23 | 0 / 0 |
| Single-reference | eLORETA | 366 | 35.28 | 33.25 | +0.13 | 0 / 0 |
| Single-reference | Continuous-ECD | 366 | 34.15 | 32.10 | -1.59 | 0 / 0 |
| Single-reference | MxNE | 366 | 36.48 | 35.22 | -1.45 | 0 / 0 |
| Single-reference | RAP-MUSIC | 366 | 32.63 | 32.14 | -0.50 | 0 / 0 |
| Single-reference | LCMV | 366 | 36.81 | 36.95 | -1.15 | 0 / 0 |
| Single-reference | irMxNE | 363 | 35.87 | 36.18 | -0.08 | 3 / 0 |
| Single-reference | Champagne | 366 | 39.64 | 37.18 | -1.81 | 0 / 0 |
| Single-reference | Gamma-MAP | 357 | 39.13 | 39.12 | +1.34 | 9 / 5 |
| Single-reference | SESAME | 364 | 30.31 | 26.51 | -3.41 | 2 / 0 |

## Output guide

tables/01: full paired ledger; 02: method summaries; 03: participant summaries; 04: reference-owner summaries; 05: equal-participant summaries after reference/run balancing; 06: unique-run summaries; 07: input hashes; 08: all nonpaired/failing jobs; 09: configuration/preprocessing checks; 10: participant run-balanced medians.

Figures: 01/02 paired distributions by strategy; 03 median paired differences; 04 participant profiles; 05 failure counts; 06 explicit sub-03 comparison. Method colors/markers follow Script 26.

Single Reference uses six parameter owners per test run; these are repeated evaluations, not six independent recordings. Use the run, participant and reference tables alongside pooled summaries. No p-values or independence claims are made.

MRI-based Script 05 outputs remain supplementary and were not read. Full limitations and input/output hashes are in comparison_manifest.json.

## Historical 12-method baseline context

Baseline source: Script 26 complete_all_pass globally selected fixed configurations, 732 unique method/run results. This is an in-sample within-subject anatomical error baseline, not the same metric or selection design as cross-model unregistered HEAD distances. No MRI mapping is added to either cross-model condition. No baseline-relative improvement percentages or paired error deltas are calculated.

Each Single Reference recording is collapsed to the median distance over the same successful reference pairs before adding one baseline observation. All three distributions use common available test runs per method/strategy. Baseline is stored once per method/run; repeating its value in two strategy views does not add independent observations.

| Strategy | Method | Common runs | Historical within-subject baseline median (mm) | No-warp run diagnostic median (mm) | Template run diagnostic median (mm) |
| --- | --- | ---: | ---: | ---: | ---: |
| LOSO | MNE | 61 | 24.58 | 32.17 | 38.16 |
| LOSO | dSPM | 61 | 18.68 | 24.06 | 25.55 |
| LOSO | sLORETA | 61 | 19.89 | 26.73 | 34.40 |
| LOSO | eLORETA | 61 | 21.38 | 25.44 | 35.35 |
| LOSO | Continuous-ECD | 61 | 17.97 | 24.70 | 32.08 |
| LOSO | MxNE | 61 | 17.60 | 28.82 | 36.46 |
| LOSO | RAP-MUSIC | 61 | 18.63 | 26.02 | 30.99 |
| LOSO | LCMV | 61 | 16.40 | 30.08 | 37.79 |
| LOSO | irMxNE | 61 | 17.60 | 30.05 | 36.49 |
| LOSO | Champagne | 61 | 24.05 | 33.64 | 33.15 |
| LOSO | Gamma-MAP | 60 | 23.02 | 29.80 | 35.54 |
| LOSO | SESAME | 61 | 14.20 | 27.30 | 25.55 |
| Single-reference | MNE | 61 | 24.58 | 39.82 | 37.63 |
| Single-reference | dSPM | 61 | 18.68 | 31.65 | 26.59 |
| Single-reference | sLORETA | 61 | 19.89 | 35.60 | 34.19 |
| Single-reference | eLORETA | 61 | 21.38 | 35.28 | 31.83 |
| Single-reference | Continuous-ECD | 61 | 17.97 | 33.56 | 31.45 |
| Single-reference | MxNE | 61 | 17.60 | 36.44 | 31.82 |
| Single-reference | RAP-MUSIC | 61 | 18.63 | 32.13 | 30.39 |
| Single-reference | LCMV | 61 | 16.40 | 36.47 | 34.68 |
| Single-reference | irMxNE | 61 | 17.60 | 35.52 | 35.36 |
| Single-reference | Champagne | 61 | 24.05 | 42.97 | 34.06 |
| Single-reference | Gamma-MAP | 61 | 22.88 | 37.87 | 39.12 |
| Single-reference | SESAME | 61 | 14.20 | 29.81 | 27.67 |

Tables 11–15 supply the original baseline rows/definitions, per-run context, method context and participant context. Figures 07/08 show the three-condition distributions; 09 shows participant profiles. Metric labels must remain visible when sharing these figures.

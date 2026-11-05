# Output folder: 09_visualize_results

Produced by [Script 09](../01_scripts/09_visualize_results.md). Local inventory and CSV record counts were read directly from disk.

## Files and records

- `00_master_three_strategies_boxplot.png` — PNG figure.
- `00_master_three_strategies_statistics.csv` — CSV, 36 records; columns: method, strategy, N, Mean_mm, Median_mm, SD_mm, Q1_mm, Q3_mm, IQR_mm, Min_mm, Max_mm
- `01_within_boxplot.png` — PNG figure.
- `02_cross_boxplot.png` — PNG figure.
- `03_pass_rate_grouped.png` — PNG figure.
- `04_reference_sensitivity.png` — PNG figure.
- `05_reference_by_method_heatmap.png` — PNG figure.
- `06_test_reference_heatmap.png` — PNG figure.
- `07_plot_manifest.json` — JSON manifest/audit.
- `README.txt` — text readme/log.

## Interpretation and downstream use

Rows are scheduled run/method/condition records unless stated otherwise. `status` distinguishes solver PASS from FAIL; PASS does not validate anatomical accuracy. Within-subject error is computed between a test subject estimate and its test-subject ground truth in the corresponding head frame. Cross-subject no-warp values are raw head-coordinate separation between the test ground truth and an estimate from a reference head model; they are not validated anatomical localization errors.

The LOSO primary contains 1 failure; the single-reference primary contains 14 failures. Sensitivity outputs preserve primary rows and mark selective whole-brain retries. Downstream consumers include Scripts 10–09 where applicable. Detailed numerical summaries are preserved in the CSVs; no unverified statistics are asserted here.

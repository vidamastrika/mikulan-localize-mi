# Output folder: 07_compare_strategies

Produced by [Script 07](../01_scripts/07_compare_strategies.md). Local inventory and CSV record counts were read directly from disk.

## Files and records

- `01_master_run_level.csv` — CSV, 12444 records; PASS=12418, FAIL=26; columns: strategy, scenario, experiment, method, test_subject, reference_subject, run, condition, status, metric, value_mm, metric_valid_as_anatomical_error, metric_interpretation
- `02_master_method_summary.csv` — CSV, 84 records; columns: scenario, experiment, method, metric, scheduled, pass_count, fail_count, pass_rate, anatomical_error_valid, n, mean_mm, median_mm, std_mm, q25_mm, q75_mm, iqr_mm
- `03_master_overall_summary.csv` — CSV, 7 records; columns: scenario, experiment, metric, scheduled, pass_count, fail_count, pass_rate, n, mean_mm, median_mm, std_mm, q25_mm, q75_mm, iqr_mm, min_mm, max_mm
- `04_baseline_vs_loso_within.csv` — CSV, 732 records; columns: method, test_subject, run, baseline_error_mm, loso_within_error_mm, loso_minus_baseline_mm
- `05_baseline_vs_loso_within_summary.csv` — CSV, 12 records; columns: method, n, mean_mm, median_mm, std_mm, q25_mm, q75_mm, iqr_mm, min_mm, max_mm
- `06_cross_no_warp_pair_summary.csv` — CSV, 732 records; columns: method, test_subject, reference_subject, run, loso_raw_mm, single_reference_raw_mm, single_minus_loso_raw_mm, interpretation
- `07_validation_report.json` — JSON manifest/audit.
- `README.txt` — text readme/log.

## Interpretation and downstream use

Rows are scheduled run/method/condition records unless stated otherwise. `status` distinguishes solver PASS from FAIL; PASS does not validate anatomical accuracy. Within-subject error is computed between a test subject estimate and its test-subject ground truth in the corresponding head frame. Cross-subject no-warp values are raw head-coordinate separation between the test ground truth and an estimate from a reference head model; they are not validated anatomical localization errors.

The LOSO primary contains 1 failure; the single-reference primary contains 14 failures. Sensitivity outputs preserve primary rows and mark selective whole-brain retries. Downstream consumers include Scripts 08–09 where applicable. Detailed numerical summaries are preserved in the CSVs; no unverified statistics are asserted here.

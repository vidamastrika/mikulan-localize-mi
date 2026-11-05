# Output folder: 08_reference_sensitivity

Produced by [Script 08](../01_scripts/08_assess_reference_sensitivity.md). Local inventory and CSV record counts were read directly from disk.

## Files and records

- `01_reference_run_level.csv` — CSV, 8784 records; PASS=8759, FAIL=25; columns: method, test_subject, run, reference_subject, scenario, status, raw_head_separation_mm, configuration_key, failure_reason, hemisphere, sensitivity_override, peak_selection_policy
- `02_reference_method_summary.csv` — CSV, 168 records; columns: scenario, method, reference_subject, scheduled, pass_count, fail_count, pass_rate, n, mean_mm, median_mm, std_mm, q25_mm, q75_mm, iqr_mm, min_mm, max_mm
- `03_test_run_sensitivity.csv` — CSV, 1464 records; columns: scenario, method, test_subject, run, scheduled_references, reference_count, pass_count, fail_count, pass_rate, range_mm, cv, n, mean_mm, median_mm, std_mm, q25_mm
- `04_method_sensitivity_summary.csv` — CSV, 24 records; columns: scenario, method, test_runs, scheduled, pass_count, fail_count, pass_rate, complete_six_reference_runs, incomplete_test_runs, median_across_reference_sd_mm, mean_across_reference_sd_mm, median_across_reference_range_mm, pooled_raw_n, pooled_raw_mean_mm, pooled_raw_median_mm, pooled_raw_std_mm
- `05_reference_pairing_matrix.csv` — CSV, 1008 records; columns: scenario, method, test_subject, reference_subject, scheduled, pass_count, fail_count, pass_rate, n, mean_mm, median_mm, std_mm, q25_mm, q75_mm, iqr_mm, min_mm
- `06_primary_vs_sensitivity.csv` — CSV, 4392 records; columns: method, test_subject, run, reference_subject, primary_status, primary_raw_mm, sensitivity_status, sensitivity_raw_mm, changed_status, changed_raw_mm, sensitivity_minus_primary_raw_mm
- `07_baseline_context.csv` — CSV, 12 records; columns: scenario, experiment, method, metric, scheduled, pass_count, fail_count, pass_rate, anatomical_error_valid, n, mean_mm, median_mm, std_mm, q25_mm, q75_mm, iqr_mm
- `08_validation_report.json` — JSON manifest/audit.

## Interpretation and downstream use

Rows are scheduled run/method/condition records unless stated otherwise. `status` distinguishes solver PASS from FAIL; PASS does not validate anatomical accuracy. Within-subject error is computed between a test subject estimate and its test-subject ground truth in the corresponding head frame. Cross-subject no-warp values are raw head-coordinate separation between the test ground truth and an estimate from a reference head model; they are not validated anatomical localization errors.

The LOSO primary contains 1 failure; the single-reference primary contains 14 failures. Sensitivity outputs preserve primary rows and mark selective whole-brain retries. Downstream consumers include Scripts 09–09 where applicable. Detailed numerical summaries are preserved in the CSVs; no unverified statistics are asserted here.

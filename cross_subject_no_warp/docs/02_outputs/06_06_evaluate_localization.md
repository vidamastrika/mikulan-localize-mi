# Output folder: 06_evaluate_localization

Produced by [Script 06](../01_scripts/06_evaluate_localization_and_coordinate_diagnostics.md). Local inventory and CSV record counts were read directly from disk.

## Files and records

- `01_input_validation.csv` — CSV, 4 records; columns: strategy, total, passed, failed, override_count, variant_pass, variant_fail, untouched_records
- `02_coordinate_validation.csv` — CSV, 61 records; columns: test_subject, run, coordinate_audit, coordinate_audit_reason, gt_head_x_mm, gt_head_y_mm, gt_head_z_mm, gt_surface_x_m, gt_surface_y_m, gt_surface_z_m, gt_hemisphere, gt_stimulation_pair, contact_sha256, transform_sha256
- `03_primary_localization.csv` — CSV, 5856 records; PASS=5841, FAIL=15; columns: strategy, scenario, fold_id, test_subject, reference_subject, run, method, condition, model_subject, configuration_key, status, failure_reason, configuration_parameters, stimulation_pair, hemisphere, channels
- `04_sensitivity_localization.csv` — CSV, 5856 records; PASS=5845, FAIL=11; columns: strategy, scenario, fold_id, test_subject, reference_subject, run, method, condition, model_subject, configuration_key, status, failure_reason, configuration_parameters, stimulation_pair, hemisphere, channels
- `05_method_summary.csv` — CSV, 72 records; columns: scenario, strategy, method, condition, scheduled, passed, failed, error_n, median_error_mm, q25_error_mm, q75_error_mm, cross_error_valid
- `06_failure_analysis.csv` — CSV, 26 records; columns: scenario, strategy, method, reference_subject, test_subject, run, condition, failure_reason
- `07_evaluation_manifest.json` — JSON manifest/audit.
- `08_raw_head_diagnostic.csv` — CSV, 10248 records; PASS=10222, FAIL=26; columns: strategy, scenario, method, reference_subject, test_subject, run, status, raw_head_separation_mm, valid_localization_error
- `README.txt` — text readme/log.

## Interpretation and downstream use

Rows are scheduled run/method/condition records unless stated otherwise. `status` distinguishes solver PASS from FAIL; PASS does not validate anatomical accuracy. Within-subject error is computed between a test subject estimate and its test-subject ground truth in the corresponding head frame. Cross-subject no-warp values are raw head-coordinate separation between the test ground truth and an estimate from a reference head model; they are not validated anatomical localization errors.

The LOSO primary contains 1 failure; the single-reference primary contains 14 failures. Sensitivity outputs preserve primary rows and mark selective whole-brain retries. Downstream consumers include Scripts 07–09 where applicable. Detailed numerical summaries are preserved in the CSVs; no unverified statistics are asserted here.

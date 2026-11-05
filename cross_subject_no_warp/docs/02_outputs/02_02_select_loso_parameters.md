# Output folder: 02_select_loso_parameters

Produced by [Script 02](../01_scripts/02_select_loso_parameters.md). Local inventory and CSV record counts were read directly from disk.

## Files and records

- `01_loso_selected_parameters.csv` — CSV, 84 records; columns: fold_id, test_subject, reference_subject, training_subjects, method, configuration_key, configuration_parameters, primary_parameter, primary_value, expected_runs, observed_runs, missing_runs, unexpected_runs, invalid_runs, duplicate_conflict, unresolved_failed_settings
- `02_fold_selection_coverage.csv` — CSV, 84 records; columns: fold_id, test_subject, reference_subject, training_subjects, method, expected_training_runs, candidate_configurations, eligible_configurations, selection_status
- `03_training_configuration_audit.csv` — CSV, 67723 records; columns: method, configuration_key, configuration_parameters, primary_parameter, primary_value, expected_runs, observed_runs, missing_runs, unexpected_runs, invalid_runs, duplicate_conflict, unresolved_failed_settings, configuration_all_pass, score_mean_subject_median_mm, pooled_median_mm, pooled_mean_mm
- `04_training_primary_value_audit.csv` — CSV, 265 records; columns: method, primary_parameter, primary_value, observed_configurations, ineligible_configurations, primary_all_pass, fold_id, test_subject, reference_subject, training_subjects
- `05_training_duplicate_audit.csv` — CSV, 12 records; columns: method, configuration_key, subject, run, rows, conflict, fold_id, test_subject, reference_subject, training_subjects
- `06_parameter_fields.json` — JSON manifest/audit.
- `07_selection_manifest.json` — JSON manifest/audit.
- `08_source_manifest.json` — JSON manifest/audit.
- `README.txt` — text readme/log.

## Interpretation and downstream use

Rows are scheduled run/method/condition records unless stated otherwise. `status` distinguishes solver PASS from FAIL; PASS does not validate anatomical accuracy. Within-subject error is computed between a test subject estimate and its test-subject ground truth in the corresponding head frame. Cross-subject no-warp values are raw head-coordinate separation between the test ground truth and an estimate from a reference head model; they are not validated anatomical localization errors.

The LOSO primary contains 1 failure; the single-reference primary contains 14 failures. Sensitivity outputs preserve primary rows and mark selective whole-brain retries. Downstream consumers include Scripts 03–09 where applicable. Detailed numerical summaries are preserved in the CSVs; no unverified statistics are asserted here.

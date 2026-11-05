# Output folder: 04_single_reference_parameters

Produced by [Script 04](../01_scripts/04_select_single_reference_parameters.md). Local inventory and CSV record counts were read directly from disk.

## Files and records

- `01_single_reference_selected_parameters.csv` — CSV, 84 records; columns: method, reference_subject, training_subjects, configuration_key, configuration_parameters, primary_parameter, primary_value, expected_runs, observed_runs, missing_runs, unexpected_runs, invalid_runs, duplicate_conflict, unresolved_failed_settings, configuration_all_pass, score_mean_subject_median_mm
- `02_reference_selection_coverage.csv` — CSV, 84 records; columns: method, reference_subject, training_subjects, expected_reference_runs, candidate_configurations, eligible_configurations, selection_status
- `03_reference_configuration_audit.csv` — CSV, 67103 records; columns: method, configuration_key, configuration_parameters, primary_parameter, primary_value, expected_runs, observed_runs, missing_runs, unexpected_runs, invalid_runs, duplicate_conflict, unresolved_failed_settings, configuration_all_pass, score_mean_subject_median_mm, pooled_median_mm, pooled_mean_mm
- `04_reference_primary_value_audit.csv` — CSV, 260 records; columns: method, primary_parameter, primary_value, observed_configurations, ineligible_configurations, primary_all_pass, reference_subject, training_subjects
- `05_reference_duplicate_audit.csv` — CSV, 2 records; columns: method, configuration_key, subject, run, rows, conflict, reference_subject, training_subjects
- `06_parameter_fields.json` — JSON manifest/audit.
- `07_selection_manifest.json` — JSON manifest/audit.
- `08_source_manifest.json` — JSON manifest/audit.
- `09_reference_test_schedule.csv` — CSV, 366 records; columns: reference_subject, test_subject, run
- `README.txt` — text readme/log.

## Interpretation and downstream use

Rows are scheduled run/method/condition records unless stated otherwise. `status` distinguishes solver PASS from FAIL; PASS does not validate anatomical accuracy. Within-subject error is computed between a test subject estimate and its test-subject ground truth in the corresponding head frame. Cross-subject no-warp values are raw head-coordinate separation between the test ground truth and an estimate from a reference head model; they are not validated anatomical localization errors.

The LOSO primary contains 1 failure; the single-reference primary contains 14 failures. Sensitivity outputs preserve primary rows and mark selective whole-brain retries. Downstream consumers include Scripts 05–09 where applicable. Detailed numerical summaries are preserved in the CSVs; no unverified statistics are asserted here.

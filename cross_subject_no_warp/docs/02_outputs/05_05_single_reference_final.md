# Output folder: 05_single_reference_final

Produced by [Script 05](../01_scripts/05_reconstruct_and_finalize_single_reference_experiment.md). Local inventory and CSV record counts were read directly from disk.

## Files and records

- `01_execution_status.csv` — CSV, 4392 records; PASS=4378, FAIL=14; columns: fold_id, reference_subject, test_subject, run, method, condition, model_subject, configuration_key, parameter_selection, status, failure_reason, configuration_parameters, stimulation_pair, hemisphere, channels, actual_channels_json
- `02_method_reference_summary.csv` — CSV, 84 records; columns: method, reference_subject, expected, pass_count, fail_count
- `03_failed_cases.csv` — CSV, 14 records; PASS=0, FAIL=14; columns: fold_id, reference_subject, test_subject, run, method, condition, model_subject, configuration_key, parameter_selection, status, failure_reason, configuration_parameters, stimulation_pair, hemisphere, channels, actual_channels_json
- `04_channel_selection_audit.csv` — CSV, 4378 records; PASS=4378, FAIL=0; columns: method, reference_subject, test_subject, run, condition, status, montage_template_subject, n_selected_channels, n_bad_test_channels, n_nominal_channels, n_replacements, actual_channels_json, nominal_channels_json, replacement_details_json, original_bad_channels_json, eeg_data_sha256
- `05_validation_report.json` — JSON manifest/audit.
- `06_provenance.csv` — CSV, 7 records; columns: artifact, sha256
- `README.txt` — text readme/log.

## Interpretation and downstream use

Rows are scheduled run/method/condition records unless stated otherwise. `status` distinguishes solver PASS from FAIL; PASS does not validate anatomical accuracy. Within-subject error is computed between a test subject estimate and its test-subject ground truth in the corresponding head frame. Cross-subject no-warp values are raw head-coordinate separation between the test ground truth and an estimate from a reference head model; they are not validated anatomical localization errors.

The LOSO primary contains 1 failure; the single-reference primary contains 14 failures. Sensitivity outputs preserve primary rows and mark selective whole-brain retries. Downstream consumers include Scripts 06–09 where applicable. Detailed numerical summaries are preserved in the CSVs; no unverified statistics are asserted here.

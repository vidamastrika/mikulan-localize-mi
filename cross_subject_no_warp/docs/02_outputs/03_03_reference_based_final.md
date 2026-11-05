# Output folder: 03_reference_based_final

Produced by [Script 03](../01_scripts/03_reconstruct_reference_based_loso_conditions.md). Local inventory and CSV record counts were read directly from disk.

## Files and records

- `01_execution_status.csv` — CSV, 1464 records; PASS=1463, FAIL=1; columns: fold_id, test_subject, reference_subject, run, method, condition, model_subject, configuration_key, status, failure_reason, configuration_parameters, stimulation_pair, hemisphere, channels, actual_channels_json, nominal_channels_json
- `02_method_condition_summary.csv` — CSV, 25 records; PASS=24, FAIL=1; columns: method, condition, status, count
- `03_merge_provenance.csv` — CSV, 1464 records; PASS=1463, FAIL=1; columns: method, test_subject, run, condition, status, record_source, configuration_key, eeg_data_sha256, epoch_times_sha256, source_csv_sha256
- `04_validation_report.json` — JSON manifest/audit.
- `05_channel_selection_summary.csv` — CSV, 1464 records; PASS=1463, FAIL=1; columns: method, test_subject, run, condition, status, record_source, reference_subject, model_subject, montage_template_subject, n_nominal, n_selected, n_original_bad, n_replacements, channel_names_unique, selected_include_bad, actual_channels_json
- `06_paired_input_audit.csv` — CSV, 732 records; columns: method, test_subject, run, identical_inputs, fields_checked
- `07_known_failure.json` — JSON manifest/audit.
- `README.txt` — text readme/log.

## Interpretation and downstream use

Rows are scheduled run/method/condition records unless stated otherwise. `status` distinguishes solver PASS from FAIL; PASS does not validate anatomical accuracy. Within-subject error is computed between a test subject estimate and its test-subject ground truth in the corresponding head frame. Cross-subject no-warp values are raw head-coordinate separation between the test ground truth and an estimate from a reference head model; they are not validated anatomical localization errors.

The LOSO primary contains 1 failure; the single-reference primary contains 14 failures. Sensitivity outputs preserve primary rows and mark selective whole-brain retries. Downstream consumers include Scripts 04–09 where applicable. Detailed numerical summaries are preserved in the CSVs; no unverified statistics are asserted here.

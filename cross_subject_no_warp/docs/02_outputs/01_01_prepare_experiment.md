# Output folder: 01_prepare_experiment

Produced by [Script 01](../01_scripts/01_prepare_experiment_schedule_and_coordinate_audit.md). Local inventory and CSV record counts were read directly from disk.

## Files and records

- `01_loso_folds.csv` — CSV, 7 records; columns: fold_id, test_subject, reference_subject, training_subjects, n_training_subjects, n_test_runs
- `02_fold_membership.csv` — CSV, 49 records; columns: fold_id, subject, role, is_reference
- `03_single_reference_cases.csv` — CSV, 61 records; columns: fold_id, test_subject, reference_subject, run
- `04_reconstruction_plan.csv` — CSV, 1464 records; columns: fold_id, test_subject, reference_subject, run, method, condition, model_subject, parameter_fold, parameter_status
- `05_coordinate_audit.csv` — CSV, 7 records; PASS=0, FAIL=0; columns: subject, status, failure_reason, forward_path, head_to_surface_path, contacts_path, h5_vs_forward_transform_max_abs, rotation_orthogonality_error, rotation_determinant, roundtrip_max_mm, source_frame_codes, source_max_head_radius_m, n_sources, n_forward_channels, h5_sha256, forward_sha256
- `06_coordinate_transforms.json` — JSON manifest/audit.
- `07_experiment_manifest.json` — JSON manifest/audit.
- `README.txt` — text readme/log.

## Interpretation and downstream use

Rows are scheduled run/method/condition records unless stated otherwise. `status` distinguishes solver PASS from FAIL; PASS does not validate anatomical accuracy. Within-subject error is computed between a test subject estimate and its test-subject ground truth in the corresponding head frame. Cross-subject no-warp values are raw head-coordinate separation between the test ground truth and an estimate from a reference head model; they are not validated anatomical localization errors.

The LOSO primary contains 1 failure; the single-reference primary contains 14 failures. Sensitivity outputs preserve primary rows and mark selective whole-brain retries. Downstream consumers include Scripts 02–09 where applicable. Detailed numerical summaries are preserved in the CSVs; no unverified statistics are asserted here.

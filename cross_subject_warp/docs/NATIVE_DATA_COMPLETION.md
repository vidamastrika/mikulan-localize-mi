# Completing native LOSO and single-reference data

`03_05_finalize_native_datasets.py` is a completion/import helper for the
existing Script 03/05 experiments. It does not add a new main numbered stage.
It preserves original source folders, refuses existing destinations and writes
into the planned warp experiment's final and sensitivity folders only after
validation succeeds.

```bash
python cross_subject_warp/scripts/03_05_finalize_native_datasets.py \
  --preparation-root cross_subject_warp/outputs/01_prepare_experiment_review
```

The helper loads the released EEG to repeat only the frozen loader,
average reference and montage/channel selection. It recomputes numerical
EEG/time fingerprints, selected/nominal/bad channels, replacements, sampling
metadata, stimulation pair and hemisphere for each preparation group. The
inverse entry points `Engine.solve` and `Engine.prepare` are disabled while this
runs. No noise covariance estimation, inverse fitting, peak extraction,
registration or anatomical mapping occurs.

Existing nonempty metadata must match the recomputed values. Only blank input
metadata is completed, with field-level evidence. Failed attempts remain FAIL;
their peaks and execution failure reasons remain untouched. Missing covariance
method labels are recovered only from unanimous successful preparation records
with the same test run, reference and montage. Those records' data fingerprints
are independently checked too. This is recovered input evidence, not a claim
that missing solver diagnostics were historically recorded.

The helper validates:

- Both 84-selection sets and all 61 runs against pinned grids/configurations.
- Historical run manifest dataset, software, selection and source identities.
- Forward/H5/contact checksums against the historical coordinate audit, plus BEM
  checksums recorded by successful ECD attempts.
- Every primary row's original/patch attempt lineage and exact retry coverage.
- One LOSO and three reference-only sensitivity substitutions against saved
  retry records, preserving documented whole-brain policy annotations.
- Native peak frame/units, finite coordinates, failure reasons and blank failed
  peaks, without rerunning a solver.
- Source stability and copy checksums before publication.

Outputs preserve the planned folder names:

| Folder | Dataset |
|---|---|
| `03_reference_based_final/` | LOSO primary |
| `05_single_reference_final/` | Reference-only primary |
| `03_gamma_wholebrain_sensitivity/` | LOSO sensitivity, one saved replacement |
| `05_wholebrain_hemisphere_sensitivity/` | Reference-only sensitivity, three saved replacements |
| `03_05_native_completion_audit/` | Attempt lineage, input checks, source hashes, preparation/donor evidence, completion manifest and copied historical audit evidence |

Each data folder contains the completed `01_execution_status.csv`, numbered
summaries/audits, a field-level metadata-completion CSV and validation report.
The original condition is retained as `source_condition`; cross rows are named
`cross_reference`. Affine/SyN still share these native peaks. Sensitivity remains
a separate scenario, and the helper does not replace primary failures with
sensitivity successes.

Completion means every planned attempt has a preserved outcome and verified
input metadata. It does not mean every solver succeeded or every optimizer
converged. Saved peaks are imported verbatim; mathematical solver reproduction
has not been performed. Historical executable hashes remain visible in the
copied manifests, including any exact intermediate patch executable that was
not archived. Physical head/surface-to-NIfTI RAS and registration QA remain
separate gates before anatomical localization errors can be calculated.

## Completed local run

The helper completed successfully. All 1,184 distinct preparation recipes were
verified against released EEG. All 11,712 primary/sensitivity rows retain their
original configurations, outcomes, failure reasons and native peak values.

| Dataset | Rows | PASS | FAIL | Rows with completed input metadata |
|---|---:|---:|---:|---:|
| LOSO primary | 1,464 | 1,463 | 1 | 1 |
| Reference-only primary | 4,392 | 4,378 | 14 | 14 |
| LOSO sensitivity | 1,464 | 1,464 | 0 | 0 |
| Reference-only sensitivity | 4,392 | 4,381 | 11 | 11 |

The completion ledger records 17 filled fields in LOSO primary, 238 in
reference-only primary and 187 in reference-only sensitivity. Existing nonempty
fields were retained. All 732 LOSO within/reference pairs have identical
recomputed EEG/time/channel fingerprints.

Both post-completion reuse audits exited 0:
`outputs/03_reference_based_completed_reuse_audit/` and
`outputs/05_single_reference_completed_reuse_audit/`. They report zero record
mismatches and no remaining incomplete-failure metadata warnings. Earlier audit
folders are preserved as historical evidence; their warnings refer to the
uncompleted original records. New copies were not written into historical folders.

`03_05_native_completion_audit/07_post_completion_validation.json` records the
post-run checks. `08_historical_executable_resolution.json` records one specific
provenance gap: the exact intermediate Script 03 patch executable matching
SHA-256 `5e8b534a695c074063f1aec8e76021d04f7250b55191aab654e8c4767abe9817`
was not found among available archived script sources. Its run manifest, 103
attempt records, configuration/selection identities and original-failure retry
lineage are retained; imported input fingerprints were independently verified.
This gap is not silently described as mathematical solver reproduction.

Compilation, 21 synthetic regression tests, overwrite protection, source/copy
checksums and post-completion record checks passed. No inverse reconstruction,
covariance fitting or registration fitting ran. Coordinate-bridge and
registration QA remain unresolved for later anatomical evaluation.

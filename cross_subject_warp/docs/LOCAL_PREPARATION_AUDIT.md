# Local preparation audit, 2026-11-02

Script 01 was implemented and run on `experiment/loso-single-reference-warp`.
No registration fitting, inverse reconstruction or localization evaluation ran.
Original inputs were preserved. This local inspection supersedes earlier
statements that these resources were unavailable in the cloud audit instance.

## Located resources

- Dataset: `/home/girvi949/Documents/DATASETS SOURCE LOCALIZATION/Localize-MI`.
- Registration/QA originals: `cross_subject/outputs/`.
- Canonical inverse originals: `cross_subject_no_warp/outputs/`.
- BEMs: `outputs/12_additional_methods_all_runs/bem/`, all seven subjects.

Copied 626 registration/QA files (18,177,989,412 bytes) into the ignored
`source/01_registration_resources/outputs/`, preserving original folder names.
The copy includes stages 09, 19, 31 and coordinate/registration QA stages
10–16, 18, 20–21. Each original and copy has matching SHA-256 hashes recorded
with absolute paths in `outputs/01_prepare_experiment/10_source_provenance.json`.
The final audit used `--verify-existing-imports`, which rechecked existing copies
without writing them. The first audit reports were retained in
`/tmp/warp-preparation-initial-audit/` while correcting sensitivity filenames and
adding explicit required EEG run-file checks; originals/imports were unchanged.

## Observed coverage

| Resource | Observed |
|---|---|
| Authoritative schedule | 366 reference/run cases, 61 runs, 42 directed pairs |
| Preparation plan | 7 folds, 12 methods, 1,464 LOSO fits planned only |
| Single-reference design | 4,392 fits planned only |
| Affine-only resources | Expected matrix and warped image present for 42/42 pairs |
| SyN resources | Expected independent affine, forward/inverse fields, both warped images, quality JSON and log present for 42/42 pairs |
| Numerical head/surface audit | PASS_NUMERICAL for all seven subjects |
| Frozen source provenance | All 69 file checksums match |
| LOSO canonical primary | 1,464 rows: 1,463 PASS, 1 FAIL |
| Single-reference canonical primary | 4,392 rows: 4,378 PASS, 14 FAIL |
| LOSO whole-brain variant | 1,464 rows: 1,464 PASS |
| Single-reference whole-brain variant | 4,392 rows: 4,381 PASS, 11 FAIL |

Canonical records and supporting evidence are hashed in place, not imported as
validated new experiment results. The variants use
`01_execution_status_wholebrain_variant.csv` and
`04_execution_status_variant.csv`, respectively. Primary and sensitivity remain
separate. Row counts are inventory evidence, not reconstruction reuse approval.

No missing resources were found among the explicitly checked required files:
61 runs' epoch arrays, sidecars, epoch/channel TSVs; seven forward/H5/contact/MRI
sets; seven BEM solutions; canonical/variant records; and registration chains.
400 local input/support files were hashed, in addition to copied and frozen
resources. The 42 saved SyN quality reports are listed in artifact 09; affine
quality and broader coordinate QA evidence are retained in the copied QA folders
and file-level provenance.

## Unresolved readiness gates

The experiment remains `BLOCKED_UNRESOLVED_COORDINATES_AND_RESOURCE_VALIDATION`.
The audit intentionally exits 2 after writing its reports.

- Establish authoritative head/surface-to-NIfTI physical RAS bridges for every
  subject. Numerical H5 inversion/forward agreement does not establish this.
- Confirm contact units and anatomical fiducial definitions from metadata.
- Match original registration MRI hashes and ANTs command/software provenance;
  current MRI hashes alone do not prove the same images were fitted historically.
- Validate image/field grids and physical point directions using MRI-only QA.
  Preserve historical RAS/LPS conversion and tested point-transform order.
  Saved SyN reports explicitly say point transforms are unvalidated; foreground
  overlap is not cortical alignment validation.
- Before reusing inverse peaks, validate configurations, software, data/channel
  fingerprints, model/BEM identity and peak policy. Preserve the historical
  hemisphere restriction and separate whole-brain sensitivities.

Checks performed: full copy checksum comparison and subsequent verification,
frozen-source hash verification, schedule/fold/reference invariants, numerical
coordinate audit, Python compilation, existing-output and existing-resource
refusal checks, and `git diff --check`. Generated datasets and imported large
resources are ignored by Git. Scripts 02–09 remain unimplemented.

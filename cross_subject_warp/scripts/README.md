# Runnable stages

`01_prepare_experiment.py` prepares the historical LOSO plans and audits local
inputs. It never calls registration or inverse fitting code.

From the project root:

```bash
python cross_subject_warp/scripts/01_prepare_experiment.py
```

Use `--dataset`, `--cross-subject-outputs`, and `--no-warp-outputs` to select
existing local resources. Defaults resolve relative to the repository, not the
current working directory. Preparation uses the authoritative historical
366-row schedule and validates its 61 runs and per-subject counts.

The script refuses existing preparation output directories and imported
registration destinations. `--verify-existing-imports` allows a fresh audit
against previously copied resources only after comparing complete file trees
and every checksum; it writes no registration files. Partial imports must be
inspected. It checks free
space, preserves source files, rejects symlinks inside imports, and verifies
source/copy SHA-256 checksums. Large copied resources remain ignored under
`source/01_registration_resources/outputs/`.

Outputs retain artifact names 01–07, add `08_registration_pair_inventory.csv`,
`09_registration_qa.csv`, `10_source_provenance.json`, and
`11_all_reference_cases.csv`. Reconstruction plans use `cross_reference`, retain
the historical condition in `source_condition`, and share fits between affine
and SyN. Registration mode belongs to the pair inventory and later evaluation.
Canonical reconstruction files are inventoried and hashed in place; importing
their results requires further configuration and fingerprint validation.

Exit code 2 means the audit completed with unresolved readiness gates. File
completeness and numerical rigidity do not validate anatomical coordinates.
Saved QA is retained as evidence; this stage does not certify field grids,
ANTs point transforms, original MRI identity, or the provisional head/NIfTI
bridge. Inspect the manifest and missing resources before further work.

`02_select_loso_parameters.py` verifies frozen raw grids and matches fresh
Script 01 plans to the historical plans, then selects 84 method/fold parameter
sets using the unchanged training-only selection policy. It does not need the
anatomical coordinate bridge and does not run inverse fitting or registration.
Sample first-run and repeat commands are comments at the top of the script.

```bash
python cross_subject_warp/scripts/02_select_loso_parameters.py \
  --preparation-root cross_subject_warp/outputs/01_prepare_experiment_review
```

Its default output is `outputs/02_select_loso_parameters/`, with numbered
artifacts 01–08 from the historical workflow. Existing output directories are
refused; use a fresh `--output-root` for a repeat. The selection manifest records
the current preparation files and unresolved anatomical readiness separately.
Frozen historical manifests are preserved. No new snapshot is created, and
source overrides must match the pinned experiment bytes.

`03_run_localization.py` is implemented but has not been run on experiment
inputs. The default phase is `reuse-audit`, which compares the primary canonical
records against the new selections and native reconstruction plan. It neither
imports peaks nor invokes solvers. Candidate matches remain pending independent
software, model/BEM and raw EEG/channel provenance validation.

```bash
python cross_subject_warp/scripts/03_run_localization.py \
  --phase reuse-audit \
  --preparation-root cross_subject_warp/outputs/01_prepare_experiment_review \
  --output-root cross_subject_warp/outputs/03_reference_based_reuse_audit
```

This writes `02_reconstruction_plan.csv`, `03_configuration_preflight.csv`,
`05_resolved_configurations.json`, `11_reuse_candidate_audit.csv` and
`12_reuse_audit_manifest.json`. It intentionally does not write an
`01_execution_status.csv` or a final canonical folder for audited candidates.
Exit 2 indicates record mismatches; exit 0 still does not approve reuse.
Primary and whole-brain sensitivity records remain separate.

Use `--phase plan` for configuration preflight only. Reconstruction requires
explicit `--phase reconstruct`; commands are in the opening comments. It uses
the frozen patched solver implementation, shares native fits between affine and
SyN, and writes standard reconstruction artifacts under `03_reference_based/`.
Cross jobs use `cross_reference` and preserve the historical condition in
`source_condition`. Outputs cannot overlap historical/input trees. Plan/audit
folders must be new; reconstruction resume requires the same frozen run manifest
and retains PASS and FAIL rows. Coordinate bridge readiness remains unresolved.

Checks performed without experiment execution: compilation, CLI help, and
synthetic regression tests covering reuse candidates, malformed peaks,
overwrite protection, and equality of solver/selection code to frozen history.
No new finalizer or validated peak import is implemented yet.

`04_select_single_reference_parameters.py` is implemented and checked without
running selection on experiment inputs. It uses Script 02's pinned grids and
selection functions, filtering to the reference subject before repair,
reliability screening or ranking. It does not consume Script 03's inverse peaks
or require its reuse audit to pass.

```bash
python cross_subject_warp/scripts/04_select_single_reference_parameters.py \
  --preparation-root cross_subject_warp/outputs/01_prepare_experiment_review
```

Default output: `outputs/04_single_reference_parameters/`. Expected: 84
reference/method selections and 366 non-self reference/test/run cases (42
subject pairs). Artifacts 01–09 retain their historical names. Existing outputs
are refused; use a new `--output-root` for repeats. An interrupted selection
leaves an IN_PROGRESS manifest; missing winners produce INCOMPLETE and nonzero
exit, without fallback to LOSO/global winners. Anatomical readiness remains
separate. This stage never invokes inverse solvers or registration fitting.

The inspected Script 03 reuse report contains 1,464 rows and one metadata
mismatch: the retained failed Gamma-MAP/sub-05/run-08/cross attempt has an empty
montage-template field. Its audit exits 2, which commonly makes the shell prompt
red. This is not evidence that new fits ran or that the failed record should be
removed. Reuse validation remains pending.

Scripts 05–09 are not implemented. Preserved historical scripts under
`source/` must not be run as new warp stages.

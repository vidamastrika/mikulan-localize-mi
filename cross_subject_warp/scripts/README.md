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

`05_run_single_reference.py` is implemented and checked without running on
experiment inputs. Script 04 must first be COMPLETE with all 84 selections.
The default phase audits existing primary single-reference records against
reference-only configurations; reference is part of each key.

```bash
python cross_subject_warp/scripts/05_run_single_reference.py \
  --phase reuse-audit \
  --preparation-root cross_subject_warp/outputs/01_prepare_experiment_review
```

Default outputs by phase:

| Phase | Folder under `outputs/` | Behavior |
|---|---|---|
| `reuse-audit` | `05_single_reference_reuse_audit/` | Check primary canonical records; no import or solver |
| `plan` | `05_single_reference_plan/` | Resolve 84 configurations and plan 4,392 cross jobs |
| `reconstruct` | `05_single_reference_reconstruction/` | Explicit native inverse fitting, shared by later affine/SyN mappings |

Plan/audit writes historical artifacts 02, 03 and 05; reuse-audit additionally
writes `11_reuse_candidate_audit.csv` and `12_reuse_audit_manifest.json`. It does
not write `01_execution_status.csv` or create a finalized canonical dataset.
Standard reconstruction numbering is retained when reconstruction is explicitly
requested. Example terminal commands are comments at the beginning of the file.

An audit exits 2 for record mismatches or incomplete metadata on failed attempts.
Absent failure metadata is distinguished from conflicting recorded settings,
never filled silently and never used to turn FAIL into PASS. Even exit 0 leaves
independent raw EEG, software, model/BEM and physical-coordinate validation
pending. Whole-brain sensitivity records remain separate from primary records.

Existing plan/audit folders and historical/input destinations are refused.
Reconstruction resume requires the same frozen scripts, selection, schedule,
solver source, coordinate audit, dataset path and software manifest. Both PASS
and FAIL remain recorded; no automatic retries. Changed settings require a new
output folder. Script 05 has no dependency on successful Script 03 reuse records;
it uses Script 03's checked solver and configuration functions.

Compilation, help output and 14 synthetic regression tests passed. Tests cover
all 4,392 unique directed jobs, reference-only parameter resolution, failed
metadata preservation, malformed peaks, overwrite protection and both no-solver
main paths. No experiment phase was run during implementation.

The helper `03_05_finalize_native_datasets.py` now completes/imports both native
primary datasets and their separate sensitivity variants. See
[the completion report](../docs/NATIVE_DATA_COMPLETION.md). It rechecks released
EEG fingerprints/channel selection without covariance estimation or inverse
fitting, fills only missing input metadata with provenance, and preserves every
original outcome/peak. Both completed-copy reuse audits pass. The exact intermediate
historical patch executable is unavailable; this is documented separately from
native data completeness.

Script 06 now supports both input/header preflight (`--phase audit`) and
saved-peak mapped evaluation (`--phase evaluate`). The native bridge helper and
registration QA helper are described below. Evaluation follows the explicitly
accepted per-mode policy: keep all rows, map only passing mode/pairs, and retain
failed pairs with blank errors. See the mapped-evaluation section at the end.

The earlier `outputs/06_evaluate_localization/` preflight report remains preserved.
Audit mode does not perform mapping or consume the later QA reports as mapping
authorization; evaluate mode performs those checks. Always use a new output
folder. Scripts 07–09 remain unimplemented. Historical scripts under `source/`
remain reference copies and must not be run as new warp stages.

### Next Script 06 step: native coordinate bridge

The actual native pial GIFTIs contain FreeSurfer `VolGeom` metadata, an
independent source of the native voxel geometry. The helper
`06_validate_coordinate_bridge.py` derives `Norig @ inv(Torig)` using that
metadata from both hemispheres. This is the documented FreeSurfer surface-RAS
to scanner-RAS relationship, not an affine fitted to contact locations.

```bash
python cross_subject_warp/scripts/06_validate_coordinate_bridge.py \
  --preparation-root cross_subject_warp/outputs/01_prepare_experiment_review
```

It checks seven subjects against the actual maskface MRI voxel geometry,
qform/sform codes and agreement, H5/forward MRI-head transform agreement, and
released surface/T1w contacts. Fixed tolerance is 0.1 mm. It records but never
applies the GIFTI unknown-to-Talairach transform to these native coordinates.
It also tests whether the older head-as-scanner identity assumption follows
from the independently derived bridge; identity is not assumed as an input.

Output: `outputs/06_coordinate_bridge/` (must be new):
`01_subject_bridge_audit.csv`, `02_native_physical_bridges.json`,
`03_contact_consistency.csv`, `04_source_provenance.json`, and
`05_bridge_manifest.json`. Passing subjects get explicit surface/head to scanner
RAS matrices in millimetres. Failed subjects do not get published bridge matrices.
Exit 0 requires all seven to pass; exit 2 writes the failed-subject report.

The helper is written and checked, not run on subject data. Compilation, help,
and 35 synthetic tests passed, including an end-to-end synthetic subject and
rejection of shifted MRI headers, invalid units/rigidity, conflicting contact
identities and output overwrites. No EEG arrays, fitting, registration, ANTs
calls, source-peak mapping or localization errors occur.

A pass addresses native geometry consistency only. Transform content checksums,
original registration MRI provenance, affine/SyN field-grid/point-direction QA
remain separate from native bridge validation; mapped evaluation now has an
explicit per-mode QA policy described below. The earlier preflight
reports remain immutable; they do not automatically consume a later bridge
report. Do not start Script 07 before evaluation tables exist.

References: [FreeSurfer coordinate systems](https://surfer.nmr.mgh.harvard.edu/fswiki/CoordinateSystems),
[NiBabel FreeSurfer headers](https://nipy.org/nibabel/reference/nibabel.freesurfer.html).


### Next Script 06 step: existing registration resource QA

The user-run native bridge check passed all seven subjects. The report also
supports the historical head-as-scanner identity to less than 0.0001 mm; use the
explicit derived matrices rather than assuming exact identity.

`06_validate_registration_resources.py` is written and checked, not run on the
actual registrations. It preserves pinned Script 32's point transform sequences
and RAS/LPS conversions. Only `antsApplyTransformsToPoints` is called; it never
invokes `antsRegistration`, inverse solvers, or localization-error calculation.

```bash
micromamba run -n localizemi-registration .venv/bin/python \
  cross_subject_warp/scripts/06_validate_registration_resources.py \
  --preparation-root cross_subject_warp/outputs/01_prepare_experiment_review
```

The micromamba invocation supplies ANTs runtime libraries while retaining the
project's Python/packages. Plain project `python` also works if that runtime is
already available; `--ants-bin` may select its `antsApplyTransformsToPoints`.
The helper verifies approximately 18 GB of imported resources at both the start
and end, so allow time for checksum reads and point-mapping checks.

Expected: 84 checks (42 directed pairs x affine/SyN). Fixed=test MRI and
moving=reference MRI. Checks cover import hashes, MRI/bridge source identities,
log/quality subject identities, saved image grids, finite vector-field grids,
MRI point-direction/intensity reproduction, and bidirectional MRI foreground
point roundtrips. The collapsed historical SyN forward AND inverse fields use
the fixed grid; the saved inverse-warped image uses the reference grid.

Preserved criteria: 200 MRI-only foreground samples, seed 2901, at least 75%
valid coverage, image reproduction RMSE <=0.015, point roundtrip <=0.1 mm.
Physical grid corner discrepancy must be <=0.02 mm. No source peaks, EEG,
contacts or localization outcomes are used to choose QA samples or transforms.
Both SyN saved images are checked for intensity consistency. No thresholds are
changed based on results.

Output: `outputs/06_registration_qa/` (new directory required):
`01_pair_mode_qa.csv`, `02_import_checksum_audit.csv`,
`03_field_grid_audit.json`, `04_source_provenance.json`, and
`05_registration_qa_manifest.json`.
Exit 0 requires all 84 checks to pass; exit 2 retains per-mode failures for
inspection. No automatic retry/fitting or silent transform substitution exists.
Input/checksum corruption raises a specific error before a passing QA report
can be published.

Fit-time MRI byte identity and original ANTs version are explicitly unverified,
not invented. Current source hashes, saved-image consistency, log identities,
and current point-tool version/hash are recorded. A numerical pass establishes
mapping implementation consistency, not cortical anatomical registration
accuracy. Mapped evaluation now permits partial pair coverage, preserving all
failed-pair flags and blank errors instead of accepting those pairs.

Compilation/help and 42 synthetic tests passed, including vector-field checks,
ordered point terms, RAS/LPS conversion, roundtrip/coverage/RMSE failures,
independent mode reporting and overwrite protection. No subject-data QA or
ANTs command was executed during implementation.

Point convention reference: [ANTs forward/inverse image and point transforms](https://github.com/ANTsX/ANTs/wiki/Forward-and-inverse-warps-for-warping-images%2C-pointsets-and-Jacobians).


### Diagnosing the six SyN roundtrip failures

The user-run registration QA returned 78/84 passing checks: all 42 affine pairs
and 36 SyN pairs. Six SyN cases exceeded the fixed 0.1 mm MRI-point roundtrip
threshold:

| Reference -> test | Reported maximum, mm |
|---|---:|
| sub-02 -> sub-05 | 0.100252 |
| sub-03 -> sub-02 | 0.106751 |
| sub-04 -> sub-01 | 0.121385 |
| sub-06 -> sub-01 | 0.125324 |
| sub-06 -> sub-04 | 0.116618 |
| sub-07 -> sub-03 | 0.105505 |

These six reported failures reached the roundtrip check after the preceding
grid/image checks; they are not reports of missing files or new inverse failures.
The current QA code does not retain partial statistics on a failed row, so use
the new diagnostic helper before drawing conclusions about the distribution.

```bash
micromamba run -n localizemi-registration .venv/bin/python \
  cross_subject_warp/scripts/06_diagnose_registration_roundtrips.py
```

The helper is checked but not run on subject registrations. It restricts work to
failed mode/pairs in the parent report. Before any ANTs point application, it
verifies relevant resources, current MRI files, pinned QA/mapping code and the
executable against the original QA source hashes. It repeats both directions
with the same seed, foreground samples and threshold, reporting count above
0.1 mm, median/p95/p99/max, destination coverage and per-point locations.

Output: `outputs/06_roundtrip_diagnostics/` (new directory required):
`01_direction_summary.csv`, `02_point_diagnostics.csv`,
`03_source_provenance.json`, `04_diagnostic_manifest.json`.
Exit 0 means diagnostics were written, **not** that registration QA passed;
exit 2 means a diagnostic direction encountered an execution error. Original
QA results/flags remain unchanged. Thresholds are not relaxed, samples are not
selected using localization outcomes, and no registration/inverse fitting occurs.

Compilation/help and four new synthetic tests passed for tail-count statistics,
explicit out-of-domain samples, invalid returned points and overwrite refusal.
Inspect these reports to separate a small tail of points from broader direction
or coverage problems. Do not silently remove outliers or accept failed pairs in
the primary evaluation. Any different acceptance threshold would require a
separately labelled sensitivity analysis with preserved primary results.


### Script 06 mapped evaluation — ready for the user to run

The user-run diagnostics found one foreground point above 0.1 mm in one
direction for each of the six flagged SyN pairs. The primary QA threshold and
those pair failure flags remain unchanged. The agreed evaluation policy is:
all 42 affine and the 36 passing SyN pairs are eligible; failed mode/pairs remain
scheduled rows with blank errors. Per-source-point QA can exclude further rows.

```bash
micromamba run -n localizemi-registration .venv/bin/python \
  cross_subject_warp/scripts/06_evaluate_localization.py --phase evaluate \
  --preparation-root cross_subject_warp/outputs/01_prepare_experiment_review \
  --output-root cross_subject_warp/outputs/06_evaluate_localization_mapped
```

The public Script 06 delegates mapped evaluation to `06_mapped_evaluation.py`.
Neither has been run on experiment inputs. Compilation/help and 52 synthetic
tests passed. New tests cover true test-frame errors, baseline nonduplication,
retained reconstruction/pair/point failures, no mapping calls for failed pairs,
field-domain checks, deduplication across primary/sensitivity, and matched-key
comparisons. No thresholds were adjusted and no fitted transforms/peaks changed.

Evaluation first checks completed table hashes, strict selection manifests,
pinned source/metric code, the seven bridge reports, pair QA metrics and all
current QA input checksums. Approximately 18 GB of resource checksums are
verified at start/end. Ground truth is loaded from frozen stimulation metadata
functions and native test contact tables only; raw EEG arrays are not loaded.
Contact/metadata hashes must match native completion provenance.

Passing native reference peaks are transformed from recorded head mm to
reference scanner RAS using the derived bridge, then mapped using the pinned
ANTs point sequences and RAS/LPS conversion. Required reference/test MRI and
SyN field domains must contain the points with the declared interpolation
margin. Every mapped source must roundtrip within 0.1 mm. Points are batched by
pair/mode and duplicate coordinates are mapped once. No source is reselected
or amplitude field resampled after mapping.

Ground truth is the native test contact midpoint, transformed only through
its own surface-to-scanner bridge. Cross errors are distances in test scanner
RAS mm; within errors are distances in test head mm. The within baseline occurs
once, labelled registration_mode `within`. Affine and SyN share native fits.
Both scenarios retain their original peak policies and reconstruction status.

Expected output is 10,980 rows per scenario: 732 within rows + 5,124 cross rows
for each registration mode. Source `status`/`failure_reason` are preserved;
use `evaluation_status` and `localization_error_valid` to identify valid errors.
Failure categories are `RECONSTRUCTION_FAIL`, `PAIR_QA_FAIL`, `POINT_DOMAIN_FAIL`,
`POINT_ROUNDTRIP_FAIL` and `MAPPING_EXECUTION_FAIL`; their errors are blank.
These rows remain in summary denominators. `PASS_WITHIN` and `PASS_MAPPED`
contribute errors. Pair-QA flags are retained even on reconstruction failures.

Artifacts 01–08 retain planned names: input validation, coordinate validation,
primary localization, sensitivity localization, method summary, failure analysis,
evaluation manifest and raw head diagnostic. Additional artifacts are
`09_registration_readiness.csv`, `10_retained_pair_qa_failures.csv`,
`11_source_provenance.json`, `12_matched_affine_syn.csv`, and
`13_source_point_mapping_audit.csv`. Raw unregistered head separation is clearly
diagnostic and is never used as cross anatomical error. The matched table uses
exact strategy/scenario/method/reference/test/run/condition keys and requires
both modes to have valid errors; marginal coverage is reported separately.

Exit 0 means every planned row was processed, with expected failures retained.
Exit 2 means reports include mapping execution errors that need inspection.
Invalid inputs/checksums stop before publication. Existing output folders are
refused. The historical fit-time MRI/ANTs provenance caveats remain attached;
numerical mapping QA is not a claim of cortical registration accuracy or solver
convergence. No inverse fitting or registration fitting occurs.

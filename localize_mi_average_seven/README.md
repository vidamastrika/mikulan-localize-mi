# Average-seven versus fsaverage: no-warp comparison

**Within-cohort design:** all seven test MRIs contribute to one group-average
T1. This is not held-out anatomy. Baseline commit:
`11f6008805d82756d18806293189e7a435122bef`; branch:
`experiment/localize-mi-average-seven`.

Groupwise MRI registration is used **only to construct the average T1**.
Localization uses rigid EEG-to-template coregistration in each arm, without MRI
warping or scaling. Evaluation directly compares model-HEAD estimates with the
original stimulation midpoint in test-HEAD coordinates, matching the existing
no-warp/fsaverage diagnostic convention. No source or reference point is warped
back to a participant MRI. Construction transforms are saved for provenance only.

These are **unregistered diagnostic distances**, not validated anatomical
localization errors. Every output records `localization_error_valid=False`.
A smaller distance does not establish better anatomical accuracy.

## Terminal inputs

From repository root:

```bash
source .venv/bin/activate
python localize_mi_average_seven/scripts/01_prepare_experiment.py
python localize_mi_average_seven/scripts/02_build_group_template.py
python localize_mi_average_seven/scripts/03_prepare_bem_source.py
python localize_mi_average_seven/scripts/04_plan_comparison.py
python localize_mi_average_seven/scripts/check_scaffold.py
```

All stages accept `--config PATH` and `--output-root NEW_DIRECTORY` and refuse
existing directories. Existing experiment results are preserved.

- **01:** Audit seven original `sub-XX/anat/sub-XX_T1w.nii` headers, qform/sform,
  millimetre units and hashes. Snapshot frozen settings and schedule. All seven
  passed lightweight geometry checks; anatomical image review is still required.
- **02:** Generate a manually executable ANTs groupwise SyN construction command
  (`outputs/02_build_group_template/01_build_template.sh`) and construction-only
  provenance manifest. Review stage 01 first; preserve input order, final template,
  deformation fields, affines and logs. This command is not run by the scaffold.
- **03:** Generate FreeSurfer/template segmentation, three-layer BEM and ico5
  source-space plans. Validate closed nested surfaces, intersections, normals,
  source containment, metre units and scanner/surface RAS bridges. Require identical
  rigid fiducial initialization and trimmed ICP in both arms: 40 iterations,
  15% trimming, median <=10 mm and p95 <=20 mm plus visual review. Failures block
  forwards/localization; never replace failed template anatomy with native anatomy.
- **04:** Generate 10,248 planned arm jobs from the frozen 5,124-job/61-run schedule,
  historical alignment audit, fsaverage reuse audit and explicit execution contracts.
  Keep sub-03 scheduled; its historical `FAIL_NUMERICAL` alignment is recorded.
  New alignment failure remains FAIL rather than causing subject deletion.

**Execution adapters are now available:** see [docs/RUN.md](docs/RUN.md) for
terminal inputs, dependencies, fiducial review and both-arm commands. The
numbered 02–04 scripts above remain safe plan generators. The new execution
entry points require `--execute` for template construction, segmentation, rigid
coregistration, forward builds and inverse fits. `04c_reconstruct.py` defaults to
ledger planning; `04d_export_evaluate.py` only exports and evaluates existing results.
No construction or inverse jobs were run during implementation.

## Matched protocol and reuse

Reuse frozen selections/schedules and the no-warp solver dispatch, plus the
fsaverage coregistration/forward validation functions, with explicit anatomy
paths. Do not invoke existing hardcoded CLIs against their old output folders.
Only anatomy changes. Both arms must share recorded participant geometry from
native forward FIFs, reference-based nominal montage, bad-channel replacement,
ordered actual channels, epoch data, average-reference scope, baseline/covariance
intervals, inverse settings and the known-stimulation-hemisphere peak rule.
Disclose that this peak rule uses known hemisphere information. Continuous-ECD
must receive its arm's BEM and matching rigid transform. No parameter retuning.

Use the recorded geometry embedded in validated forwards. Released electrode TSV
coordinates are identical across participants and are provenance-audit inputs.
The original stimulation midpoint is converted from native surface to test HEAD
using its audited rigid H5, as in `cross_subject_fsaverage/scripts/07_evaluate_localization.py`.
That rigid frame conversion is not MRI registration. Never mix metres and mm,
HEAD and surface RAS, or substitute a nearest-source/template-projected reference.

Existing fsaverage diagnostic estimates match the endpoint in principle. Reuse
still requires per-job matching configuration, data/preprocessing, electrode
geometry, channel/reference and coregistration fingerprints, matching peak policy,
and reviewed rigid alignment. The reuse audit currently denies approval because
several fingerprints are missing. `localization_error_valid=False` is expected
for this design and is **not** itself a reason to reject diagnostic results.
Unregistered fsaverage diagnostics must never be relabeled as anatomical errors.

## Evaluation terminal input and CSV contract

**Do not run stage 05 with placeholder filenames.** The results CSV does not exist until
both arms have run via the execution adapters. The audited reference
CSV must also be supplied. Use `04d_export_evaluate.py` to export both numerical inputs and evaluate them.
Planning stages do not create either numerical input.
The uppercase filenames below are placeholders, not files shipped with this
repository. Replace both with real existing paths when those inputs are ready:

```bash
python localize_mi_average_seven/scripts/05_evaluate_no_warp.py \
  --results TWO_ARM_HEAD_RESULTS.csv --references ORIGINAL_TEST_HEAD_REFERENCES.csv
```

Results require `strategy,method,reference_subject,test_subject,run,arm,status,
failure_reason,alignment_status,coordinate_frame,coordinate_units,
estimated_head_x_mm,estimated_head_y_mm,estimated_head_z_mm`, plus the protocol
fingerprints listed in the stage 04 contract. Frame must be `model_head`, units
`mm`; arm is `average_seven` or `fsaverage`, and status is PASS/FAIL.

References require `test_subject,run,coordinate_frame,coordinate_units,
gt_head_x_mm,gt_head_y_mm,gt_head_z_mm,provenance`. Frame must be `test_head`, units
`mm`. Provenance must identify the original midpoint and audited rigid conversion.
No MRI mapping status, registration warp or native-scanner coordinate is accepted
in place of this contract.

Stage 05 joins the full planned universe, rejects duplicate/extra keys, retains
raw PASS/FAIL and alignment failures, marks missing jobs FAIL, and outputs:

- `01_all_outcomes.csv`: diagnostic distances and failure reasons.
- `02_paired_differences.csv`: `diagnostic_difference_mm = average_seven - fsaverage`,
  only when both succeed and protocol fingerprints match.
- `03_pass_fail_counts.csv`: coverage including sub-03.
- `04_manifest.json`: diagnostic interpretation, inputs and provenance.

Positive difference means a larger average-seven **diagnostic** distance.
Lightweight checks exercise the schedule, known distances, unit/frame rejection,
protocol mismatch, failure retention, diagnostic labels and no-overwrite protection.

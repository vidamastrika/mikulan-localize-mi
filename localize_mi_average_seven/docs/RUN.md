# Executing the no-warp experiment

Run from repository root, on `experiment/localize-mi-average-seven`. Nothing below
has been launched during implementation. These commands are intentionally explicit;
inspect logs and review gates between stages. Every new output requires a fresh
directory, except reconstruction's explicit `--resume`.

## Environment and inputs

```bash
source .venv/bin/activate
command -v antsMultivariateTemplateConstruction2.sh
command -v recon-all
python localize_mi_average_seven/scripts/check_scaffold.py
```

Install/configure ANTs and FreeSurfer externally if missing, and source FreeSurfer's
`SetUpFreeSurfer.sh`; a FreeSurfer license may be required. MNE, scipy, numpy,
pandas, nibabel and h5py are needed. SESAME/Champagne additionally require the
same sesameeg/invertmeeg packages as the frozen solver. Missing method packages
are recorded as solver failures, not silently substituted methods.

Default fsaverage location: `~/mne_data/MNE-fsaverage-data/fsaverage`. Pass
`--fsaverage-root PATH` at the fsaverage coregistration stage if different. The
seven MRI/EEG inputs and frozen solver/parameter snapshots must remain available.
No downloads, data reconstruction or dependency installation is automatic.

## 1. Audit and construct the average T1

If stage 01 was already run successfully, use its existing output; do not rerun
against an occupied directory.
The saved configuration must match the current no-warp config. If an earlier
audit used an older config, create a fresh audit with `--output-root NEW_PATH`,
then pass that same path as `--preparation-root NEW_PATH` to template construction
and both reconstruction invocations.

```bash
python localize_mi_average_seven/scripts/01_prepare_experiment.py
python localize_mi_average_seven/scripts/02_execute_template.py --execute \
  --output-root localize_mi_average_seven/outputs/02_template
```

The second command checks the passing stage 01 audit and every MRI hash before
launching ANTs iterative groupwise construction. Review its log and final template.
The usual final ANTs filename is `group_template0.nii.gz`; verify that filename
and image coverage before passing it to segmentation. ANTs intermediate transforms
are construction provenance only and never applied to source/reference points.
The wrapper must also report `status: COMPLETE` in `04_execution_manifest.json`.
ANTs creates an initial average with the same filename before registration; a
file left by a failed run must not be used for segmentation. The generated
command uses `-m CC` for the metric; `-s` is ANTs' smoothing option. On failure,
keep the run directory, choose a fresh directory for retry and use that retry's
template path in the segmentation command below.

## 2. Segment template and build anatomy

```bash
python localize_mi_average_seven/scripts/03_execute_anatomy.py --execute \
  --phase segment \
  --template localize_mi_average_seven/outputs/02_template/group_template0.nii.gz \
  --output-root localize_mi_average_seven/outputs/03_segmentation

python localize_mi_average_seven/scripts/03_execute_anatomy.py --execute \
  --phase bem --output-root localize_mi_average_seven/outputs/03_bem
```

FreeSurfer writes only to `localize_mi_average_seven/work/freesurfer/average_seven`.
The BEM phase requires `recon-all.done`; it computes watershed surfaces, a
three-layer BEM and ico5 sources. Mesh closure/degeneracy and MNE's BEM validation
are checked. Review nesting, intersections, source containment and segmentation
in a viewer; automatic topology checks do not establish anatomical validity.

Mark and save template NAS/LPA/RPA fiducials in **MRI surface-RAS metres** as:

```text
localize_mi_average_seven/work/freesurfer/average_seven/bem/average_seven-fiducials.fif
```

For example, open the MNE coregistration/fiducial GUI in an interactive Python
session with `mne.gui.coregistration(subject='average_seven',
subjects_dir='localize_mi_average_seven/work/freesurfer')`. Inspect the landmarks
and save the fiducials; these cannot be invented from unreviewed template geometry.

## 3. Rigid EEG coregistration in both arms

```bash
python localize_mi_average_seven/scripts/04a_coregister.py --execute --arm average_seven
python localize_mi_average_seven/scripts/04a_coregister.py --execute --arm fsaverage
```

Both arms use the same three-fiducial HEAD-to-MRI initialization followed by the
baseline's rigid trimmed ICP, same thresholds and recorded native forward sensor
coordinates. There is no anatomy scaling or MRI registration during comparison.
Every participant, including sub-03, gets a numerical status and saved transform.

For **each arm**, inspect sensor overlays with `mne.viz.plot_alignment` using its
native forward Info, saved transform and arm anatomy. Open:

```text
localize_mi_average_seven/outputs/04a_coregister/average_seven/03_review_REQUIRED.json
localize_mi_average_seven/outputs/04a_coregister/fsaverage/03_review_REQUIRED.json
```

After independent visual review, change `anatomy_review` to `PASS` and each
reviewed, numerically eligible participant to `PASS` under `subjects`. Leave
failed/uncertain subjects pending or FAIL. The review binds to the exact hashed
coregistration manifest; it cannot override a numerical alignment failure.
The old sub-03 failure remains historical evidence, but its new arm-specific
alignment is tested anew; a failing new alignment is never reconstructed.

## 4. Build forwards

```bash
python localize_mi_average_seven/scripts/04b_make_forward.py --execute --arm average_seven
python localize_mi_average_seven/scripts/04b_make_forward.py --execute --arm fsaverage
```

Both use EEG only, `mindist=0`, and identical unchanged sensor coordinates and
channel order. Outputs include all seven subject statuses. Failed alignment,
unreviewed subjects and failed forward calculations remain FAIL in the planned
jobs. Old fsaverage forwards are not automatically reused: the matched fiducial
initialization and reviewed protocol are now regenerated in both arms.

## 5. Plan and run frozen inverse jobs

Planning creates full ledgers and verifies frozen code, grids, data and anatomy;
it hashes EEG files but does not run inverse fits:

```bash
python localize_mi_average_seven/scripts/04c_reconstruct.py --arm average_seven
python localize_mi_average_seven/scripts/04c_reconstruct.py --arm fsaverage
```

Pilot one LOSO fit per arm:

```bash
python localize_mi_average_seven/scripts/04c_reconstruct.py --execute --resume \
  --arm average_seven --strategy LOSO --method MNE --test-subject sub-01 --run run-01 --limit 1
python localize_mi_average_seven/scripts/04c_reconstruct.py --execute --resume \
  --arm fsaverage --strategy LOSO --method MNE --test-subject sub-01 --run run-01 --limit 1
```

Complete the remaining LOSO fits:

```bash
python localize_mi_average_seven/scripts/04c_reconstruct.py --execute --resume --arm average_seven --strategy LOSO
python localize_mi_average_seven/scripts/04c_reconstruct.py --execute --resume --arm fsaverage --strategy LOSO
```

The LOSO filter selects 732 scheduled jobs per arm across 61 runs and 12
methods. Single-reference rows stay pending in the full ledger. In the
exported evaluation, select LOSO rows, since pending rows are exported as
`NOT_RUN` failures. LOSO holds the test subject out of parameter selection.
The shared Maskface template includes all seven subjects' MRIs.

Frozen no-warp Engine.prepare/solve performs all 12 methods, reference montage
selection/replacement, original average-reference ordering, covariance and target
window processing and peak selection. Only forward/BEM lookup is injected.
Continuous-ECD receives its arm's BEM and forward transform. All settings resolve
from the frozen selections/grid snapshots. Completed PASS/FAIL records are never
retried by resume. Inputs, forward hashes and package versions must still match.

## 6. Generate real evaluation inputs and evaluate

```bash
python localize_mi_average_seven/scripts/04d_export_evaluate.py \
  --output-root localize_mi_average_seven/outputs/04d_export_evaluate
```

This verifies both run manifests and exports:

```text
outputs/04d_export_evaluate/01_two_arm_head_results.csv
outputs/04d_export_evaluate/02_original_test_head_references.csv
outputs/04d_export_evaluate/05_evaluate_no_warp/01_all_outcomes.csv
outputs/04d_export_evaluate/05_evaluate_no_warp/02_paired_differences.csv
```

The references reuse the baseline audited original surface-to-test-HEAD midpoint
table. Coverage, stimulation pair and hemisphere must match. No point is warped.
Pending jobs export as FAIL/NOT_RUN while retaining `solver_status=PENDING`.
Individual failures remain in coverage denominators. Pairing requires both PASS
and identical settings, actual channel lists, preprocessing/data, reference,
electrode geometry and coregistration-protocol fingerprints. Distances remain
unregistered HEAD diagnostics with `localization_error_valid=False` throughout.

Only lightweight checks have been run by the implementation agent. Heavy-stage
commands and their outputs require validation on your ANTs/FreeSurfer/MNE setup;
GUI anatomy/fiducial/coregistration review remains a required human step.

Adapter checks (also no heavy computations):

```bash
python localize_mi_average_seven/scripts/check_execution.py
```

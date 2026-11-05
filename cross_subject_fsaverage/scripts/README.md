# Fsaverage scripts

The approved sequence is `01_prepare_experiment.py`, `02_coregister_eeg.py`,
`03_register_ground_truth.py`, `04_make_forward.py`, `05_reconstruct.py`,
`06_peak_sensitivity.py`, `07_evaluate_localization.py`,
`08_compare_no_warp.py`, and `09_visualize_results.py`. Scripts 01–03 are
implemented. Later scripts and their numbered output directories will be
created after each preceding stage has been run and reviewed.

From the repository root, with the existing virtual environment:

```bash
.venv/bin/python cross_subject_fsaverage/scripts/01_prepare_experiment.py \
  --output-root cross_subject_fsaverage/outputs/01_prepare_experiment
```

`--fsaverage-root` defaults to
`~/mne_data/MNE-fsaverage-data/fsaverage`; override it if the subject files are
elsewhere. Script 01 does not download data, perform registration, construct a
forward model, or fit an inverse. It refuses an existing numbered output folder.
The `outputs/` tree is ignored by Git; keep its manifest and frozen copies with
your local experiment record.

Script 02 fits candidate rigid transforms from each subject's native forward
sensor coordinates to the fsaverage scalp. The released EEG electrode TSVs are
identical across subjects and are used only to audit the discrepancy. Run:

```bash
.venv/bin/python cross_subject_fsaverage/scripts/02_coregister_eeg.py \
  --output-root cross_subject_fsaverage/outputs/02_coregister_eeg
```

The default input is Script 01's numbered output. Its per-subject transforms
are candidates: `PASS_NUMERICAL_REVIEW_REQUIRED` does not establish anatomical
accuracy. Review the sensor-to-scalp distances and mapped electrode locations
before using a transform in a forward model.

Script 03 independently registers each subject's native MRI to fsaverage for
stimulation ground-truth mapping. It does **not** clear Script 02's failed
sub-03 EEG coregistration. First run its lightweight input audit:

```bash
.venv/bin/python cross_subject_fsaverage/scripts/03_register_ground_truth.py \
  --phase audit \
  --output-root cross_subject_fsaverage/outputs/03_register_ground_truth_audit
```

Review `01_input_inventory.csv` and the manifest. Then, only in a new output
directory, run the full registration locally:

```bash
.venv/bin/python cross_subject_fsaverage/scripts/03_register_ground_truth.py \
  --phase register \
  --output-root cross_subject_fsaverage/outputs/03_register_ground_truth
```

The register phase uses the local ANTs environment at
`~/micromamba/envs/localizemi-registration/bin` by default; supply `--ants-bin`
if it is elsewhere. This stage fits seven new native-to-fsaverage MRI
registrations. The earlier native-to-native affine/SyN transforms target
different subjects and cannot be reused as fsaverage registrations. Numerical
point roundtrips and image outputs require anatomical review before Script 04.

Script 03 now stages temporary whitespace-free aliases for ANTs inputs and
outputs, because the installed shell wrapper loses spaces in MRI paths.
Each registration writes `command.json` with the original paths and wrapper
arguments. Missing or empty required outputs are reported explicitly even
when the wrapper exits zero. Preserve the existing failed registration folder;
use a new output directory for a repaired run:

```bash
.venv/bin/python cross_subject_fsaverage/scripts/03_register_ground_truth.py \
  --phase register \
  --output-root cross_subject_fsaverage/outputs/03_register_ground_truth_pathfix
```

## Current simplified workflow: Script 04

MRI registration in Script 03 is optional and is not required for this workflow.
`04_make_forward.py` uses the frozen Script 01 schedules/configurations and the
existing Script 02 candidate transforms. Sub-03 stays blocked. Numerical PASS
for other subjects does not establish independently verified anatomical accuracy.
No thresholds, electrode positions, or historical outcomes are changed.

Audit (lightweight; no forward construction):

```bash
.venv/bin/python cross_subject_fsaverage/scripts/04_make_forward.py \
  --phase audit \
  --output-root cross_subject_fsaverage/outputs/04_make_forward_audit
```

Build (execute locally after reviewing audit):

```bash
.venv/bin/python cross_subject_fsaverage/scripts/04_make_forward.py \
  --phase build \
  --output-root cross_subject_fsaverage/outputs/04_make_forward
```

Both commands refuse existing output directories. Optional overrides:
`--preparation-root`, `--coregistration-root`, `--fsaverage-root`, `--output-root`,
`--n-jobs` (default 1). HEAD sensor coordinates are retained unchanged; MNE uses
the saved rigid HEAD-to-fsaverage MRI transform once. The fsaverage source/BEM
files must match Script 01 hashes. The leadfield uses all 256 channels; later
reconstruction applies the unchanged reference-based montage/subset policy.
`mindist=0` mm is explicit MNE default behavior, matching the original forward
builder's omitted argument; no source-distance threshold is introduced.

Outputs are `01_subject_forward_status.csv`, the exact copied
`02_frozen_reconstruction_plan.csv`, `03_job_forward_availability.csv`,
`04_forward_manifest.json`, frozen `source_configurations/`, and (build only)
`forwards/sub-XX_fsaverage-fwd.fif`. Status CSVs checkpoint each subject.
Forward build failures remain in the full ledger; successful forward PASS means
numerical construction only, not inverse completion or anatomical validation.

## Script 05: reuse the no-warp reconstruction engine

`05_reconstruct.py` subclasses the unchanged no-warp `Engine`. Its preparation
adapter substitutes a test-specific fsaverage forward lookup while leaving the
native reference lookup for montage selection untouched. The inherited ECD
branch receives the fsaverage BEM and the forward's MRI/head transform. All 12
method branches, covariance/target windows, averaging, and primary peak policy
remain in the existing engine. No MRI registration, affine/SyN mapping or
localization-distance calculation runs here.

Prepare and inspect the full ledger (no EEG loading/inverse):

```bash
.venv/bin/python cross_subject_fsaverage/scripts/05_reconstruct.py --phase plan
```

Run one MNE pilot locally, resuming that plan:

```bash
.venv/bin/python cross_subject_fsaverage/scripts/05_reconstruct.py \
  --phase reconstruct --resume --strategy LOSO --method MNE \
  --test-subject sub-01 --run run-01 --limit 1 --quiet
```

After inspecting the pilot status and channel audit, continue locally:

```bash
.venv/bin/python cross_subject_fsaverage/scripts/05_reconstruct.py \
  --phase reconstruct --resume --quiet
```

Default output: `cross_subject_fsaverage/outputs/05_reconstruct`. Use
`--output-root` for a new experiment directory. `--resume` requires unchanged
input/code hashes and package versions and never retries PASS/FAIL/BLOCKED rows.
Only PENDING cases are run; filters do not remove cases from the full ledger.
Sub-03 remains blocked even if a forward is later added to the folder.
The frozen configuration owner is the test subject for LOSO and reference
subject for Single-reference. Reference sub-03 may still supply a historical
parameter/channel-selection identity for other test subjects.

Outputs include `01_execution_status.csv` (all 5,124 cases),
`02_reconstruction_plan.csv`, `05_resolved_configurations.json`,
`06_frozen_run_manifest.json`, `07_execution_summary.csv`,
`09_channel_selection_audit.csv`, and failure traces when needed. Each attempt
records runtime and successful fits retain actual EEG/time hashes, montage
choices and peak coordinates. PASS indicates solver completion only. No
anatomical localization-error validity is asserted. Existing baseline, no-warp,
affine and SyN records remain unchanged.

## SESAME capacity compatibility continuation (Script 05b)

The stopped original run remains immutable. `05b_reconstruct_sesame_compat.py`
is a separately versioned continuation of Script 05, using the same no-warp
Engine. `sesame_capacity_compat.py` changes only the Euclidean neighbourhood
capacity guard from 100 to 256 in process memory, without editing the installed
library. The automatic 10 mm radius, connectivity/minimum checks, neighbour
ordering and transition-probability construction remain unchanged.

Preflight imports the original full checkpoint into a fresh directory. It
retains successful records and unrelated failures, and marks only the original
SESAME maximum-neighbour failures as pending retries. Original status, failure
reason and runtime are preserved in added provenance fields and
`08_import_audit.csv`. Sub-03 remains blocked. Parent inputs, code and package
versions must still match their frozen manifest. `--import-status` defaults to
`outputs/05_reconstruct/01_execution_status.csv`; keep this parent unchanged.

Run preflight:

```bash
.venv/bin/python cross_subject_fsaverage/scripts/05b_reconstruct_sesame_compat.py --phase plan
```

Then run ONLY one SESAME pilot:

```bash
.venv/bin/python cross_subject_fsaverage/scripts/05b_reconstruct_sesame_compat.py \
  --phase reconstruct --resume --strategy LOSO --method SESAME \
  --test-subject sub-01 --run run-01 --limit 1 --quiet
```

New output root: `cross_subject_fsaverage/outputs/05_reconstruct_sesame_compat`.
Inspect that pilot before running any full continuation. A complete successful
SESAME fit is not established by capacity/geometry tests. The dense source
neighbour matrix remains large and the first pilot may take time. No coarser
source space or automatic radius adjustment is introduced.

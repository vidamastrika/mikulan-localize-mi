# 09 Comparative review and three-arm alignment

Read the [comparative report](../outputs/09_comparative_anatomy/review_run_01/REPORT.md)
and [model summary](../outputs/09_comparative_anatomy/review_run_01/01_model_summary.csv).
Keep anatomical approval pending until an explicit reviewer decision.

## After anatomical approval

Follow the [stage 08 anatomical approval instructions](../08_anatomical_review_bem/README.md#anatomical-approval-gate)
to create and complete the separate approval form against stage 08 review run 01.
Include the stage 09 observations in the rationale. Use the commands below
from the repository root. The BEM command requires the approved form.

```bash
export MPLCONFIGDIR=/tmp/maskface-surface-diagnostics-matplotlib
base=localize_mi_average_seven/maskface_template
out08="$base/outputs/08_anatomical_review_bem"
out09="$base/outputs/09_comparative_anatomy"
bem="$out08/bem_run_01"

.venv/bin/python "$base/08_anatomical_review_bem/02_build_bem.py" \
  --review-root "$out08/review_run_01" \
  --approval "$out08/anatomy_approval_run_01.json" \
  --output-root "$bem" --execute
```

Inspect `01_downsampled_geometry.json` for ico=4 topology, nesting and
intersections, `02_source_containment.json` for the ico5 active cortical
vertices, and `03_solution_validation.json` for the three conductivities
(0.3, 0.006 and 0.3 S/m), compartments and FIF round trips. Proceed only
with a COMPLETE manifest. If an output directory already exists, use a fresh
number and update the variables accordingly.

## Fiducials for the two average-seven arms

Mark, save, and review LPA, nasion and RPA separately in each template's own
MRI surface RAS. Use the existing fsaverage MRI fiducials for the third arm.
Open the MNE GUIs and save outside the existing subject and BEM directories.

```bash
pydeface=localize_mi_average_seven/work/freesurfer/average_seven
pyfid="$out09/pydeface-reviewed-fiducials.fif"
mffid="$out09/maskface-reviewed-fiducials.fif"
fid_review="$out09/fiducial_approval_run_01.json"

.venv/bin/python - "$pydeface" "$bem" "$pyfid" "$mffid" <<'PY'
import json, sys
from pathlib import Path
import mne
old, bem, old_fid, new_fid = map(Path, sys.argv[1:])
m = json.loads((bem / 'manifest.json').read_text())
if m['status'] != 'COMPLETE':
    raise SystemExit('BEM validation is incomplete')
if old_fid.exists() or new_fid.exists():
    raise SystemExit('Choose fresh fiducial filenames or reuse the reviewed files')
mne.viz.set_3d_backend('pyvistaqt')
for subject, subjects_dir, destination in [
    (old.name, old.parent, old_fid),
    (m['subject'], m['subjects_dir'], new_fid),
]:
    print(f'Save the reviewed landmarks for {subject} to {destination.resolve()}', flush=True)
    mne.gui.coregistration(subject=subject, subjects_dir=subjects_dir,
                           head_high_res=False, block=True)
PY
```

After both files have been independently reviewed, record their hashes and
the decision. The command refuses to overwrite an existing record.

```bash
.venv/bin/python - "$pydeface" "$bem" "$pyfid" "$mffid" "$fid_review" <<'PY'
import hashlib, json, sys
from pathlib import Path
old, bem, old_fid, new_fid, record = map(Path, sys.argv[1:])
def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()
manifest = bem / 'manifest.json'
m = json.loads(manifest.read_text())
if m['status'] != 'COMPLETE':
    raise SystemExit('BEM validation is incomplete')
with open('/dev/tty', 'r+') as tty:
    tty.write('Reviewer of both saved MRI fiducials: ')
    tty.flush()
    reviewer = tty.readline().strip()
    tty.write('Type APPROVED after checking all three landmarks on each template: ')
    tty.flush()
    decision = tty.readline().strip()
if not reviewer or decision != 'APPROVED':
    raise SystemExit('Fiducial review remains pending')
paths = {
    'pydeface_average_seven': (old / 'mri/T1.mgz', old / 'bem/outer_skin.surf', old_fid),
    'maskface_average_seven': (Path(m['subjects_dir']) / m['subject'] / 'mri/T1.mgz', Path(m['scalp']), new_fid),
}
arms = {
    arm: dict(t1_sha256=sha(t1), scalp_sha256=sha(scalp), fiducials_sha256=sha(fid))
    for arm, (t1, scalp, fid) in paths.items()
}
with record.open('x') as f:
    json.dump(dict(decision=decision, reviewer=reviewer, bem_manifest_sha256=sha(manifest), arms=arms), f, indent=2)
    f.write('\n')
PY
```

## Three-arm rigid comparison

```bash
.venv/bin/python "$base/09_comparative_anatomy/01_compare_rigid_alignment.py" \
  --bem-root "$bem" --pydeface-fiducials "$pyfid" --maskface-fiducials "$mffid" \
  --fiducial-review "$fid_review" --output-root "$out09/alignment_run_01" --execute
```

Each participant's 256 original HEAD-coordinate EEG points are loaded once
from the released native forward FIF and reused for every arm. ICP uses up to
40 iterations and 15% trimming, with rigid rotations and translations only.
The gate uses distances for all 256 channels, with median ≤10 mm and p95 ≤20 mm.
Inspect `01_alignment.csv` for per-arm results, `02_three_arm_comparison.csv`
for paired results, and `03_visual_review_pending.json` and `transforms/` for
visual sensor-to-scalp review. Numerical failures remain failures. The
comparison ends at rigid alignment.

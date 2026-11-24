# 08 Anatomical review, BEM and rigid alignment

The original maskface watershed surfaces from FreeSurfer run 07 are the
provisional candidate. The completed review package is
[review_run_01](../outputs/08_anatomical_review_bem/review_run_01/REPORT.md).
Open its report, slice positions, and regional comparison CSV. The figures
use the same head-relative slice positions for maskface, fsaverage and
sub-01 through sub-07. Inspect the vault, face/orbits, inferior skull base
and both lateral views. Matching the offsets supports visual comparison
without establishing voxelwise anatomical correspondence.

All commands below use Bash from the repository root.

```bash
export MPLCONFIGDIR=/tmp/maskface-surface-diagnostics-matplotlib
code=localize_mi_average_seven/maskface_template/08_anatomical_review_bem
out=localize_mi_average_seven/maskface_template/outputs/08_anatomical_review_bem
review="$out/review_run_01"
bem="$out/bem_run_01"
```

To regenerate the review package, choose a fresh review directory first.

```bash
.venv/bin/python "$code/01_prepare_review.py" --output-root "$out/review_run_02"
```

## Anatomical approval gate

Review the scalp and both skull contours against the MRI. Require plausible
layer correspondence in the vault, face/orbits and inferior skull base.
Consider whether the outer skull extends into facial or inferior signal and
whether the surface RAS conversion produces plausible layer placement.
The reference meshes and their geometry flags provide context. Vertex gaps
are geometric distances. If any required region remains uncertain, keep the
decision pending and stop here.

The following command makes a separate pending decision form and inserts the
completed review manifest hash. It refuses to overwrite an existing form.

```bash
approval="$out/anatomy_approval_run_01.json"
.venv/bin/python - "$review" "$approval" <<'PY'
import hashlib, json, sys
from pathlib import Path
review, output = map(Path, sys.argv[1:])
manifest = (review / 'manifest.json').read_bytes()
assert json.loads(manifest)['status'] == 'COMPLETE'
record = json.loads((review / 'APPROVAL_PENDING.json').read_text())
record['review_manifest_sha256'] = hashlib.sha256(manifest).hexdigest()
with output.open('x') as f:
    json.dump(record, f, indent=2)
    f.write('\n')
PY
```

After explicit anatomical approval, the reviewer fills in their name, date
and rationale, including observations at the face/orbits and skull base.
Set each of the four regional decisions to `ACCEPTABLE_FOR_EXPLORATORY_BEM`
and the overall decision to `APPROVED_FOR_BEM_CONSTRUCTION` only if the
observed correspondence supports that decision. Leave the review package and
all hashes intact. The BEM script verifies the exact approved review and
the T1 and surface hashes before calling MNE BEM construction.

## BEM and cortical sources

```bash
.venv/bin/python "$code/02_build_bem.py" --review-root "$review" \
  --approval "$approval" --output-root "$bem" --execute
```

To test numerical feasibility while anatomical review is pending, use
`--exploratory` in place of `--approval "$approval"` and choose a fresh
output directory. The build verifies the completed review and its input
hashes, records the pending review in the manifest, and applies the same
numerical validation.

This uses the unmodified watershed vertices in FreeSurfer surface RAS.
The MNE model uses 0.3, 0.006 and 0.3 S/m for brain, skull and scalp and
ico=4. The downsampled meshes must pass topology, outward orientation,
self-contact, inter-layer contact and vertex nesting checks. All active
ico5 cortical source vertices must lie inside the downsampled inner skull.
The BEM solution and source space must pass FIF round-trip checks.
A failure stops progression and preserves diagnostic files. Inspect
`01_downsampled_geometry.json`, `02_source_containment.json` and
`03_solution_validation.json`. Use a fresh output number for any retry.

## Rigid alignment after BEM and fiducial review

After the BEM manifest reports COMPLETE, open the MNE GUI on the staged
subject. Use manually reviewed LPA, nasion and RPA. The MRI fiducial frame
must be surface RAS in metres. Keep scaling disabled. Save the fiducials to
the path printed below, outside the completed BEM directory.

```bash
fid="$out/maskface-run-01-fiducials.fif"
.venv/bin/python - "$bem" "$fid" <<'PY'
import json, sys
from pathlib import Path
import mne
bem, fid = map(Path, sys.argv[1:])
m = json.loads((bem / 'manifest.json').read_text())
assert m['status'] == 'COMPLETE'
print(f'Save reviewed MRI fiducials to {fid.resolve()}', flush=True)
mne.viz.set_3d_backend('pyvistaqt')
mne.gui.coregistration(subject=m['subject'], subjects_dir=m['subjects_dir'],
                        head_high_res=False, block=True)
PY
```

Once the saved landmarks have been independently reviewed, create a
fiducial decision tied to the saved coordinates and completed BEM run.

```bash
fid_review="$out/fiducial_approval_run_01.json"
.venv/bin/python - "$bem" "$fid" "$fid_review" <<'PY'
import hashlib, json, sys
from pathlib import Path
bem, fid, dest = map(Path, sys.argv[1:])
with open('/dev/tty', 'r+') as tty:
    tty.write('Reviewer of the saved MRI fiducials: ')
    tty.flush()
    reviewer = tty.readline().strip()
    tty.write('Type APPROVED after reviewing all three landmarks: ')
    tty.flush()
    decision = tty.readline().strip()
if not reviewer or decision != 'APPROVED':
    raise SystemExit('Fiducial decision remains pending')
with dest.open('x') as f:
    json.dump(dict(decision='APPROVED', reviewer=reviewer,
                   fiducials_sha256=hashlib.sha256(fid.read_bytes()).hexdigest(),
                   bem_manifest_sha256=hashlib.sha256((bem / 'manifest.json').read_bytes()).hexdigest()),
              f, indent=2)
    f.write('\n')
PY

.venv/bin/python "$code/03_compare_rigid_alignment.py" --bem-root "$bem" \
  --fiducials "$fid" --fiducial-review "$fid_review" \
  --output-root "$out/alignment_run_01" --execute
```

Both arms use the ordered geometry of the original 256 EEG channels from
each participant's released native forward FIF, including sub-03. They
reuse the frozen three-fiducial initialization and rigid trimmed ICP
with up to 40 iterations, 15% trimming and nearest-vertex acceptance
thresholds of median ≤10 mm and p95 ≤20 mm. Inspect
`01_paired_alignment.csv`, each arm's saved transforms and
`03_review_REQUIRED.json`. Visual overlay review remains required for
numerical passes. This stage ends at rigid alignment.

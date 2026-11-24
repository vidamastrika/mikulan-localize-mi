# BEM repair checkpoint — 2026-11-16

## Local repair candidate: numerical checks pass, visual review pending

### Candidate fiducial placement

The GUI's scalp/MRI/initial-fiducial inputs have been checked using
`mne.coreg.Coregistration`. For manual placement in the user's terminal:

```bash
MNE_LOGGING_LEVEL=ERROR MPLCONFIGDIR=/tmp/mpl-bem QT_API=pyside6 \
  .venv/bin/python localize_mi_average_seven/scripts/03g_place_candidate_fiducials.py --open
```

Initial landmarks may be automatically estimated from fsaverage; they must be
inspected and manually placed on the template. Unlock fiducials, place
LPA/Nasion/RPA, and use **Save MRI Fid.** Save in the candidate subject's BEM
directory as `average_seven-fiducials.fif`. Do not enable anatomy scaling or run
EEG alignment in this placement session. The candidate's anatomy approval and
experiment promotion remain pending; saving a fiducial file alone is not
anatomical approval. No heavy EEG jobs have been launched.

Interactive review from the repository root (no sudo):

```bash
MNE_LOGGING_LEVEL=ERROR MPLCONFIGDIR=/tmp/mpl-bem QT_API=pyside6 \
  .venv/bin/python localize_mi_average_seven/scripts/03f_review_anatomy.py
```

Top row is the unrepaired atlas candidate; bottom row is the local repair.
Select Repair, Pial outlier or Whole head; sliders scroll the three voxel axes.
Red is inner skull, yellow outer skull, cyan skin, green pial cortex, and the
magenta cross identifies the pre-existing pial outlier when near a viewed slice.
Inspect several slices around each region and the overall segmentation fit.
This viewer never adopts anatomy or changes approval status. Static images are
in `outputs/03f_anatomy_review/`. It uses actual triangle-plane intersections in
FreeSurfer surface-RAS-to-voxel coordinates. Static export and a Qt offscreen
smoke check were run; interactive display must be opened in the user's terminal.

`work/bem_candidates/local_inner_1mm` preserves atlas surfaces except for 15
inner-skull vertices. Nearest-triangle projection places the local inner-skull
patch inward of the outer skull with a nominal 1-mm clearance and a cosine
taper within 15 mm of the audited outside vertices. Maximum movement relative
to atlas candidate: 2.210 mm. Vertex counts and triangles are unchanged. No
original anatomy or downstream manifest has been replaced.

The initial plan to expand outer skull was rejected after measuring only
0.839 mm to skin at one protruding inner-skull point. Inner-skull protrusion
relative to outer skull reached 1.210 mm. The inward repair follows the local
projection concept in MNE's surface-fixing tutorial, but its geometry parameters
are this experiment's explicit candidate choices, not an anatomically validated
prescription: https://mne.tools/stable/auto_tutorials/forward/80_fix_bem_in_blender.html

Checks completed:

- Closed/manifold mesh, no degenerate triangles or reversed affected face normals.
- Full-resolution adjacent-shell vertex containment: zero outside vertices.
- VTK adjacent-shell triangle collision tests: zero contacts.
- Changed inner-skull faces versus nonadjacent faces: zero intersections; pairs
  sharing vertices are excluded from this particular local test.
- MNE ico4 three-layer geometry and BEM solution: PASS.
- Candidate ico5 sources generated: 10,242 per hemisphere; all inside the
  full-resolution repaired inner skull.

The full pial check found one right pial vertex (index 73565) outside the inner
skull by 0.102 mm at surface RAS [38.476, 25.836, -0.052] mm. This is remote from
the repair and already outside both the atlas and original surfaces. It remains
an anatomical-review issue; source containment passing does not erase it.

Inspect `outputs/03b_nesting_local_inner_1mm/02_sagittal.png` and its axial and
coronal companions. Candidate provenance, movements, BEM checks, collision
checks and cortical/source checks are saved beside the candidate. Visual review
of anatomy and landmarks is still required by `docs/RUN.md`; candidate reports
retain `anatomy_approved=false`. After review, promotion must keep surfaces,
BEM/source files, hashes and anatomy paths consistent, then proceed to manually
reviewed fiducials and both-arm rigid coregistration. Do not rerun the original
watershed stage over this repair.

Atlas-mode validation: watershed completed, but MNE ico4 BEM geometry still
fails inner-skull containment. Full-resolution checks find 9/10,242 inner-skull
vertices outside the outer skull (original: 10); outer skull has zero vertices
outside skin. The remaining crossing is in the same anterior midline region.
Atlas mode changed the inner-skull and brain meshes but left outer skull and
skin geometry unchanged. No solution was built or candidate adopted. See
`work/bem_candidates/atlas_mode/validation.json` and
`outputs/03b_nesting_atlas_candidate/` for MRI views and vertex audit.
Next work should assess the local skull boundaries against MRI and choose an
anatomically justified segmentation/mesh correction, followed by full geometry,
intersection and BEM-solution checks; do not delete the offending vertices.

Update: the user ran the T1-mode candidate. Watershed finished successfully,
but all three BEM meshes have exactly identical vertices and triangles to the
original. MNE ico4 geometry validation still fails inner-skull containment.
The candidate was staged separately with MRI volume metadata restored; see
`work/bem_candidates/t1_mode/validation.json`. No BEM solution was generated.
The original surfaces remain unchanged. T1 mode alone does not repair this case.
The next candidate can explicitly enable `-atlas` (original log reports atlas
mode off); validate it independently and do not treat successful segmentation
as anatomical approval.

The completed template is `outputs/02_template_retry1/group_template0.nii.gz`.
The initial `02_template` run failed and must not be used. FreeSurfer completed
on November 13. MNE currently rejects the BEM because inner skull is not fully
inside outer skull. A full-resolution vertex audit found 10/10,242 inner-skull
vertices outside outer skull, near anterior midline, surface RAS bounds
[-3.64, 55.30, 7.95] to [7.01, 65.23, 13.25] mm.

The read-only audit script `scripts/03b_audit_bem_nesting.py` saves vertex indices,
coordinates, both adjacent-shell containment checks and focused MRI plots. Output
directories must be fresh. Passing vertex checks alone does not establish absence
of triangle intersections or anatomical validity.

Docker access was attempted again November 16. The socket denies access and
`sudo -n docker` requires a password. No candidate regeneration has run, and no
original surface has changed.

## Separate candidate generation

Once Docker authentication is available, run from the repository root in an
interactive terminal. The following candidate explicitly enables watershed's
T1 flag, which the original MNE call left disabled. This is a candidate for MRI
review, not an established fix. Keep the original surfaces for comparison.

```bash
repo="$PWD"
candidate="$repo/localize_mi_average_seven/work/bem_candidates/t1_mode"
subject="$repo/localize_mi_average_seven/work/freesurfer/average_seven"
mkdir -p "$repo/localize_mi_average_seven/work/bem_candidates"
mkdir "$candidate"
sudo docker run --rm --user "$(id -u):$(id -g)" \
  --volume "$repo:$repo:ro" --volume "$candidate:$candidate" \
  --volume /home/girvi949/.freesurfer/license.txt:/license.txt:ro \
  --env FS_LICENSE=/license.txt \
  --entrypoint mri_watershed freesurfer/freesurfer:8.2.0 \
  -T1 -useSRAS -surf "$candidate/average_seven" \
  "$subject/mri/T1.mgz" "$candidate/ws.mgz" \
  > "$candidate/watershed.log" 2>&1
```

Inspect the log and all resulting shells against the MRI. Before adopting any
candidate, update its surface volume metadata as MNE's watershed wrapper does,
check mesh topology, nested surfaces and intersections, and run MNE's ico4 BEM
geometry and solution validation. Do not move vertices merely to silence the
nesting error. Preserve candidate commands, logs and hashes. The downstream
anatomy manifest, source space and visual review are still outstanding.

# 07 Surface diagnostics

This follows [Stage 2 surface QA run 10](../outputs/06_surface_QA/run_10/REPORT_stage2.md),
using ANTs construction run 05 and FreeSurfer run 07. Stage 2 compared the watershed
vertices directly with scanner-RAS MRI. Run 07 explicitly logged surface RAS for
`-useSRAS`. Use the corrected overlays from this diagnostic for anatomical interpretation.
The rigid frame conversion preserves the previously reported nearest-vertex distances.

## Manual execution

Run from the repository root, in a foreground terminal so Docker can prompt for
sudo credentials. Each command creates a fresh output directory. The pinned
FreeSurfer 8.2.0 image must already be available locally. Candidate generation
runs only watershed on the completed FreeSurfer T1. Each candidate retains its
command, log, surfaces and manifest with hashes.

```bash
export MPLCONFIGDIR=/tmp/maskface-surface-diagnostics-matplotlib
diag=localize_mi_average_seven/maskface_template/07_surface_diagnostics/diagnose.py
out=localize_mi_average_seven/maskface_template/outputs/07_surface_diagnostics

for setting in baseline_replay t1 atlas; do
  sudo -v || break
  .venv/bin/python "$diag" candidate --setting "$setting" \
    --output-root "$out/candidates/${setting}_run_01" \
    --runtime docker --docker-command 'sudo docker' \
    --image freesurfer/freesurfer:8.2.0 \
    --license /home/girvi949/.freesurfer/license.txt
done

.venv/bin/python "$diag" review --output-root "$out/run_01" \
  --candidate-root "$out/candidates/baseline_replay_run_01" \
  --candidate-root "$out/candidates/t1_run_01" \
  --candidate-root "$out/candidates/atlas_run_01"
```

A failed candidate with a manifest can be included in review and will appear in
the failure CSV. If a command never created a candidate directory, omit its
`--candidate-root`. To review the baseline and references alone, omit all three
candidate arguments. Use a fresh output number for any retry. Run 10 and all
other upstream manifests and hashes are verified before processing.

The replay checks reproducibility on the same image and build. `-T1` explicitly
requests normalized T1 processing. `-atlas` enables atlas analysis. Each setting
changes one option relative to the replay. FreeSurfer's implementation derives
the outer skull by smoothing and shrinking the fitted outer skin. Accordingly,
an option that changes brain segmentation may leave the facial outer-skull
contour substantially unchanged. Treat this as an empirical question and
inspect all three layers. [FreeSurfer watershed implementation](https://raw.githubusercontent.com/freesurfer/freesurfer/dev/mri_watershed/mri_watershed.cpp)

## Review

The four focused views for each head contain axial, coronal and sagittal slices
on that head's MRI. The hotspot view centers all three planes on the outer-skull
vertex with the largest outer-to-inner gap. Also inspect the directed inner-to-outer
tail to reproduce the interpretation of Stage 2. Identify the anatomical sites
of excursions and record whether they follow visible cranial boundaries or
continue along facial and inferior signal. Compare skull and scalp separately.

The four exclusive regional bins use scanner RAS relative to the median
inner-skull vertex. Maskface settings share the baseline anchor. At relative
S ≥ 15 mm use cranial vault. Below that, A ≥ 25 mm is facial. Among the remaining
vertices, S < −25 mm is inferior head, and the rest are cranial transition.
Left and right sectors start beyond ±20 mm. These are spatial proxies with
identical rules across models. Differences in anatomy, head orientation, mesh
density and coverage affect the comparisons. Empty bins have no measurement.

The compressed per-vertex CSV supplies both endpoints and directions. The
comparison CSV gives regional median, p95, maximum, sample count and global-tail
count for every model. The tail CSV breaks the upper five percent down by
source region, lateral sector and target region. Nearest-vertex gaps are geometry
proxies. They do not measure skull thickness.

Compare the paired group overlays and normalized intensity cue CSV with the
same-participant Pydeface/maskface pairs. Check whether facial or inferior
departures coincide with added signal while comparing with the vault control.
Paired signal measurements cannot establish which voxels are synthetic or
establish causation. Group templates were constructed independently. A tissue
provenance mask or a matched input perturbation would be needed for a stronger
attribution.

Inspect the topology, same-layer and inter-layer triangle screens, and vertex
containment for each candidate. The VTK screens can report unresolved checks.
The generated REPORT.md records outcomes and leaves anatomical review pending.
Evaluate tissue correspondence before using any candidate for BEM. Candidate
generation and review do not create a BEM solution, forward, or inverse.

# Scalp geometry investigation — manual execution

From the repository root:

```bash
cd /home/girvi949/projects/mikulan-localize-mi
.venv/bin/python localize_mi_average_seven/diagnostics/02_scalp_geometry/diagnose.py \
  --output-root localize_mi_average_seven/diagnostics/02_scalp_geometry/run_01
```

Uses the existing virtual environment: NumPy, SciPy, nibabel, MNE, pandas, Matplotlib. No downloads or external registration commands are required. The script verifies the completed `01_eeg_alignment/run_01` input/output hashes and the template-construction input hashes. It refuses any existing output directory; use `run_02` for a retry. It prints progress and leaves manifest status RUNNING if interrupted, COMPLETE only after all outputs and final input-hash checks succeed.

The exact geometry and alternative dense scalp calculations can take substantial time. Images are at least 180 DPI, default 220, with large panels, physical-mm axes, limited electrode labels and separate target views. The script never applies head scaling, refits transforms, changes thresholds, alters anatomical files, or calculates forward/inverse models. Diagnostic image sampling uses trilinear interpolation in memory and does not write resampled MRI volumes.

Optional synthetic checks (no participant data or MNE import):

```bash
.venv/bin/python localize_mi_average_seven/diagnostics/02_scalp_geometry/check_geometry.py
```

Paths can be overridden with `--previous`, `--dataset`, `--construction`, `--average-seven`, and `--fsaverage`. Required defaults are the existing completed template in `outputs/02_template_retry1`, the original FreeSurfer reconstruction in `work/freesurfer/average_seven`, the installed fsaverage, and the dataset symlink. The original scalp must reproduce the preceding run’s exact distances; the script stops if it differs. Alternative scalps remain diagnostic only.

If Matplotlib needs a writable cache, prefix the command with `MPLCONFIGDIR=/tmp/average-seven-matplotlib`. The user’s normal local terminal should have access to their MNE config; no MNE config or environment setting is modified by this script.

## Outputs

| File | Purpose |
|---|---|
| `01_construction_log_evidence.csv` | Exact numbered interpolation, averaging, reference-space, transform and cropping-related log excerpts |
| `02_mri_geometry.csv` | Inputs, retained construction stages, FS import/T1, fsaverage; affines, voxel dimensions, FOV and intensity-support bounds |
| `03_registration_transforms.csv` | ANTs affine parameters and displacement-field summaries; coarse physical field-only Jacobians |
| `04_scalp_geometry_topology.csv` | Dimensions, area, volume, closure, winding consistency, components and topology |
| `05_regional_coverage.csv` | Area-weighted sectors/Z bands, MRI boundary support, scalp proximity to image FOV |
| `06_surface_sections.csv` | Exact triangle-plane contour spans and lengths |
| `07_mri_radial_edge_cues.csv` | Radial MRI intensity-drop candidates near 1024 sampled scalp vertices per primary model |
| `08_electrode_signed_distances.csv` | All 256 electrodes × seven participants × three models; exact distances, conditional sign, 3D closest points and MRI-FOV status |
| `09_alternative_scalps.csv` | Available watershed/candidate/head-mesh comparisons and targeted alternative sensor distances/signs |
| `10_fiducial_MRI_audit.csv` | Fiducial locations, explicit hypotheses, scalp distances and MRI samples |
| `11_residual_MRI_paths.csv` | Full 3D MRI intensity profiles along problematic residual vectors, including e238–e252 |
| `12_targeted_figure_index.csv` | Which electrode is selected for each primary residual figure |
| `13_inside_outside_counts.csv` | Per-participant/model conditional inside/outside/unresolved counts |
| `14_radial_edge_summary.csv` | Regional MRI edge-cue summaries |
| `15_regional_planarity_cues.csv` | Area-weighted normal coherence, axis alignment and patch extents |
| `16_FreeSurfer_processing_evidence.csv` | Recorded conformation/resampling/import evidence from recon-all logs/commands |
| `coordinate_bridges.json` | Explicit surface-RAS → scanner-RAS matrices and native volume-header checks |
| `construction_provenance.json`, `averaging_check.json` | Construction command, provenance and descriptive retained-warped-mean check |
| `manifest.json`, `REPORT.md` | Completion status, hashes, versions, measurements, limitations and review guidance |

High-resolution PNG groups:

- **All nine primary head models:** axial, anterior coronal and left sagittal MRI views with exact scalp intersection contours.
- **Construction versus segmentation:** original final template, FS imported MRI, processed T1, and retained registered-input mean, with the same scalp mapped using actual headers.
- **Seven retained registered inputs versus final template:** common physical slices at the largest sub-03 average-seven residual, separately for axial, coronal and sagittal views. The retained warped inputs may precede the final shape update; these are not an independent registration accuracy test.
- **Every participant/model:** three-plane MRI views at the worst e238–e252 residual and the worst residual outside that block. Average-seven also shows the worst problematic electrode on the original template MRI.
- **Alternative scalps:** primary and alternative contours on the original model MRI, with no replacement or fitting.
- **Fiducials:** MRI panels through each labelled landmark, keeping native sidecar interpretations separate and estimated template landmarks marked unreviewed.

## Interpretation rules

All surface coordinates are RAS mm: X right, Y anterior, Z superior. Native GIFTI VolGeom scanner geometry must match the released native MRI before overlays are allowed. Template bridges use the actual T1 affine and tkregister affine. ANTs parameters/vectors are LPS; warp vectors are converted explicitly for physical Jacobian sampling. No ANTs transformation is applied to electrodes.

Signed distances are positive outside and negative inside **relative to the estimated mesh**. A sign is withheld unless closed, connected, consistently wound, nondegenerate topology, winding-number classification and three independent oblique ray parities agree, and the point is more than 0.1 mm from the surface. These are numerical sign-reliability conditions, not altered alignment thresholds. Self-intersection absence is not fully certified; even reported signs are conditional and do not independently validate anatomy or digitization.

A displayed residual arrow is its projection into the plane through the electrode. The nearest point’s off-plane displacement is printed, and full 3D points/vectors are recoverable from CSVs. MRI edge cues, support bounds, planarity and regional areas are descriptive. They cannot independently certify segmentation, cropping, registration, or an invalid template.

Use MRI evidence to distinguish hypotheses: a missing boundary in the input/template suggests coverage or construction effects; an MRI boundary clearly extending beyond a scalp contour suggests segmentation mismatch; native electrode inconsistency remains possible where coordinate provenance is unresolved. Any causal conclusion should cite specific MRI slices and geometry evidence, rather than electrode distances alone.

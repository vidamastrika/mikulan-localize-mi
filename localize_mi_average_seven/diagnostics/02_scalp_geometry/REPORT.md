# Scalp geometry investigation — preparation report

Status: **PREPARED_NOT_EXECUTED**. Run the command in README.md to produce computed CSVs, high-resolution PNGs, a reproducibility manifest and `run_01/REPORT.md`. Previous outputs remain unchanged.

## Evidence inspected during preparation

The previous diagnostic manifest reports COMPLETE and the earlier review verified 174 input and 24 output hashes. The new runner rechecks them before analysis. Saved exact electrode-to-scalp distances must reproduce for all seven native heads and both template fits.

The completed construction manifest lists all seven T1w MRI hashes. Its shell command requests four iterations of `antsMultivariateTemplateConstruction2.sh` with SyN and CC. The construction log records creation of an initial population average, output physical space based on the largest image, linear interpolation and repeated template shape updates. These are confirmed logged operations, not proof of all upstream processing details. A complete ANTs script/version may not remain accessible; missing crop keywords do not prove absence of earlier cropping.

The FreeSurfer reconstruction retains its imported MRI (`mri/orig/001.mgz`) and processed T1 (`mri/T1.mgz`). These permit separate examination of the final group template, import/conformation and downstream scalp segmentation. Their actual headers will be audited; diagnostic overlays never rely on an assumed common voxel index.

Available scalp candidates include original watershed, atlas-mode, T1-mode, local inner-skull repair scalp, average-seven head FIF, and fsaverage head/dense/medium FIF representations. Availability or numerical agreement does not establish independent anatomical validity. No candidate is adopted.

Native scalp GIFTIs contain full FreeSurfer volume geometry despite unknown-to-unknown GIFTI coordinate-system labels. The inspected sub-01 header reconstructs the released T1w affine closely; the runner requires that agreement separately for all seven participants before allowing native MRI overlays. Original electrode provenance remains uncertain.

## Planned measurements and limitations

The script audits scalar image geometry, FOV, multiple intensity-support bounds, ANTs affine/warp records and coarse displacement-field Jacobians. It compares exact surface-plane contours, surface area by regional sector/Z band, dimensions, manifold topology, and planarity cues. Radial MRI intensity profiles provide candidate boundary offsets, with explicit warnings about competing tissue/masking edges.

All electrode signs are conditional geometric classifications of the estimated scalp. Closed consistent topology plus winding/ray agreement are required, and near-boundary or ambiguous points stay unsigned. Full self-intersection certification is outside this diagnostic; no anatomical correctness is inferred from sign alone.

Large readable MRI panels separate axial, coronal and sagittal views. Targeted residual figures retain full exact closest points and print projection offsets. Fiducial panels separate repeated/unverified sidecar interpretations from estimated/unreviewed template landmarks.

This preparation does not conclude that average-seven is anatomically invalid. Construction, segmentation, coverage and coordinate inconsistencies remain competing hypotheses until the generated MRI evidence is reviewed. No scalp/BEM edits, head scaling, threshold changes, refitting or forward/inverse calculations are performed.

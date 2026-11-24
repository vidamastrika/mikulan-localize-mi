# Review of completed run_02

The run is COMPLETE. Independently rechecked all 270 recorded input hashes and 101 output hashes: no mismatches. Outputs include 16 CSVs and 81 PNGs. This review leaves the completed run, its manifest and all earlier results unchanged.

## Main interpretation

**Anatomical coverage mismatch is supported as a contributor to anterior residuals. A new scalp-segmentation collapse or a failed MRI registration has not been established. Electrode conventions remain unresolved.**

In the reviewed average-seven original-template, FS-import and processed-T1 overviews, the scalp follows the visible external MRI boundary reasonably closely, including a sharply cut-away anterior/inferior facial region. This boundary is already present in the original final template; it is not introduced only by FreeSurfer import or the local inner-skull repair.

In the reviewed sub-03 native overview, the released MRI has removed facial tissue while the original native scalp extends into the missing-image region. At axial Z=-48.0 and -3.0 mm, the anterior native scalp is visibly beyond the cut-away MRI boundary; sagittal X=1.2 mm shows the same distinction. Thus the native and average-seven surfaces do not represent equivalent facial coverage. This is consistent with differing relationships to released MRI defacing, but exact upstream surface-generation history is not verified.

The average-seven original-template sagittal X=0.8 mm and axial Z=-30.8 mm panels show the corresponding cut-away region followed by the template scalp. The sub-05 average-seven e238 sagittal panel at X=44.7 mm shows a +41.2 mm residual anterior/inferior to the retained MRI/scalp boundary. These observations support a coverage mechanism for some anterior errors; they do not explain all lateral residuals or independently validate electrode positions.

## Confirmed geometry and containment results

All nine primary scalps have a single component, Euler characteristic 2, no boundary/nonmanifold edges, no inconsistent edge winding and no degenerate faces in the implemented audit. This rules out those measured topology failures, not anatomical truncation or all self-intersections. A surface can close smoothly across removed facial anatomy.

All 162 average-seven exact triangle residuals above 20 mm are classified OUTSIDE_CONDITIONAL. Per subject the counts are 20, 38, 48, 36, 12, 6 and 2. The previous nearest-vertex count for sub-02 was 39; one vertex distance above 20 mm becomes a triangle distance below 20 mm. This does not change the alignment gate or its outcome.

The largest residual is sub-03 e248: +60.30 mm, at surface RAS (-100.14, 83.93, -25.03) mm; its nearest scalp point is (-65.86, 34.74, -18.60) mm. These coordinates place the electrode beyond the retained anterior/left surface. They do not establish whether its true anatomical location is correct.

The native configurations themselves have 1,780/1,792 electrodes outside, seven inside and five unresolved. This broad positive offset means native surfaces cannot be treated as an independent zero-residual truth. Cap standoff, surface definition and coordinate provenance remain possible contributors.

All reported signs remain conditional on the documented topology/winding/ray checks. Full self-intersection absence and electrode anatomical provenance are not certified.

## Construction, FOV and alternative representations

Inputs are approximately 1 mm isotropic. The final template, retained registered inputs, intermediate templates and FS import/T1 have the same 256-cubed grid within the implemented affine tolerance. The average-seven scalp has no vertices within two voxels of the MRI image-box boundary; its minimum image-boundary margin is approximately 7.21 voxels. This argues against direct image-box clipping of that scalp, but does not exclude removal of anatomy inside the FOV or upstream masking.

All 14 audited forward/inverse displacement fields have positive sampled field-only Jacobians (minimum approximately 0.246). This is a coarse diagnostic, not full-resolution/composite registration certification. The normalized retained-input mean correlates 0.977 with the final template; that does not independently establish averaging correctness, particularly because final shape updates intervene.

The average-seven watershed, atlas-mode, T1-mode and repaired-candidate scalp arrays are identical. Its head FIF matches within about 0.000004 mm at the reported p95 vertex comparison. These are effectively the same scalp representation, not independent alternatives capable of resolving a segmentation hypothesis.

Fsaverage dense and medium alternatives have nonmanifold topology under the explicit audit, and the medium mesh also triggered MNE's reader warning. Their signs were withheld. The accepted primary fsaverage scalp has no corresponding measured topology defect; the alternate-mesh warning does not invalidate that primary scalp or change its historical 6/7 numerical result.

## Limitations of the generated figures

The construction input-comparison sagittal plane was selected through the worst electrode at X=-100.1 mm, outside the head. Its blank panels cannot assess registration quality. The corresponding coronal plane at Y=83.9 mm and the selected inferior Z range also misses the retained head. The axial comparison shows only a small edge of the head. These particular figures need scalp-centered or residual-aligned/oblique slices before making regional registration claims.

Electrode-centered target figures deliberately project nearest points into each plane and print their off-plane displacement. For e248, the coronal and sagittal nearest points are about 49.2 and 34.3 mm off-plane. Those panels illustrate the geometry but cannot demonstrate MRI boundary agreement at both endpoints. The whole-head overviews and near-coplanar sub-05 e238 sagittal view are more informative for the current coverage observation.

## Next discriminating evidence

Prepare additional original-template/registered-input slices through the nearest scalp point and the residual midpoint, plus a plane containing the full residual vector. Review anterior and left-lateral coverage separately, including high-residual native electrodes. This can clarify where MRI support ends without changing fitting, thresholds or anatomy.

Independent digitization/export records or expert verification of the native electrode convention are still needed to separate remaining electrode inconsistency from real anatomical/cap differences. No BEM, coregistration or localization approval follows from this review.

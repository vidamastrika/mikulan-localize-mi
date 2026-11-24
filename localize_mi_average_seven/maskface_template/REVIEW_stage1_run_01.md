# Review of Stage 1 input feasibility run_01

The run completed. Independently verified all 544 input hashes and 33 output hashes, with no mismatches. Eight CSVs and 21 PNGs are present. All seven MRI pairs have identical voxel grids/affines and finite data. Original files and Stage 1 outputs remain unchanged.

## Important diagnostic limitation

The two image versions have substantially different intensity conventions: maskface values are approximately 1/255 of the Pydeface values in shared brain controls. The original script used one raw-intensity display window, making maskface panels nearly black. Its raw absolute-difference maps and exact-equality fractions therefore cannot be interpreted as anatomical changes. Original layer curves also need the intensity convention accounted for before meaningful visual comparison.

Geometry, file integrity, relative-within-image support counts and Pearson correlations remain informative. Original raw difference/equality metrics are retained for reproducibility, but must not be used to infer brain damage or loss of tissue fidelity.

## Evidence from existing brain-control CSV samples

A descriptive least-squares relationship Pydeface ≈ gain × maskface + offset, using the already saved 512 brain-inward proxy samples per participant, gives:

| Subject | Gain | Offset in Pydeface units | Absolute residual p95 in Pydeface units |
|---|---:|---:|---:|
| sub-01 | 255.000 | -0.00004 | 0.00040 |
| sub-02 | 254.417 | 0.12957 | 0.30605 |
| sub-03 | 255.300 | -0.09865 | 0.26489 |
| sub-04 | 255.129 | -0.05563 | 0.29982 |
| sub-05 | 254.781 | 0.02822 | 0.30357 |
| sub-06 | 254.204 | 0.20585 | 0.35381 |
| sub-07 | 254.819 | 0.04130 | 0.23834 |

These are in-sample descriptive checks, not independent validation. Raw brain-control correlations range from approximately 0.999973 to effectively 1.0. This strongly supports an intensity-convention difference as the dominant reason for the large raw brain differences; it does not independently prove intact anatomy or identify the precise export operation.

## Corrective supplement prepared

Run from the repository root:

```bash
.venv/bin/python localize_mi_average_seven/maskface_template/scripts/01b_review_intensity_conventions.py \
  --output-root localize_mi_average_seven/maskface_template/outputs/01b_intensity_review/run_01
```

The supplement preserves the original run and creates a fresh output directory. It fits intensity gain/offset on alternating brain-control vertices and reports errors on held-out vertices, keeping both depths for a vertex together. It also reports a fixed-factor-255 comparison, inventories NIfTI storage slope/intercept, and generates corrected paired MRI figures, layer profiles and regional metrics.

This is an in-memory display/metric calibration only. It does not rescale head geometry, rewrite MRI voxels, choose anatomy from EEG outcomes, or produce construction inputs. Signal-support counts retain their original per-image relative cutoffs. There is no spatial registration, construction, segmentation or inverse computation.

Synthetic tests passed for known gain/offset recovery, held-out independence, immutable image adapters, adjusted regional metrics and corrected figure rendering. No full supplemental participant analysis was executed during this review.

## Feasibility decision

Technical input feasibility is supported for all seven images. Strong shared brain signal is encouraging for a separate exploratory construction, but the original shared-window images cannot complete the required anatomical review. Review the corrected supplement before deciding whether modified scalp/skull anatomy is suitable for the proposed variant.

Additional maskface signal remains unverified as original-versus-filtered tissue without an unmodified reference. FreeSurfer success, BEM boundary fidelity and source containment remain untested. Stage 2 remains unauthorized and has not been started.

# Stage 1 review after intensity-convention correction

## Review outcome

**The inputs are technically feasible and sufficiently promising for a separate exploratory Stage 2 construction trial, with mandatory anatomical QA. They are not validated scalp/skull anatomy or an accepted head model.** This is a recommendation after reviewing Stage 1, not automatic execution or authorization of later computations. Stage 2 has not been started, and all existing stage-gate files remain unchanged.

Verified the supplemental run's 579 input hashes and 26 output hashes: no mismatches. It completed with four CSVs and 21 PNGs. The original Stage 1 run and its outputs are preserved. All 336 regional signal-support counts reproduce exactly between the original and supplemental tables.

## Confirmed findings

All seven paired MRIs passed the original technical checks: matching voxel grids/affines and finite image data. The approximately 255-fold intensity-convention difference explains why the original common-window maskface figures were black and why raw absolute differences were misleading. It is an intensity relationship, not a spatial scale difference.

The supplement fits gain/offset to 256 inward-brain control samples and validates on 256 samples from different native reference vertices for each participant. Held-out p95 absolute differences are normalized by each participant's Pydeface positive-intensity p99:

| Subject | Held-out p95 difference (% of Pydeface p99) |
|---|---:|
| sub-01 | 0.00027% |
| sub-02 | 0.2068% |
| sub-03 | 0.1543% |
| sub-04 | 0.1572% |
| sub-05 | 0.1709% |
| sub-06 | 0.2360% |
| sub-07 | 0.1413% |

Held-out correlations exceed 0.99997. The fixed factor-255 diagnostic gives similarly small errors and does not require a subject-specific fitted spatial transform. Gains/offsets are for review only; none were written into MRI files or selected as construction preprocessing.

NIfTI storage conventions also differ: maskface inputs are float32 with dataobj slope 1; several Pydeface inputs are uint8 with nonunit slopes. Nibabel already applies those stored slopes. This is not a reason to multiply stored slopes a second time, and the exact upstream export operation is not established from the fitted gains.

## What the corrected anatomy comparisons show

Reviewed the corrected cranial overview contact sheet for all seven participants and detailed facial/cranial panels for sub-03, with additional facial panels for sub-05 and sub-07. The sampled deep brain appearance and internal structures agree closely. The additional maskface facial/inferior structures are visible where Pydeface removed them, while superficial boundary differences remain.

For sampled inward cranial brain and inner-skull profiles, the largest participant/region p95 difference is approximately 0.244% of Pydeface p99. This supports retained common interior signal at those reference locations. These are correlated anatomical proxy samples, not an independently defined whole-brain mask or a full-volume tissue-preservation guarantee.

Common cranial signal has regional p95 differences up to approximately 0.35% of p99 at the 5% support cutoff. However, this aggregate can hide minority regions with larger differences. Near-surface cranial outer-skull profiles reach approximately 14.7% and near-surface scalp profiles 11.9% at the reported p95, using original native reference surfaces. Thus the masking effect is not confined to an unimportant removed face, and deep-brain agreement cannot certify scalp/skull fidelity.

Additional facial support remains unverified as original versus filtered/modified boundary anatomy. The release method citation indicates intentional surface-layer modification, but does not establish the exact variant, parameters or affected voxels. Original native surfaces provide a useful reference; they are not independently validated segmentations of these maskface images.

## Implications for the experiment

1. Retain original average-seven/full-montage results and all failed alignments as the baseline.
2. A maskface-only template and FreeSurfer reconstruction are reasonable next experimental tests in this separate variant. They would test processing feasibility and resulting geometry, rather than assume an anatomically superior model.
3. Keep the raw maskface files pinned. Do not silently apply the fitted diagnostic intensity calibration to construction inputs. Any later input-intensity handling must be explicit and checked at MRI import; it is distinct from prohibited spatial head scaling.
4. Inspect the resulting scalp, skull and brain boundaries against the actual maskface template, with regional thickness, nesting, intersection and source-containment checks before considering BEM acceptance. Superficial changes and modified anterior geometry require particular attention.
5. Do not expect improved facial coverage to resolve sub-03's preserved left-cranial mismatch. That issue predates this candidate variant and no new alignment has yet been computed.
6. FreeSurfer success, usable three-layer segmentation and unchanged-threshold alignment remain untested. Better signal coverage alone is not a feasibility pass for those later stages.

No template, reconstruction, surface, registration, head scaling, forward model or inverse localization was produced during this review. Stage 2 remains unstarted.

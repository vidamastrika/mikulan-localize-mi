# Review of anatomical coverage run_01

Status: COMPLETE. Independently checked all 436 recorded input hashes and 101 output hashes: no mismatches. The run contains 13 CSVs and 83 PNGs. Original full-montage statistics were reproduced. This review does not change any completed output, manifest, surface, sensor set or threshold.

## Conclusion

**Upstream anonymization contributes to the anterior coverage problem, but facial removal does not explain all alignment failures. A supported left-cranial discrepancy persists, especially for sub-03.** Retain current anatomy/full-montage results as the baseline. Plan any reduced-montage or alternative-anatomy work as a separate, explicitly labelled sensitivity experiment, without choosing a channel subset because it produces more numerical passes.

## Origin and alternative released MRI

All seven T1w sidecars document Pydeface, the exporter copies T1_pydefaced.nii into the template input paths, and the release README distinguishes original-MRI surface reconstructions from anonymized MRIs. This establishes that facial removal predates averaging; it is not caused solely by template construction.

The sub-03 anonymization comparison visibly shows cut-away facial anatomy in the Pydeface input and more image support in the existing maskface representation, against the same original native scalp. All 34 channel patches classified LOW_MRI_SUPPORT_PROXY in the Pydeface MRIs become MRI_SUPPORTED_PROXY in maskface. An additional 75 uncertain patches become supported; 287 remain uncertain. This makes the existing maskface representation worth assessing before considering a new anatomical construction. It is still anonymized/modified anatomy, not an unmodified anatomical ground truth, and no substitution is approved or performed.

Construction progression and retained registered-input residual-aligned panels now contain substantial MRI anatomy. They support inspection of the boundary at each stage, but earlier stages and retained warped inputs are not guaranteed anatomically identical to the final shape-updated template. Additional effects of registration/averaging on coverage are not independently quantified.

## Regional contribution

Facial plus inferior-lateral geometric proxies account for approximately 74.8%, 71.0%, 74.7%, 83.3%, 72.1%, 60.0% and 61.0% of average-seven squared triangle residuals for sub-01 through sub-07. These are geometric regions, not validated physical sensor-anatomy labels; the e238–e252 block must not be classified uniformly from numbering alone.

MRI support at inward probes is not proof that the superficial scalp is intact. Native electrode/scalp correspondence remains uncertain. In particular, substantial numbers of cranial-proxy patches are uncertain for sub-05 and sub-07, which strongly reduces the common all-participant channel intersection.

## Fixed-transform sensitivity

The table counts participant configurations meeting the original median ≤10 mm and p95 ≤20 mm numerical limits on each stated subset. Only the first row is the preserved full-montage outcome. Subset rows are not newly accepted alignments; no transform was refitted.

| Subset | Channels | Average-seven | fsaverage |
|---|---:|---:|---:|
| Original full montage | 256 | 3/7 | 6/7 |
| Participant-specific MRI-supported | 142–235 | 3/7 | 7/7 |
| Participant-specific MRI-supported cranial | 63–150 | 6/7 | 7/7 |
| Common MRI-supported across all seven | 60 | 3/7 | 6/7 |
| Common MRI-supported cranial across all seven | 27 | 7/7 | 7/7 |

The 27-channel result uses only about 10.5% of the original montage. Its labels are common, but physical correspondence and representative spatial coverage are not established. It cannot justify selecting that subset as a new primary montage or claiming that alignment is solved. The broader common 60-channel result retains the original difference. These are fixed-transform statistics and do not predict performance after refitting, forward calculation or inverse localization.

## Independent left-cranial finding

For sub-03, the same 42 left-cranial MRI-supported channels have nearest-vertex p95 distances of 22.34 mm in the native model, 18.41 mm with fsaverage and 26.81 mm with average-seven. The corresponding right-cranial average-seven p95 is 18.04 mm over 41 channels. The supported-left average-seven e62 figure shows an approximately 31.6 mm residual beyond a visibly retained MRI/scalp boundary, rather than an absent facial region.

This supports a discrepancy beyond facial removal. Its presence in the native configuration also prevents attributing it solely to group-template construction. Head/cap geometry, original surface definitions and unresolved native electrode conventions remain competing explanations. Other participants do not show a uniform left-worse-than-right pattern.

## Figure and BEM review

All 126 full-residual oblique panels contain both endpoints within numerical tolerance (maximum off-plane error approximately 8e-15 mm). No target plane was flagged for negligible MRI support; the minimum positive-image fraction is about 20%. These figures resolve the prior blank-plane limitation. The third panel remains an axial slice through the closest scalp point, and explicitly omits off-plane electrodes.

The average-seven layered figure shows both scalp and outer-skull boundaries following the cut-away anterior region. Brain/inner-skull contours are available for review, but these selected sections cannot certify intact compartments, all cortical coverage, nesting or anatomical suitability of the three-layer BEM. Dark skull or CSF cannot be interpreted as missing anatomy from intensity probes alone. The repair candidate remains unapproved.

The repair comparison CSV compares the initial/original inner skull against the final candidate; it is not a measurement of the local repair operation alone. The candidate's documented repair baseline is the atlas-mode inner skull. Widespread original-to-candidate differences therefore must not be described as widespread modifications caused by the local repair.

## Recommendation

1. Preserve the current template and all full-montage failures as the baseline; the evidence does not establish global anatomical invalidity.
2. Review the supported-left channels and their native/average-seven MRI geometry, including acquisition/export coordinate provenance, before changing anatomy to improve fit.
3. Assess the existing maskface MRIs for faithful cranial/skull coverage as a possible separate construction input. Their improved facial support does not automatically establish anatomical correctness or justify immediate template rebuilding.
4. A separate reduced-montage sensitivity experiment is defensible only with independently reviewed coverage labels, spatial coverage assessment and a frozen common channel rule in both arms. Do not choose 27 channels solely because 7/7 meet the limits.
5. A common HydroCel template montage remains feasible only as a separate declared model assumption with one pinned resource, a justified physical scale convention and equivalent rigid landmark placement. Physical eN-to-EN mapping remains unverified; neither scaling nor projection onto the cut-away face was performed here.

No new anatomy, BEM, forward calculation or inverse localization is approved by this review.

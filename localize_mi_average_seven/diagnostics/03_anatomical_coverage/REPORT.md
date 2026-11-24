# Anatomical coverage — preparation report

Status: **PREPARED_NOT_EXECUTED**. The script will write measured results and recommendations to a fresh `run_01/REPORT.md`. Existing diagnostics and their reports remain unchanged.

## Evidence read before preparation

Read `02_scalp_geometry/REVIEW_run_02.md`, corresponding MRI-coordinate and residual output schemas, and the earlier full-montage metrics. The preceding run established consistent geometry hashes, a cut-away average-seven facial boundary, native surfaces extending into absent released MRI facial anatomy, and unresolved left-lateral errors. Its electrode-centered construction panels cannot establish regional registration quality where slices lie outside the head.

A documentary link now establishes the upstream origin more directly. The native MRI sidecars identify Pydeface. `_08_bids_conversion.py` copies `T1_pydefaced.nii` to each released `<subject>_T1w.nii`, and the completed construction command uses those T1w paths. The dataset README states that the released BEM/pial/inflated surfaces were created with original MRIs, while the released MRIs are anonymized. Thus missing facial anatomy is documented before averaging; template construction is not its sole origin. How much additional coverage loss registration, averaging or segmentation introduces is not independently quantified.

The release also contains existing maskface MRI alternatives. The diagnostic compares these as already available representations, without reconstructing faces, substituting inputs or constructing a new template.

## Prepared analysis

Native closest-scalp locations and MRI intensity probes define fixed anatomical/coverage proxies independently of template residual size. The script identifies facial, inferior-lateral, cranial and uncertain channel locations; measures regional residual contribution; and computes fixed-transform supported-subset statistics while preserving original full-montage outcomes and thresholds. It also exports common subsets requiring consensus across all seven participants.

Supported left/right cranial tests and nearest mirrored native-scalp pairs examine lateral mismatch separately from facial coverage. These are descriptive comparisons, not validated physical sensor pairings or a new rigid fit.

New oblique planes contain the full residual segment. Scalp-centered anatomical views and improved construction-stage panels expose image support rather than repeating the blank electrode-centered planes. Layered MRI overlays and regional probe summaries examine BEM-relevant skull and brain segmentation boundaries, without computing a BEM or adopting the existing repair candidate. Low MRI signal at skull/CSF is explicitly not treated as proof of damage.

## Montage feasibility and recommendation scope

The existing `numbered_hydrocel_v1` decision explicitly records assumed/unverified eN-to-EN correspondence and MNE's fixed 95 mm normalization. The new diagnostic assesses a common HydroCel montage procedure without placing sensors or applying normalization/scaling. A separate experiment would need one pinned resource, a declared physical scale convention, identically defined rigid landmark placement and explicit treatment of unsupported facial sensors in both arms. Earlier assumed correspondence is not promoted to verified acquisition identity.

Retain current anatomy and full-montage results as the baseline. A separate reduced-montage sensitivity experiment is reasonable to plan only with explicit provisional coverage definitions, equivalent channel sets and independent review; its primary scientific interpretation must not replace the existing experiment. Alternative anatomy would be a separate investigation, not an unreviewed repair of removed faces. Final recommendations will be informed by the generated supported-left and BEM-boundary evidence.

## Validation

Synthetic checks passed for anatomical proxy rules, MRI-support ambiguity, inward patch geometry, orthonormal/copanar residual planes, geometric left/right pairing and empty/nonfinite metric handling. A synthetic MRI/mesh plotting check also passed and verified recorded endpoint coplanarity. No participant dataset analysis was executed during preparation.

# Progress update: average-seven versus fsaverage experiment

Date: 16 November 2026

## Objective

Compare EEG source localization using an MRI template constructed from the
seven Localize-MI participants with localization using fsaverage. Both arms
are intended to use matched EEG processing and inverse settings, with rigid
EEG-to-template alignment and no scaling or anatomical warping during
localization. All seven participants contribute to the group template, so this
is a within-cohort comparison, not a held-out anatomy evaluation.

## Work completed

The ANTs group-template construction completed successfully after a corrected
retry, and FreeSurfer segmentation completed on 13 November. Three-layer BEM
surfaces and an ico5 source space were prepared.

The initial BEM failed because a small inner-skull patch crossed the outer
skull near the forehead/nose bridge. Explicit T1 mode produced unchanged
surfaces; atlas mode reduced the number of outside vertices from 10 to 9 but
did not resolve the failure.

A separate local repair candidate moved 15 inner-skull vertices inward by at
most 2.21 mm, retaining the mesh connectivity. It now passes numerical BEM
geometry and solution checks, adjacent-shell nesting/contact checks, and source
containment checks: all 20,484 ico5 source points are inside the inner skull.
The original surfaces were preserved. This is a computationally usable
candidate, not an anatomically approved model. One pre-existing right pial
vertex elsewhere remains approximately 0.10 mm outside the inner skull.

## Current blocker: EEG-to-template alignment

MNE-estimated template fiducials were used for a separate exploratory rigid
alignment, without asking an inexperienced reviewer to place anatomical
landmarks manually. The existing numerical acceptance criteria were retained:
median sensor-to-scalp distance <=10 mm and 95th percentile <=20 mm.

| Arm | Nominal numerical passes | Nominal numerical failures |
|---|---|---|
| Average-seven | sub-05, sub-06, sub-07 | sub-01, sub-02, sub-03, sub-04 |
| fsaverage | All except sub-03 | sub-03 |

All nominal medians were below 10 mm; failures were due to the 95th-percentile
criterion. We ran 266 fits comprising nominal alignment and +/-2-mm landmark
perturbations. These perturbations did not change pass/fail classifications,
although some final fits were sensitive to initialization.

A further 224 diagnostic fits tested 16 rigid starting positions for each
participant/arm. Average-seven failures for sub-01 through sub-04 persisted
across all tested starts. No production transform was replaced by a selected
diagnostic best fit.

## Interpretation and limitations

The coordinate audit found no surface-header or millimetre/metre mismatch.
The average-seven scalp mesh is narrower than fsaverage (154.4 versus 170.0 mm
in surface-RAS x extent); shape or coverage differences may contribute to poor
fits, but this does not establish the cause.

All seven native forward MRI-to-HEAD transforms have identity rotations and
translations matching MRI volume centers. This is consistent with a
scanner-RAS-style convention, but does not independently establish how the
electrodes were digitized or exported. The HEAD label alone therefore does not
establish a conventional fiducial-defined frame. Released EEG sidecars repeat
the same landmarks across participants and cannot resolve this uncertainty.
Original digitization files have not been located in the searched directories.

Numerical scalp proximity is not anatomical validation. No forward calculations
for the two-arm comparison or inverse/localization fits have been launched.
There is therefore no result demonstrating that either template improves source
localization. The planned unregistered HEAD-coordinate distances are diagnostic
quantities and must not be reported as validated anatomical localization errors.

## Proposed next steps for discussion

1. Obtain the original digitization/export records, or authoritative clarification
   of the electrode and landmark coordinate conventions from the dataset authors.
2. Have an experienced EEG/MRI reviewer assess the repaired BEM, remaining pial
   outlier, estimated fiducials and sensor overlays.
3. Decide whether the rigid, no-scaling design should retain failed alignments as
   feasibility outcomes, or whether a separately specified methodological study
   is needed. Do not relax thresholds or change anatomy/scaling silently.
4. If a pilot is justified after review, sub-05/06/07 are the nominal paired
   numerical passes. Retain all seven participants in coverage/failure reporting
   and disclose the restriction and selection bias.

The experiment is currently an exploratory anatomy/alignment feasibility study.
The immediate need is methodological and provenance review, rather than another
long reconstruction run.

## Supporting artifacts

- `docs/BEM_REPAIR.md`: surface repair and validation history.
- `docs/EXPLORATORY_ALIGNMENT.md`: nominal and landmark-sensitivity results.
- `docs/SCALP_FIT_DIAGNOSIS.md`: coordinate audit and rigid-start diagnostics.
- `outputs/04x_exploratory_alignment_retry1/08_summary.png`: alignment summary.
- `outputs/04y_scalp_fit_diagnosis_retry1/08_diagnosis.png`: geometry/start audit.
- `outputs/03f_anatomy_review/`: anatomical comparison images.

# Estimated-landmark exploratory alignment — 2026-11-16

The user authorized automatically estimated landmarks and initialization
sensitivity analysis instead of being asked to place unfamiliar anatomy.
This is a separate exploratory protocol; formal anatomy review and forward gates
remain unchanged. No participant is removed, no scaling is used, and no forward
or inverse jobs were run. Anatomical localization validity remains false.

## Completed output

Use `outputs/04x_exploratory_alignment_retry1/05_manifest.json` and the adjacent
CSV/PNG files. The initial `04x_exploratory_alignment` directory is incomplete:
a strict rotation check rejected float32 fsaverage landmark arithmetic. The
runner now casts estimated landmarks to float64 before constructing rigid
transforms. Both arms were rerun in the fresh retry directory; all 266 initial
and final transform pairs passed proper rigid-transform checks.

MNE `get_mni_fiducials` estimates landmarks in each arm. For average-seven it
uses the inverse FreeSurfer Talairach affine; fsaverage uses its packaged
landmarks. This estimation is solely for rigid initialization. No source,
reference stimulation point or EEG sensor geometry is warped or scaled.
Estimated landmarks are stored only in the exploratory output folder, with
hashes and explicit unreviewed provenance.

Seven native forward FIFs provide recorded HEAD sensor coordinates. Identical
40-iteration, 15%-trimmed nearest-scalp-vertex ICP and frozen numerical criteria
(median <=10 mm; p95 <=20 mm) are used for both arms. Scalp geometry is converted
from FreeSurfer millimetres to metres. Each arm/participant has one nominal fit
plus 18 one-landmark/one-axis perturbations at +/-2 mm. Perturbation results
measure sensitivity to initialization, not confidence intervals or independent
anatomical accuracy.

| Participant | Average-seven p95 (mm) | fsaverage p95 (mm) |
|---|---:|---:|
| sub-01 | 23.62 FAIL | 9.43 PASS |
| sub-02 | 30.64 FAIL | 12.04 PASS |
| sub-03 | 39.39 FAIL | 22.60 FAIL |
| sub-04 | 34.09 FAIL | 15.36 PASS |
| sub-05 | 15.18 PASS | 11.41 PASS |
| sub-06 | 17.03 PASS | 8.22 PASS |
| sub-07 | 14.51 PASS | 8.02 PASS |

All median distances are below 10 mm, so nominal failures are due to p95.
Every participant/arm retains the same numerical pass/fail result across its
18 perturbations. Worst post-ICP sensor RMS movement for sub-04 is 11.18 mm
in average-seven and 8.97 mm in fsaverage. A numerical pass is not approval.

Outputs include 14 nominal FIF transforms, estimated landmark FIFs, 266 fits,
3,584 per-sensor distance records, seven three-projection overlay figures,
worst-channel listings and a summary plot. Gray points in overlays are scalp
projections, not an occlusion-aware surface rendering. These pictures help
diagnosis but cannot establish anatomical correspondence.

Checks verified complete 7x2 subject/arm coverage, 19 fits per combination,
finite metrics, proper rotations without reflections/scaling, and pending
review/invalid anatomical-error labels throughout.

## Next investigation

The inner-skull repair does not change the scalp, so it cannot resolve these
EEG-to-scalp fit failures. Review average-template scalp extent/shape, surface
RAS metadata and electrode-frame provenance; inspect failed-channel locations
and compare initialization basins. Do not tune thresholds, drop failing
participants, substitute native anatomy, silently permit scaling, or label
these candidate fits as reviewed. A better scalp fit to fsaverage is not evidence
that its cortical source localization is more accurate.

Reproduce in a fresh directory:

```bash
MNE_LOGGING_LEVEL=ERROR MPLCONFIGDIR=/tmp/mpl-bem OPENBLAS_NUM_THREADS=4 \
  .venv/bin/python localize_mi_average_seven/scripts/04x_exploratory_alignment.py \
  --output-root localize_mi_average_seven/outputs/04x_NEW_RUN
```

MNE estimation reference:
https://mne.tools/stable/generated/mne.coreg.get_mni_fiducials.html

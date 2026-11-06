# Explicit exploratory mapping assumption

The user authorized proceeding without contacting the authors and requested an
assumption supported by the available documentation and relevant research.
This decision is recorded in `decision.json`. It does **not** establish physical
sensor correspondence or supersede the original strict audit's BLOCKED result.
`script01_before_assumption_mode.py.txt` preserves the preparation code used for
that earlier audit. No author message was sent.

## Model assumptions fixed before geometry or localization

- Localize-MI `eN` is assigned to MNE `GSN-HydroCel-256` electrode `EN`, N=1…256.
  This is **ASSUMED_UNVERIFIED** acquisition/export correspondence.
- The cap/resource family is selected from the original acquisition description
  and audited EGI-origin template, not from localization errors or measured
  participant sensor positions. `EGI_256` is not substituted.
- The exact MNE resource is pinned by SHA-256 in `decision.json`.
- e256 maps to E256; no additional Cz waveform/channel is invented.
- MNE's fixed 95 mm spherical normalization and FidNz/FidT9/FidT10 interpretation
  are declared template conventions. Template-fiducial HEAD definition plus the
  predefined fsaverage HEAD/MRI transform are not individual registration.
- No scale search, sensor projection, ICP, affine/SyN fitting or use of individual
  digitization is introduced. The original geometry thresholds remain unchanged.

## Relevant published practice

MNE's version-matched tutorial demonstrates EEG forward modeling with standard
sensor coordinates and fsaverage when individual MRI is unavailable:
https://mne.tools/1.12/auto_tutorials/forward/35_eeg_no_mri.html
It supports the general template-modeling approach. Its 10-05 example does not
validate the HydroCel channel mapping, so its electrode definitions are not used
to relabel our channels or construct a nearest-neighbour substitute.

Nielsen et al. (2023), *Evaluating the influence of anatomical accuracy and
electrode positions on EEG forward solutions*, NeuroImage 277:120259:
https://doi.org/10.1016/j.neuroimage.2023.120259
The study compares alternatives when digitized positions are unavailable and
finds that electrode specification influences forward errors. Measured-position
templates may outperform manufacturer layouts. This supports reporting geometry
uncertainty and does not prove this experiment's mapping or anticipated accuracy.

The experiment-specific support is the original HydroCel acquisition description,
the public importer's row labels, all 61 channel-metadata tables, and the audited
template provenance described in `source/01_montage/documentary_review_2026-11-06/`.
None supplies the missing acquisition/export lookup. We retain that uncertainty
rather than claiming that another researcher has validated it.

## Run preparation manually

From the repository root:

```bash
.venv/bin/python cross_subject_fsaverage/template_montage_no_digitization/scripts/01_prepare_experiment.py \
  --mapping-policy documented-assumption \
  --assumption-record cross_subject_fsaverage/template_montage_no_digitization/assumptions/numbered_hydrocel_v1/decision.json \
  --source-root cross_subject_fsaverage/template_montage_no_digitization/source_assumed_v1 \
  --output-root cross_subject_fsaverage/template_montage_no_digitization/outputs/01_prepare_experiment_assumed_v1
```

The new audit will generate all 5,124 planned jobs, a 256-row assumed mapping,
and template-only geometry QA if the record and resource match. Mapping labels,
coordinate rows, subset definitions and provenance retain the assumption status.
An unchanged geometry gate failure still yields BLOCKED. A numerical success is
`EXPLORATORY_NUMERICAL_REVIEW_REQUIRED`, with physical correspondence and
anatomical localization accuracy still unverified. Forward modeling remains
unauthorized by preparation and Script 02 remains unimplemented.

The code was inspected statically and syntax-checked; this preparation mode has
not been executed. Share the new readiness and geometry reports before the next
stage. Repeated audits must use fresh source/output directories.

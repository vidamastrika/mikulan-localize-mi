# Coordinate provenance audit — 4 November 2026

## Scope and conclusion

Read-only audit of original preprocessing/export code, modern baseline/parameter-grid
code, frozen no-warp solver code, no-warp and warp reconstruction/evaluation,
all seven current forward/H5/EEG coordinate artifacts, and saved anatomical bridge
and registration QA. No inverse, forward, registration, or expensive mapping was
rerun for this audit. The pending fsaverage registration session was interrupted
(exit 130); its partial output folder is not a completed registration result.
Existing results, coordinates, and transforms were not edited.

**The EEG export has a confirmed frame-label inconsistency and shared geometry.
This does not demonstrate that the original forward solutions are invalid.**
Their numerical transforms are internally consistent for all seven subjects.
Independent anatomical EEG-to-MRI registration accuracy is not established by
these checks. The stimulation contacts require a separate coordinate chain.

## Seven-subject audit table

N = numerical matrix/metadata consistency, F = coordinate-frame correctness,
A = independent anatomical evidence, P = missing provenance. Detailed condition
impacts are also included in `02_subject_coordinate_provenance_audit.csv`.

| Subject | Runs | N | F | A | P | Contact bridge max residual (mm) | Impact B / NW / affine / SyN / fsaverage |
|---|---:|---|---|---|---|---:|---|
| sub-01 | 11 | PASS, inverse H5 = forward | Forward tags consistent; EEG TSV mislabeled/shared | Not independently established | HPTS, original epochs, accepted coregistration record | 0.014208 | B conditional / NW within conditional, cross diagnostic / affine conditional / SyN conditional+QA exclusions / HOLD |
| sub-02 | 6 | PASS, inverse H5 = forward | Forward tags consistent; EEG TSV mislabeled/shared | Not independently established | HPTS, original epochs, accepted coregistration record | 0.013770 | B conditional / NW within conditional, cross diagnostic / affine conditional / SyN conditional+QA exclusions / HOLD |
| sub-03 | 7 | PASS, inverse H5 = forward | Forward tags consistent; EEG TSV mislabeled/shared | Not independently established | HPTS, original epochs, accepted coregistration record | 0.009715 | B conditional / NW within conditional, cross diagnostic / affine conditional / SyN conditional+QA exclusions / HOLD (scalp fit FAIL) |
| sub-04 | 9 | PASS, inverse H5 = forward | Forward tags consistent; EEG TSV mislabeled/shared | Not independently established | HPTS, original epochs, accepted coregistration record | 0.005810 | B conditional / NW within conditional, cross diagnostic / affine conditional / SyN conditional+QA exclusions / HOLD |
| sub-05 | 14 | PASS, inverse H5 = forward | Forward tags consistent; EEG TSV mislabeled/shared | Not independently established | HPTS, original epochs, accepted coregistration record | 0.012250 | B conditional / NW within conditional, cross diagnostic / affine conditional / SyN conditional+QA exclusions / HOLD |
| sub-06 | 7 | PASS, inverse H5 = forward | Forward tags consistent; EEG TSV mislabeled/shared | Not independently established | HPTS, original epochs, accepted coregistration record | 0.012790 | B conditional / NW within conditional, cross diagnostic / affine conditional / SyN conditional+QA exclusions / HOLD |
| sub-07 | 7 | PASS, inverse H5 = forward | Forward tags consistent; EEG TSV mislabeled/shared | Not independently established | HPTS, original epochs, accepted coregistration record | 0.010451 | B conditional / NW within conditional, cross diagnostic / affine conditional / SyN conditional+QA exclusions / HOLD |

B means the modern historical within-subject baseline used in our experiment
comparisons. “Conditional” means coherent as an error in native or registered
coordinates under the released anatomical model, without claiming independently
validated registration accuracy. Forward tags alone do not prove physical frame
correctness. Neither a small roundtrip nor a good scalp residual supplies A.

## 1. EEG coordinate lineage

1. `_01_preprocess.py` reads subject-specific `*_egi_dig.hpts` with
   `mne.channels.read_montage(..., unit='mm')` and attaches the montage before
   epoch processing. The historical API call documents digitization input, but
   its exact reader behavior/version and original HPTS files were not recovered.
2. `fx_source_loc.py::make_forward_solution` reads an original FIF epoch file,
   then calls `mne.make_forward_solution(epo.info, trans_fname, src_fname,
   bem_fname)`. Thus native sensor geometry and an explicit transform enter the
   leadfield before BIDS export. `make_anony_fwd` similarly uses epoch info.
3. `_08_bids_conversion.py` copies the native forward FIF and writes the saved
   `*-trans.fif` matrix into `*_from-head_to-surface.h5`. It selects a subject
   epoch file for the EEG electrode/landmark exports.
4. `fx_bids.py::make_electrodes_eeg_tsv` directly copies `epo.info['chs'][...]
   ['loc'][:3]`, removes STI, rounds to five decimals, and writes metres.
   `make_coordsys_eeg_json` directly copies digitized landmarks but labels both
   electrode and landmark coordinate systems T1w. Neither applies the native
   head-to-MRI matrix. These functions do not convert to scanner RAS either.
5. `fx_bids.py::export_data_bids` exports EEG values into NPY plus channel/time
   metadata. Released NPY epoch arrays contain no independent digitization
   record. Original pre-export FIF epochs were not found in this audit.
6. Modern `src/localize_mi/io.py::load_run` reads the released arrays and JSON,
   but obtains montage positions from the forward via `_create_forward_montage`
   with `coord_frame='head'`. It does not read the EEG electrode TSV as geometry.
   The frozen no-warp `io.py` and `metrics.py` equal the current files bytewise.

### What is actually established about the TSV frame?

All seven EEG electrode TSV hashes are identical. Their values match sub-01's
forward HEAD positions to 0.004920 mm RMS (consistent with export rounding),
whereas sub-02–07 differ from their own forward HEAD geometry by 16.903–48.321 mm
RMS. All seven sidecars also share exactly the same landmark coordinates.
This is strong evidence for sub-01-like HEAD geometry exported under a T1w label;
it is not evidence that each TSV contains its own subject's HEAD montage.
The exact historical cause of the shared export cannot be reconstructed from
current code and released files alone.

### Missing transform versus missing documentation

For HEAD positions h in metres and saved H5 H:

- Surface RAS: `s = H @ [h, 1]` (metres).
- Native scanner/T1w RAS in mm: `r = (Norig @ inv(Torig)) @ [1000*s, 1]`.

The export applies neither step. If the desired release frame was HEAD, the
metadata label/documentation is wrong. If T1w was intended, the corresponding
HEAD-to-anatomical conversion is missing. Applying each subject's H to the
shared TSV would not restore missing subject-specific digitization. T1w is
not interchangeable with MNE/FreeSurfer surface RAS. In this dataset the checked
combined HEAD-to-scanner bridge happens to be nearly identity; that does not
establish a conventional fiducial-defined HEAD origin or validate the TSV label.
See [FreeSurfer coordinate definitions](https://surfer.nmr.mgh.harvard.edu/fswiki/CoordinateSystems)
and [MNE frame definitions](https://mne.tools/stable/auto_tutorials/forward/20_source_alignment.html).

## 2. Native forward consistency independently of EEG sidecars

Fresh checks directly read every forward and H5. All seven have 256 finite
sensor positions, HEAD sensor tags (FIFF 4), loaded HEAD source-space tags
(4;4), MRI->HEAD transform tags (5->4), determinant 1, zero rotation
orthogonality discrepancy, and exactly zero discrepancy between `inv(H5)` and
forward `mri_head_t`. No forward contains `info['dig']` entries. The inspected
transforms have identity rotation; independent checks are needed to determine
how the released HEAD frame was established.

`01_fresh_forward_and_tsv_checks.csv` records the complete measurements.
These checks establish internal numerical and tagging consistency, not a fresh
verification of the leadfield computation or anatomical alignment. Source
coordinates are reported as loaded by MNE, not asserted to be the on-disk
source-space frame before loading.

Existing `cross_subject_warp/outputs/06_coordinate_bridge/` validates the
surface/scanner bridge from both pial GIFTI volume geometries against MRI
headers and named stimulation contacts. Its 67 recorded source hashes were
recomputed in this audit: all match. This anatomical-image geometry evidence
is independent of EEG TSVs, but remains internal dataset evidence rather than
an independent digitization/coregistration accuracy assessment.

## 3. Stimulation ground truth: separate from EEG electrodes

`fx_bids.py::make_electrodes_ieeg_tsv` exports contact tables from `x_surf` and
`x_mri` families, dividing millimetres by 1000. These are distinct tables and
coordinate families from EEG channel `loc`. Surface contacts are native surface
RAS; T1w contacts are native MRI physical coordinates. Their common unit does
not imply a common frame.

`localize_mi.metrics::load_stimulation_info` / `load_stimulation_midpoint`
parse the named bipolar pair from run metadata and average its two surface
contact coordinates. Hemisphere is inferred from the stimulation label, not
from EEG geometry. Baseline cortical peak coordinates from the HEAD source
space are transformed with H before comparison to the surface midpoint.
No-warp Script 06 instead maps that surface midpoint through `inv(H)` into
its own test HEAD coordinates; it never uses the EEG TSV as ground truth.

`06_validate_coordinate_bridge.py::volume_geometry` derives
`Norig @ inv(Torig)` from native GIFTI volume geometry; `contacts_in_mm`
checks corresponding surface and T1w contacts after explicit unit conversion.
All seven saved checks pass: maximum residuals 0.005810–0.014208 mm. This supports
frame/unit consistency, not independent accuracy of contact localization from
CT/MRI or surgical records. Original contact annotation/registration provenance
remains necessary for anatomical accuracy claims.

For affine/SyN evaluation, `06_mapped_evaluation.py::evaluate_records` converts
reference HEAD peaks to reference scanner RAS, applies saved reference-to-test
registration with explicit RAS/LPS handling in the pinned point mapper, and
compares to the test midpoint in test scanner RAS. Ground truth uses only its
own native bridge. A Talairach annotation on GIFTI is not silently substituted
for the native bridge. Direction, grid/domain, and roundtrip QA apply separately.

## 4. Experimental exposure and validity of reported metrics

| Condition / code path | EEG TSV used for geometry? | Interpretation after audit |
|---|---|---|
| Modern historical baseline / `scripts/09_full_parameter_grid.py`, 15, 23 via `load_run` | No; native forward montage | Native within-subject errors remain numerically coherent, conditional on anatomical sensor/contact accuracy; historical parameter selection remains in-sample |
| No-warp LOSO within | No; test model montage and H5 | Same-subject HEAD distance remains a native anatomical-model localization error under those assumptions |
| No-warp cross / both parameter strategies | No; explicitly chosen reference-model geometry | Raw distance between different subjects' HEAD coordinates remains diagnostic only; cannot be promoted to anatomical error |
| Warp reconstruction / `03_run_localization.py`, reused by 05 | No; explicitly chosen model geometry | Native peaks are not invalidated solely by EEG sidecar labeling |
| Final affine mapped evaluation | No | Registration-dependent localization error in test scanner RAS for QA-passing mappings; independent anatomical quality still requires review |
| Final SyN mapped evaluation | No | Same, with pair/point QA exclusions; failures have unavailable anatomical errors |
| Legacy `fx_bids.py::load_bids` | Yes | Exposed: builds HEAD montage from released TSVs and shared landmarks; any historical outputs using this path need artifact-specific provenance before acceptance |
| Fsaverage Script 02 | TSV read for discrepancy audit only | Its forward-derived ICP remains a candidate; labels/shared fiducials cannot independently validate it; sub-03 gate remains FAIL |

Scope limitation: the modern baseline's current code and frozen snapshot support
the stated geometry path. The baseline result snapshot alone does not prove
which historical code revision produced every value and lacks peak coordinates.
Original `_03_source_loc.py`/`fx_source_loc.py` operate on FIF epochs; results
from that earlier workflow must be distinguished from the modern baseline and
any consumer of `fx_bids.load_bids`. Do not infer that all past outputs share
one loader.

### Saved primary mapped-evaluation counts (not recomputed mappings)

From `cross_subject_warp/outputs/06_evaluate_localization_mapped/03_primary_localization.csv`:

| Strategy | Mode | Valid numerical errors | Other records |
|---|---|---:|---|
| LOSO | within | 732 | 0 |
| LOSO | affine | 731 | 1 reconstruction FAIL |
| LOSO | SyN | 731 | 1 reconstruction FAIL |
| Single-reference | affine | 4,378 | 14 reconstruction FAIL |
| Single-reference | SyN | 3,684 | 694 pair-QA exclusions; 14 reconstruction FAIL |

The corresponding saved pair QA has 42/42 affine passes and 36/42 SyN passes.
These refer to this primary output and its saved QA policy; later threshold
sensitivity folders define different cohorts and must retain their policy labels.
`localization_error_valid=True` is an implementation acceptance flag, not a
certificate of independent anatomical registration accuracy. Comparisons across
conditions must match cases and respect excluded failures. Older unmapped or
unbridged errors cannot be accepted merely because a later evaluation passes.

## 5. Recommended next steps and scientific claim limits

1. Keep the fsaverage pipeline on hold. The partial repaired registration is
   interrupted; do not consume it as a completed mapping.
2. Recover original HPTS files, original FIF epochs with digitization, native
   trans FIFs with frame tags, source/BEM construction provenance, and accepted
   anatomical coregistration screenshots or reports for every subject. Confirm
   whether released HEAD geometry was deliberately re-expressed in scanner axes.
3. Recover stimulation annotation and CT/MRI registration provenance separately.
   Review contact overlays and sensor/head alignment for all seven subjects;
   sub-03 is a priority, not a uniquely proven invalid subject.
4. Trace historical baseline and any `load_bids` consumers to immutable code and
   geometry artifacts. Retain current outputs as reported with their assumptions;
   do not silently repair exported geometry or retroactively change error values.
5. If corrected EEG metadata/geometry is later justified, create a versioned
   derivative with explicit units, frame definitions, transforms, and hashes.
   Do not transform a duplicated montage as if it were seven native montages.
6. Accept affine/SyN accuracy claims only for documented QA cohorts, with
   anatomical registration review. Native bridge consistency and MRI roundtrips
   alone are insufficient. No-warp cross distances remain non-anatomical.

Audit limitations: no leadfields or inverse solutions were recomputed; original
unreleased acquisition/annotation records are missing; no independent anatomical
registration was established. These unresolved items are recorded as missing
evidence rather than converted into either automatic PASS or blanket invalidity.

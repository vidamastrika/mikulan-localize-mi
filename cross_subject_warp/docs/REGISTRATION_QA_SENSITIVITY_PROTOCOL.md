# Registration-QA attribution and 0.15 mm sensitivity protocol

Status: COMPLETE FOR THE APPROVED QA-SENSITIVITY SCOPE. The user ran the revised
attribution audit, all-pair QA015 run, mapped sensitivity evaluation and
cross-policy comparison. All outputs are in separate directories. The existing
0.1 mm primary analysis remains authoritative and unchanged.

## 1. Observed baseline and scope

Read-only inspection of saved historical outputs established:

- Primary registration QA: 42/42 affine and 36/42 SyN checks passed; six SyN
  mode/pairs remain FAIL. The completed primary Script 06 output retains them.
- The diagnostic subset contains 2,400 sampled points across both directions of
  these six pairs. Of these, 2,366 are destination-interior/valid roundtrips;
  34 remain explicitly outside the accepted destination interior.
- Six valid points exceed 0.1 mm; none exceed 0.15 mm. Each flagged pair has one
  exceedance in one direction. Errors range from 0.1002518763 to 0.1253244961 mm.
- Their source-image FOV edge distances range from about 35 to 70 voxels. This
  does not establish distances to intermediate displacement-field domains or
  exclude interpolation/inverse-field problems.
- Script 06 completed with 10,980 rows per peak scenario, no mapping-execution
  failures, 19,056 PASS_MAPPED rows across the two scenarios, 1,464 PASS_WITHIN,
  1,388 PAIR_QA_FAIL and 52 RECONSTRUCTION_FAIL rows.
- All 732 primary within-subject errors match the completed no-warp evaluation;
  the largest absolute difference is 7.105427357601002e-15 mm.

The 2,400 samples are a selected diagnostic subset, not a random sample of all
registration mode/pairs. Do not extrapolate its exceedance rate to the cohort.
Roundtrip consistency is not proof of anatomical registration accuracy.

## 2. Freeze the primary analysis

Preserve without modification:

- `outputs/06_registration_qa/` and `outputs/06_roundtrip_diagnostics/`.
- `outputs/06_evaluate_localization_mapped/` and native primary/peak-sensitivity
  tables, selection manifests, coordinate bridges and frozen source snapshots.
- All original and imported affine matrices, SyN affine components, forward and
  inverse fields, warped images, MRI inputs and fitting/QA logs.
- The primary pair tolerance AND individual source-point tolerance of 0.1 mm.

Every new stage requires an absent destination, validates input hashes and
records a parent-manifest chain, executable/source hashes and software versions.
Partial/interrupted results cannot be labelled COMPLETE. No overwrites, in-place
repairs, global-constant monkeypatching, refitting, peak reselection or automatic
changes to primary registration status are allowed. Experimental outputs and
source files are not fixtures to be edited to make tests pass.

## 3. Explicit policy and analysis axes

The proposed sensitivity changes **pair-level MRI-sample roundtrip tolerance
only**. Retaining source-point validation is interpreted conservatively as
retaining its existing 0.1 mm numerical threshold, not just retaining its name.

| Requirement | Preserved primary policy | Proposed QA sensitivity |
|---|---|---|
| `qa_policy` | `primary_rt010` | `pair_rt015_source_rt010` |
| Pair MRI-sample maximum roundtrip | <=0.1 mm, both directions | <=0.15 mm, both directions |
| Individual reconstructed source-point roundtrip | <=0.1 mm | <=0.1 mm, unchanged |
| MRI sample selection | 200 per direction, seed 2901, original foreground rule | Identical points and rule |
| Minimum valid MRI coverage | 75% per direction | Unchanged |
| Image reproduction RMSE | <=0.015, same forward and saved SyN inverse image checks | Unchanged |
| Physical grid discrepancy | <=0.02 mm over grid corners | Unchanged |
| Native coordinate bridges/units | Seven verified bridges, RAS mm / ANTs LPS conversions | Unchanged |
| Image/field domain margins | Original MRI and required intermediate-field interior checks | Unchanged |
| Files, interpolation and transform order | Existing hashed transforms, verified point sequence and production interpolation | Unchanged |
| Inverse fits, parameters, montage and peak extraction | Frozen native records | Unchanged |
| Failure accounting | Retain rows, native failures and invalid errors | Unchanged |

Changing source-point tolerance to 0.15 mm would be a second experimental change;
it is NOT included in this proposal. No diagnostic interpolation alternative
will be used to generate sensitivity localization errors.

Retain two independent axes in every table/key:

1. Registration QA policy: `primary_rt010` or `pair_rt015_source_rt010`.
2. Existing peak scenario: original `scenario=primary` or `scenario=sensitivity`
   (the latter is the historical mixed whole-brain peak policy).

Add an explicit `peak_scenario` alias if needed for readable reports, but retain
the original scenario values and keys. Never describe QA sensitivity and
whole-brain peak sensitivity as the same analysis. There are four policy/scenario
cells per strategy; neither axis changes the native reconstruction attempts.

## 4. Mechanism investigation before sensitivity acceptance

### A. Reproduce and inspect intermediate domains

Repeat exactly the saved 2,400 samples with unchanged files, tool, samples,
foreground seed and production point sequence. Compare original and reproduced
coordinates, validity masks, residual vectors and counts, not maxima alone.
Keep all samples, including the 34 outside the destination interior.

For each of the six offending points, all original samples, and diagnostic-only
small neighbourhoods/voxel-centre probes:

- Record the initial scanner RAS and corresponding ITK LPS positions.
- Record each affine/field intermediate position, its continuous voxel index,
  fractional voxel position, and distance to the relevant MRI AND field-domain
  boundaries. Source FOV distance alone is insufficient.
- Check that every interpolator stencil lies in its declared domain. Record
  any zero-displacement/outside-domain fallback separately, never as a valid
  inverse check. Preserve the original masks for reproducing primary QA;
  additional diagnostic probes do not change its pass/fail result.
- Record finite field values, voxel directions/origins/spacings and component
  conventions. Do not infer vector-component signs from filenames or assume
  that raw NIfTI components can be added to RAS coordinates without validation.

ITK documents displacement addition, a default vector-linear interpolator and
zero displacement outside the field. These behaviours motivate explicit domain
checks; they do not identify the cause of these six errors by themselves.
[ITK DisplacementFieldTransform](https://docs.itk.org/projects/doxygen/en/stable/classitk_1_1DisplacementFieldTransform.html)

### B. Test transform composition independently

Compare the production composite ANTs point call with individually staged calls
using the SAME files, physical coordinates and documented order. Validate
RAS/LPS conversion, affine inversion and the affine fixed centre/offset through
an independent affine calculation. Do not approximate a centred affine by
applying its linear matrix and translation without its centre.

Use analytic noncommuting synthetic transforms to establish the independent
implementation and point-order/sign conventions BEFORE interpreting real cases.
Do not try arbitrary real-data transform orders and select one by MRI or
localization-error improvement. The accepted production chain remains fixed.

In native physical LPS coordinates, define the saved affine as A, the forward
field map W(x)=x+u(x), and the inverse field map V(x)=x+v(x). The checked chain is:

- Test -> reference: A(W(x)).
- Reference -> test: V(A^-1(y)).

Measure full versus staged point discrepancies in both directions. The reference
cycle residual is compared with A's linear part applied to the corresponding
field-closure residual; distinguish this affine amplification from a changed
composition order. The derivation/order must first pass independent controls.

The production order is supported by the official ANTs point examples, which
also distinguish point transformations from image resampling and specify LPS.
[ANTs point-data conventions](https://github.com/ANTsX/ANTs/wiki/Applying-transforms-to-point-data)

### C. Measure stored-field inverse closure

Using a vector convention/interpolator established by the independent controls,
evaluate both closure vectors within valid field domains:

- u(x) + v(x+u(x)).
- v(z) + u(z+v(z)), where z=A^-1(y) for reference-origin probes.

Record each component and vector norm, at original off-grid probes and nearby
field-grid centres. Compare with the staged and full composite residual vectors.
Measure affine-only inverse residual separately as a control.

Use native ITK/SimpleITK reading/interpolation where available, or an independently
validated physical-vector reader. A SciPy/raw-array implementation is not accepted
until its physical vector convention matches analytic controls and the native
reader. Report missing optional diagnostic backends instead of treating them as
successful checks or installing software silently.

Agreement between the observed cycle residual and stored-field closure supports
an **operational inconsistency of the saved interpolated forward/inverse pair**.
It does not, by itself, prove whether inversion estimation, finite storage or
interpolation of an underlying continuous transform caused that inconsistency.

### D. Bound precision and interpolation effects

Record stored field data types, executable/version evidence, CSV input/output
precision and supported tool precision options. Test high-precision diagnostic
serialization/native in-memory application only if the installed tool/backend
supports it. Do not invent a CLI flag or claim float64 casting recovers information
already lost in stored float32 values.

Use analytic controls to measure the I/O and affine-only precision floor before
looking at the six cases. Freeze any software-agreement tolerance from those
controls and documented numeric formats; do not enlarge it to encompass real
residuals. Repeat staged calculations with the same stored samples in double
arithmetic to isolate arithmetic/I/O from the field values themselves.

Compare production linear interpolation with diagnostic-only alternatives and
voxel-centre/neighbourhood probes as mechanism evidence. These alternatives must
never replace production interpolation, rewrite fields or determine QA/localization
acceptance. Do not numerically invert or refit the real fields to manufacture a
better roundtrip. No inverse EEG reconstruction occurs.

### E. Attribution decision and stop conditions

Report evidence per pair/direction, allowing mixed or unresolved conclusions:

| Evidence | Permitted conclusion |
|---|---|
| Analytic controls or full/staged agreement fail | Diagnostic implementation/composition discrepancy needs resolution; do not attribute to interpolation |
| Affine-only residual is appreciable relative to the cycle | Investigate affine parsing/inversion, centre, convention or I/O precision |
| Full/staged/native-reader results agree and residual follows field closure | Stored interpolated fields are not exact mutual inverses at those points |
| Off-grid/precision probes change residual beyond the independently established precision floor | Evidence of interpolation/precision sensitivity; quantify it |
| Multiple mechanisms contribute or continuous original transforms are unavailable | Mixed/undetermined attribution; retain this limitation |

Small errors and deep source-FOV locations are insufficient to declare a numerical
cause. Distinguishing discretized inverse-field inconsistency from interpolation
of an unavailable continuous inverse may remain non-identifiable. Report that
rather than claiming certainty.

If a reproducible coordinate, composition or out-of-domain implementation error
is discovered, stop sensitivity acceptance and document it. A larger threshold
must not conceal a software/convention error. Preserve primary outputs; any later
corrected analysis would require separate versioned outputs and explicit scope.

## 5. Separate 0.15 mm registration-QA run

After the mechanism controls pass (an exact scientific attribution may remain
unresolved), rerun all 84 mode/pair checks under the explicit new pair policy.
Do not assess only the six failures or regenerate samples until they pass.
Use identical MRI samples, all original non-roundtrip gates, transforms and tool.
Log intermediate metrics even when a check fails; do not lose partial statistics.

For each of 84 mode/pairs, store `primary_qa_status`, `primary_reason`,
`qa_sensitivity_status`, both directional coverage/roundtrip metrics, unchanged
image/grid checks, thresholds, hashes and parent report identity. A previous
primary failure can become `PASS_QA015_ONLY` only in the new sensitivity report
following this full run. It remains FAIL in every original primary artifact.

Retrospective numerical classification of the SAVED samples suggests:

| Policy | Affine pair checks | SyN pair checks | Total |
|---|---:|---:|---:|
| Existing primary 0.1 mm | 42 PASS / 0 FAIL | 36 PASS / 6 FAIL | 78 PASS / 6 FAIL |
| 0.15 mm candidate classification, not a new QA result | 42 candidates / 0 known threshold failures | 42 candidates / 0 known threshold failures | 84 candidates / 0 known threshold failures |

The second row is not authorization or automatic promotion of failed pairs.
Publish actual new-policy pass/fail counts only after all unchanged conditions
are revalidated. New source-point results remain subject to 0.1 mm and may still
be invalid even for pairs accepted under QA015.

## 6. Sensitivity localization and matched comparisons

Reuse exactly the same native configurations, peaks, primary/peak-sensitivity
records, target midpoint definition and validated bridges. Transform only the
predicted reference source. Map only mode/pairs accepted under the selected QA
policy; apply the ORIGINAL source-point threshold and all MRI/intermediate-field
domain checks. Keep reconstruction, pair QA, point QA and execution failure
statuses separate. No successful source is substituted for a failed inverse fit.

Retain 10,980 evaluation rows in each peak scenario per QA policy (21,960 total
for a policy), including 732 within baselines per peak scenario. Within baselines
are not duplicated by registration mode. Do not duplicate or pool the QA policies
as if they were independent observations.

Every evaluation/comparison key includes QA policy plus strategy, original peak
scenario, method, reference, test, run, condition and registration mode as
appropriate. Compute and report:

1. **Within-policy affine vs SyN:** exact matched valid rows for each strategy,
   method and peak scenario. Report error distributions and the paired SyN-minus-
   affine difference. All failed/ineligible rows remain in marginal coverage
   denominators; do not replace their errors with zeros or arbitrary penalties.
2. **Common cohort across policies:** the intersection where BOTH modes are
   valid under BOTH policies. Compare primary and QA015 on this identical set.
   With unchanged transforms and the fixed source threshold, accepted common
   coordinates/errors should be unchanged; verify to 1e-9 mm (a software regression
   check, NOT a QA acceptance tolerance). Investigate any larger difference.
3. **Newly eligible cohort:** QA015 matched-valid rows absent from the primary
   matched cohort. Report these separately, including subjects/references/methods,
   native outcomes and residual/coverage metrics. This is a coverage change,
   not improved registration or changed inverse estimates.

Show per-mode scheduled, reconstruction-failed, pair-excluded, point-excluded,
execution-failed and valid counts, plus matched counts and unique subject-pair
coverage. Stratify by strategy/method/peak scenario; any pooled row-level summary
is descriptive. Do not assume repeated methods/runs/references are independent
or add unplanned significance tests. Keep the 0.15 mm analysis explicitly
**post-hoc exploratory registration-QA sensitivity**, chosen after observing the
six residuals. It cannot retrospectively redefine or replace the primary result.

Existing primary-QA baseline, read from the completed matched CSV (pooled
method/run records, descriptive only; delta is median of paired differences):

| Strategy | Peak scenario | Matched n | Affine median mm | SyN median mm | Median paired delta mm |
|---|---|---:|---:|---:|---:|
| LOSO | primary | 731 | 23.752162 | 20.746372 | -1.309644 |
| LOSO | sensitivity | 732 | 23.656991 | 20.810159 | -1.308202 |
| Single-reference | primary | 3,684 | 24.139785 | 23.483672 | -0.582835 |
| Single-reference | sensitivity | 3,687 | 24.139785 | 23.483672 | -0.584204 |

QA015 localization results and matched counts are not yet available and must
not be predicted from the pair-QA candidate count.

## 7. Implementation and new output locations

The new driver `scripts/06_registration_qa_sensitivity.py` and
`scripts/qa_sensitivity/` helpers provide explicit investigate, QA, evaluate and
compare stages. Keep the existing primary Script 06 entry points,
helpers, constants and defaults byte-for-byte unchanged. Reuse immutable point
application/geometry functions and saved native inputs where compatible.

Represent pair and source tolerances as separate immutable policy fields; do not
monkeypatch existing module globals or pass a 0.15 mm policy into a 0.1 mm source
validator by accident. Any generalized kernel must demonstrate primary-policy
parity against real saved outputs before sensitivity evaluation can publish.

All paths below are NEW under `cross_subject_warp/outputs/`:

| Stage | New directory | Principal new artifacts |
|---|---|---|
| Attribution | `06_registration_qa_attribution_domain_review/` | replay checks, intermediate domain audit, composite/staged and affine residuals, field closure, precision/interpolation probes, cause assessment, source manifest |
| QA sensitivity | `06_registration_qa_sensitivity_rt015/` | all 84 policy-labelled checks, original-vs-new status ledger, directional metrics, both-threshold counts, unchanged-gate audit, new manifest |
| Mapped sensitivity | `06_evaluate_localization_qa_rt015/` | historical 01–08 evaluation names, new QA-policy labels, native/QA/point failure tables, matched mode pairs and source-point audit |
| Cross-policy comparison | `06_compare_registration_qa_thresholds/` | coverage ledger, within-policy matched statistics, common-cohort checks, newly eligible cohort, per-method/strategy/peak tables, provenance and test report |

All directories must be absent; no `--overwrite`. Preserve the current native,
bridge, QA, diagnostic and evaluation inputs as read-only. The output manifests
must include the parent primary analysis, explicit thresholds and axis labels,
software/executable hashes, checks run, checks failed/not run, and limitations.
Sample terminal commands are in the driver's opening comments and in
`scripts/README.md`. The user runs each real-data stage explicitly; actual
analysis stages have not been run during implementation.

## 8. Test and validation procedure

Prioritize available real historical evidence; use synthetic cases only for
specific mechanics/edge cases not established by historical data.

| Coverage | Evidence/check | Status now |
|---|---|---|
| Native/no-warp consistency | 732 within errors, native keys/configs/outcomes, immutable hash receipts | Both peak scenarios passed at 7.105427e-15 mm maximum difference |
| Current primary evaluation | 21,960 rows, recorded failure partition and zero mapping-execution failures | Historical-table regression checks passed; primary artifacts/hash receipts preserved |
| Existing diagnostic reproduction | All 2,400 points and masks; six threshold exceedances; 2,366 valid | User replay passed all 12 directions; one field-outside MRI-interior sample excluded from scoring |
| Primary algorithm parity | Real saved pair statuses, mapped rows and matched cohorts unchanged at original policy | All 84 primary QA flags reproduced; common-cohort coordinate/error parity passed |
| Composition and vector convention | Analytic noncommuting affine/displacement and RAS/LPS/centred-affine controls | Six native ANTs controls passed; full/staged real point agreement within frozen budget |
| Inverse/interpolation attribution | Controlled exact-inverse and approximate inverse fixtures | Saved interpolant closure reproduced all six exceedances; unique historical cause remains unidentifiable |
| Threshold boundaries | <=0.1 and <=0.15, just-above values, pair vs source policies kept separate | Synthetic edge checks passed |
| Failure and coverage accounting | Original FAIL flags preserved; missing/domain-invalid points never become valid; matched-key and denominator integrity | Synthetic and historical-table checks passed |
| QA015 scientific results | Full non-roundtrip gates and 0.1 mm source validation on real historical resources | 84/84 pass at pair .15 mm; original .1 mm remains 78/84; source point threshold unchanged |
| Matched comparison | Within-policy, common cohort, newly eligible and no-warp checks | 10,222 matches total; 1,388 newly eligible; common cohort and both no-warp checks passed |
| Source immutability | Pre/post hashes for original CSVs, manifests, code and transform inputs | Stage manifests verified; primary output roots unchanged |

The full repository suite and focused sensitivity suite passed; the final count
is recorded in `REGISTRATION_QA_SENSITIVITY_VALIDATION.md`. User-run real-data
stages completed without registration fitting or inverse reconstruction. No
code-coverage percentage is claimed. No synthetic pass is evidence of
anatomical accuracy, improved registration or validity of a changed threshold.

Routine tests must not invoke EEG inverse reconstruction or registration fitting,
install packages, modify real fixtures, or vary experimental parameters to get a
pass. Optional ANTs/native-reader integration checks are separately marked with
runtime/dependency requirements. Real MRI/field QA and sensitivity mapping are
explicit user-run analysis stages, not hidden expensive unit tests.

Every implemented stage must emit a validation report listing test/check IDs,
input source (synthetic or historical), passed/failed/skipped/not-run counts,
coverage of mechanisms and failure branches, numerical agreement bounds and
scientific limitations. A percentage code-coverage number may be reported only
if actually measured; it is not a scientific validation metric.

## 9. Primary artifact hash receipts (read-only at proposal time)

Paths are relative to `cross_subject_warp/outputs/`. Hash receipts were checked
again against the preserved primary outputs during implementation:

| Artifact | SHA-256 |
|---|---|
| `06_registration_qa/01_pair_mode_qa.csv` | `5fd4d0bfdcb426b227f7c41775d8499dceefca688dc35a342024d54aed54b9c1` |
| `06_registration_qa/05_registration_qa_manifest.json` | `91bba516e18b7596323283afc625ba4c0923d706c0a75ca48c0af914d265ed39` |
| `06_roundtrip_diagnostics/02_point_diagnostics.csv` | `98ab1c4acd1be71675850f8a3735695632000bf18e24eabcc610c22eb8d727e2` |
| `06_evaluate_localization_mapped/07_evaluation_manifest.json` | `afd6d1991ca7d35c3b97c23415ac1d6a8ad5a9f3bf7ffae1b2c380818e1127f5` |
| `06_evaluate_localization_mapped/12_matched_affine_syn.csv` | `7358fd859a7e03466f30493249e4d35bae9b6c7c3db4fc7c21b74a0555db47f3` |

These receipts do not replace a full source/transform manifest or a fresh QA015
run. They anchor preservation checks and keep the existing primary decisions
separate from proposed exploratory sensitivity results.

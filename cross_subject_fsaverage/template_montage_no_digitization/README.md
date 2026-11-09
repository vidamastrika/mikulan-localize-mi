# Fsaverage with template electrodes and no individual digitization

This separate Localize-MI variant assumes test EEG and known channel identities,
without test MRI, measured electrode positions or individual coregistration.
It transfers the frozen LOSO and Single Reference configurations to fsaverage
anatomy and a documented, common template cap. Both anatomy and sensor geometry
change relative to the earlier no-warp condition; this is not an anatomy-only
comparison.

## Primary comparison policy: no MRI alignment

The user explicitly selected the original no-warp evaluation convention for
the primary comparison. Use Script 04's `diagnostic_distance_mm` and the
no-warp Script 06 `08_raw_head_diagnostic.csv` rows with `scenario=primary`.
Match strategy, method, reference subject, test subject and run. The no-warp
`sensitivity` rows use a different mixed peak-selection policy and must not be
silently substituted into the primary comparison.

Neither primary metric applies MRI registration to estimates or ground truth:
both subtract the recorded model/template HEAD estimate from the audited test
HEAD ground-truth coordinate in millimetres. These remain unregistered
coordinate diagnostics, not anatomically validated localization errors.
Original dataset ground-truth coordinates and their established coordinate
conversion are still evaluation references; this policy does not claim that
ground-truth acquisition itself required no anatomical information.

Script 04 already supplies 5,119 distances across 5,124 planned jobs, retaining
five failures with missing distances. Compare paired successful jobs and report
each condition's failure rates separately; never replace failed distances with
zero. Include all seven subjects and retain LOSO/Single Reference separately.

Script 05 MRI registrations are supplementary evaluation research only. They
must not replace Script 04 distances or be mixed into the primary no-warp
comparison. All seven subjects have now passed Script 05 numerical QA across
the sub-01 pilot and remaining-subject folders; that is not anatomical approval.
The completed reconstruction and prior evaluation outputs remain unchanged.

**Scripts 01–09** are implemented. Preparation, scalp review, BEM interpolation
review, the shared forward build and reconstruction are complete. Reconstruction
finished with 5,119 PASS and five Gamma-MAP FAIL jobs. All 732 LOSO jobs and all
588 sub-03 jobs passed. Script 04 produced 5,119 diagnostic distances, preserving
the five failures; anatomical errors remain unavailable. Script 05 completed
numerical mapping QA for all seven subjects, as supplementary research only.
Script 06 completed the primary comparison and historical baseline context.
Script 07 restyled its nine figures to match the no-warp method years/families.
Script 08 exports no-warp-style summary statistics for the exact Figure 08 data.

## Run Script 01 locally

From the repository root:

```bash
.venv/bin/python cross_subject_fsaverage/template_montage_no_digitization/scripts/01_prepare_experiment.py
```

The default montage candidate is `GSN-HydroCel-256`. `EGI_256` is an explicit
alternative, not an interchangeable net definition. MNE's numbered names alone
cannot establish correspondence with Localize-MI. The default audit therefore
expects **BLOCKED** (exit code 3) until independent evidence is supplied. This is
an intentional scientific gate, not a reason to bypass validation or try
montages until a distance threshold passes.

The script freezes seven subjects, 61 runs, 12 methods and 168 selected
method/owner configurations. It plans 732 LOSO and 4,392 Single Reference jobs
(5,124 evaluations, not unique forward computations). References, training
exclusions, configuration keys and recorded settings must match the frozen
design. The schedule is still written when montage correspondence is blocked,
provided the frozen design itself validates.

## Outputs

Under `outputs/01_prepare_experiment/`:

| File | Contents |
| --- | --- |
| `01_experiment_manifest.json` | Assumptions, input hashes, software versions, policies, counts and limitations |
| `02_planned_reconstruction_schedule.csv` | Exact planned job identities and resolved configurations, if design validates |
| `03_frozen_configuration_inventory.csv` | Snapshot paths, original paths, SHA-256 and sizes |
| `04_template_montage_availability.csv` | Installed EGI/GSN candidates and comparison resources, counts, labels, frames, fiducials and resource hashes |
| `05_channel_mapping_validation.csv` | Documentary mapping validation or explicit missing/ambiguous correspondences |
| `08_readiness_report.json` | BLOCKED or numerical preparation PASS requiring review |
| `README.md` | Output guide |

Only after a candidate's documentary mapping is validated does the script
attempt geometry QA and produce:

- `06_sensor_scalp_geometry_qa.csv`: all mapped electrodes, template/native/HEAD/
  fsaverage MRI positions, scalp vertex distances and local normal offsets.
- `07_coordinate_transform_provenance.json`: explicit transforms, units,
  unchanged numerical gates, template scaling convention and geometry summary.
- `09_fiducial_alignment_qa.csv`: template-to-fsaverage fiducial residuals.

`source/01_montage/` receives unvalidated raw candidate resources and MNE parser
snapshots. With accepted evidence it also receives documentary snapshots,
validated coordinate tables, template/fsaverage-only geometry inputs and nominal
32/64/128 channel lists. No generated coordinate table is supplied by this
implementation before the user executes Script 01.

`source/02_frozen_configurations/` receives byte-exact schedule, selection,
resolved-configuration and policy-code snapshots. The old policy code is
**reference material**; its existing loader/Engine reads individual anatomy and
must not be used unchanged in this variant.

## Scientific interpretation and later stages

MNE applies a predefined 95 mm normalization to its spherical standard montage
resources with `head_size='auto'`. This is an explicit model assumption, not
proof of the raw resource's physical units or a fit to Localize-MI. No scale
search, scalp projection or coordinate optimization is permitted.

The template-native-to-HEAD transform uses template fiducials only; the
HEAD-to-MRI transform is the predefined fsaverage transform. Neither transform
is fitted to a test subject. EGI_256 fiducials are estimated from electrodes by
MNE; HydroCel FidT9/FidT10 labels also require evidence that their interpretation
as the chosen anatomical landmarks is appropriate.

The geometry gates retain the existing median scalp distance <=10 mm and p95
<=20 mm, with rigid-transform tolerance 1e-5. Distances are to scalp vertices
(upper bounds on triangle-surface distance). Local normal offsets do not prove
inside/outside placement. Fiducial residuals and anatomical consistency still
require review, and Script 01 never authorizes automatic forward modeling.
Numerical PASS is not proof of anatomical correspondence.

Montage sizes, average reference before subset selection, target/covariance
windows, bad-channel replacement algorithm and original peak policies are
preserved. The subset and replacement geometry necessarily becomes the common
template rather than a participant forward. Reference identities remain frozen
parameter-transfer identities, not a source of sensor coordinates. Changes in
actual nominal subsets must be disclosed in later comparisons. The known
stimulation hemisphere used by original cortical peak selection is an additional
prior beyond channel identities; it is retained explicitly for comparability.
A future fully blinded peak policy would be a separately labeled sensitivity.

No ground-truth stimulation positions or localization outcomes are read to
choose the montage. Unregistered coordinate distances remain diagnostic rather
than validated anatomical localization errors. Scripts 02 and 03 are described
below; error evaluation is not implemented for this variant yet.

## Evidence and repeat audits

See `source/01_montage/README.md` for the evidence schema. An explicit evidence
run must use new source/output roots after the initial audit:

```bash
.venv/bin/python cross_subject_fsaverage/template_montage_no_digitization/scripts/01_prepare_experiment.py \
  --mapping-evidence /absolute/path/to/evidence.json \
  --source-root cross_subject_fsaverage/template_montage_no_digitization/source_reviewed \
  --output-root cross_subject_fsaverage/template_montage_no_digitization/outputs/01_prepare_experiment_reviewed
```

Optional overrides: `--montage-kind`, `--fsaverage-root`,
`--frozen-design-root` (the original fsaverage preparation's configuration folder,
with the original full plan in its parent), and `--resolved-configurations`.
No existing experiment files or existing snapshots are replaced. Dependencies:
NumPy, pandas, SciPy, MNE and the existing local frozen files; nothing downloads.

## User-authorized exploratory assumption mode

The user subsequently authorized proceeding without author contact using a
clearly documented numbering assumption. Script 01 now supports a separate
`--mapping-policy documented-assumption` mode. The default verified mode and its
previous BLOCKED audit remain intact. See
[the assumption decision and executable command](assumptions/numbered_hydrocel_v1/README.md).

This mode assigns eN to HydroCel EN as **ASSUMED_UNVERIFIED** and allows geometry
inspection in fresh `source_assumed_v1/` and
`outputs/01_prepare_experiment_assumed_v1/` directories. The resource is pinned;
all bijection, coordinate, transform, and scalp-distance checks remain active.
No evidence approval flags are fabricated. Numerical success is labeled
`EXPLORATORY_NUMERICAL_REVIEW_REQUIRED`, rather than confirmation of channel
identity. No contacts are made and no preparation is automatically executed.

## Review the prepared template against the scalp surface

A review-only mode in Script 01 provides a stronger placement check than the
initial nearest-vertex normal offsets. Run locally:

```bash
.venv/bin/python cross_subject_fsaverage/template_montage_no_digitization/scripts/01_prepare_experiment.py \
  --review-preparation-root cross_subject_fsaverage/template_montage_no_digitization/outputs/01_prepare_experiment_assumed_v1 \
  --output-root cross_subject_fsaverage/template_montage_no_digitization/outputs/01_geometry_review_assumed_v1
```

The review reads the existing template coordinates and frozen fsaverage scalp,
verifies source snapshot hashes and transform consistency, and inspects mesh
closure, consistent orientation and connectivity. It computes exact distances
to scalp triangles and classifies placement using generalized winding numbers.
Negative signed distances mean inside, positive outside. A numerical 0.001 mm
surface tolerance and winding tolerance 1e-5 are solely classification settings;
the original median/p95 preparation gates are not changed. Ambiguous points
retain missing signed distances. Mesh self-intersections are not tested.

A fresh output directory receives:

- `01_scalp_overlay_views.png`: four template/scalp views, colored by 3-D
  placement classification, in fsaverage surface RAS coordinates.
- `02_signed_scalp_distance.png`: signed-distance histogram and twelve largest
  boundary separations.
- `03_sensor_triangle_distance.csv`: original coordinates plus triangle
  distances, winding numbers and surface classifications.
- `04_geometry_review.json`: mesh checks, counts, depths, input hashes, software
  versions, previous QA values and interpretation limits.
- `README.md`: output guide.

Both figures use 300 dpi. No electrode is moved or projected onto the scalp;
nearest-surface calculations are distance queries only. No individual anatomy,
digitization, ground-truth locations, forward construction or inverse solution
is used. The preparation audit is not overwritten or upgraded to an anatomical
PASS. The user ran this review locally; it was not executed automatically.

## Audit interpolation on the actual BEM scalp

The completed outer-scalp review found 191 template locations inside the scalp
and 65 outside, with maximum separation about 16.43 mm. These findings remain
placement concerns. MNE's EEG BEM solver selects a nearest triangle and
recomputes unconstrained weights on its plane. These weights may extrapolate
beyond triangle edges. Its BEM scalp mesh can differ from `outer_skin.surf`.
The next audit measures that solver behavior directly without a forward matrix.

Run from the repository root with the existing MNE 1.12.1 environment:

```bash
.venv/bin/python cross_subject_fsaverage/template_montage_no_digitization/scripts/01_prepare_experiment.py \
  --audit-bem-preparation-root cross_subject_fsaverage/template_montage_no_digitization/outputs/01_prepare_experiment_assumed_v1 \
  --output-root cross_subject_fsaverage/template_montage_no_digitization/outputs/01_bem_sensor_audit_assumed_v1
```

This mode verifies preparation snapshots and the recorded BEM hash, reads the
existing three-layer BEM solution, and uses the same geometric interpolation
helper as installed MNE's `_bem_specify_els`. It does not call that solver or
multiply a BEM solution. It checks mesh topology, independently calculates
point-to-triangle distances, and records effective interpolation positions,
barycentric weights and displacement vectors. The 0.001 mm comparison tolerance
checks agreement with an independent triangle-plane projection; it does not
replace the original QA gates. Negative/out-of-range weights, differences from
true nearest-surface distances and effective positions off the finite scalp
surface are recorded as findings. Weights are never clipped or renormalized.

The fresh `outputs/01_bem_sensor_audit_assumed_v1/` folder will contain:

- `01_bem_interpolation_overlay.png`: original and effective sensor locations
  with displacement lines on the actual BEM scalp (300 dpi).
- `02_bem_interpolation_displacements.png`: displacement distribution and
  twelve largest displacements (300 dpi).
- `03_effective_bem_sensor_locations.csv`: original and effective coordinates,
  triangle identities, interpolation weights, distances and classifications.
- `04_bem_interpolation_audit.json`: input/code hashes, topology checks,
  displacement summary, software versions and modeling assumptions.
- `README.md`: output guide.

The effective coordinates are derived audit data; they never replace the stored
montage. The audit assumes one unit-weight monopolar location per EEG channel
and no separate reference-electrode location; average referencing is a separate
operation. It inspects template/fsaverage assets only. Unverified channel
correspondence and original scalp-placement concerns remain explicit. The
expected status is `BEM_INTERPOLATION_AUDIT_REVIEW_REQUIRED`; inspect its results
before deciding any later reconstruction policy. This new mode has only been
syntax-checked and statically inspected, and has not been executed automatically.

The first audit attempt stopped with `Interpolation weights lie outside scalp
triangles`. That check incorrectly assumed MNE clamps its weights to triangle
edges. Script 01 now checks the actual plane-reprojection behavior described
above and reports extrapolation explicitly. The failed attempt created no audit
output directory, so its terminal command can be reused. Original preparation
outputs and anatomical QA thresholds are unchanged.

## Script 02: common template/fsaverage forward

After reviewing the completed BEM audit, the user authorized continuing with
the fixed template coordinates and standard MNE interpolation. Script 02 is
`scripts/02_make_template_forward.py`; it does not reuse the old individual
coregistration or native-forward loaders. All seven subjects, including sub-03,
share one full 256-channel forward because both anatomy and sensor geometry
are common. The existing 732 LOSO and 4,392 Single Reference evaluations retain
their complete identities and frozen parameters. Later inverse processing must
use the frozen template channel-subset and bad-channel replacement policies.

The completed interpolation audit found a 6.705 mm median, 14.770 mm p95 and
16.322 mm maximum displacement. Fourteen electrodes use extrapolated triangle
weights; their effective locations are at most 0.0853 mm from the actual BEM
scalp. These findings do not validate the original template placement or eN->EN
correspondence. Keep all coordinates and weights unchanged and retain the
exploratory interpretation.

Run the build locally from the repository root:

```bash
.venv/bin/python cross_subject_fsaverage/template_montage_no_digitization/scripts/02_make_template_forward.py \
  --phase build \
  --output-root cross_subject_fsaverage/template_montage_no_digitization/outputs/02_template_forward_assumed_v1
```

For provenance/design checks alone, use `--phase audit` with a separate output
directory such as `outputs/02_template_forward_audit_assumed_v1`. Both phases
require fresh output directories and MNE 1.12.1; no downloads occur.

Outputs include `01_forward_manifest.json`, the unchanged full schedule,
per-job forward availability, original sensor coordinates, frozen configuration
copies, the fixed HEAD-to-fsaverage transform and a script snapshot. Build also
writes `forwards/template_fsaverage-fwd.fif`. Forward checks cover channel order,
finite free-orientation leadfields, unchanged HEAD sensor positions, absence of
a separate reference-electrode location, and saved/reloaded forward integrity.
The original fsaverage forward settings remain EEG-only with `mindist=0` mm.
Sampling frequency 1 Hz is a placeholder for the static forward and is not a
preprocessing setting. No BEM construction, individual fitting, EEG processing,
inverse solution or error evaluation is performed.

Successful audit status is `READY_EXPLORATORY_FORWARD_BUILD`; successful build
status is `PASS_EXPLORATORY_FORWARD_NUMERICAL_ONLY`. Failures after output setup
are preserved in the manifest and `failure_traceback.txt`. No existing output
is overwritten or resumed. The user completed this build: 20,484 sources,
256 × 61,452 finite leadfield entries, all 5,124 jobs available, and unchanged
input hashes and frozen schedule.

## Script 03: frozen-parameter reconstruction

`scripts/03_reconstruct.py` reuses only explicitly selected definitions from
the frozen no-warp numerical solver. AST extraction excludes its original
Engine constructor, prepare/forward methods, complete grid/pilot module imports,
and entrypoints. A new EEG-only loader mirrors the frozen loader's sample,
event, timing, baseline and bad-channel behavior. It opens only the four EEG
run files: `epochs.npy`, `channels.tsv`, `epochs.tsv`, and `epochs.json`.
No individual forward, sensor coordinates, contact positions, MRI or individual
transform is opened. The frozen inverse helper module is imported separately;
its numerical processing functions are unchanged.

Average referencing precedes subset selection. The template subsets frozen in
Script 01 are checked against the original geometric algorithm, using original
template coordinates. Bad-channel replacements follow the original reserved
healthy-channel/nearest-unused-channel policy. Known stimulation hemisphere
is inferred from the EEG Description naming convention without loading contact
coordinates. This remains an informed-hemisphere experiment. Dipole methods
retain maximum-GOF peak selection. Parameters are never selected again.

The SESAME capacity-only adapter is copied to
`scripts/sesame_capacity_compat.py`. It retains the previous fsaverage
compatibility behavior: capacity 100→256, unchanged neighborhood radius,
membership, ordering and probabilities. It patches memory only and records
the original function hash; installed packages are never edited.

Run a plan first (streaming SHA-256 reads of all 61 EEG runs, no epoch array
loading, preprocessing or inverse computation):

```bash
.venv/bin/python cross_subject_fsaverage/template_montage_no_digitization/scripts/03_reconstruct.py --phase plan
```

Then run a pilot covering all 12 methods on one LOSO run:

```bash
.venv/bin/python cross_subject_fsaverage/template_montage_no_digitization/scripts/03_reconstruct.py \
  --phase reconstruct --resume \
  --strategy LOSO --test-subject sub-01 --run run-01 --limit 12 --quiet
```

Inspect the pilot before scaling up. The eventual full-run/resume command is:

```bash
.venv/bin/python cross_subject_fsaverage/template_montage_no_digitization/scripts/03_reconstruct.py \
  --phase reconstruct --resume --quiet
```

All commands use `outputs/03_reconstruct_assumed_v1/`. Filters are intentionally
excluded from the immutable run fingerprint, allowing a pilot and later full
run in the same checkpoint. Code, input hashes, dataset path, package versions
and modeling policies must remain unchanged. Both PASS and FAIL records are
preserved; only PENDING jobs run. Use a separate output directory for changed
implementations or diagnostic failure retries. One writer is enforced by a lock.

Outputs contain the full 5,124-row status ledger, full plan, resolved parameters,
strict run fingerprint, EEG input inventory, summary, per-job channel-selection
audit, reused-source snapshots and failure tracebacks. Every attempted fit is
checkpointed. Estimates are recorded in template HEAD mm and fsaverage
MRI/surface-RAS mm. Full source time courses are not saved. These coordinates
are not registered to individual stimulation locations; no localization errors
are calculated. Solver success remains exploratory with unverified physical
channel correspondence and original template-placement limitations.

## Git backup scope

The repository includes scripts, assumption decisions, documentary evidence,
frozen configurations and template resources, completed preparation/geometry/BEM
audit reports and figures, and the shared-forward provenance and availability
tables. The generated forward FIF and active `03_reconstruct_assumed_v1/`
checkpoint directory stay local. The running checkpoint is not a final result
and must not be restored from Git over an active run.

Rebuilding requires the existing Localize-MI EEG dataset, pinned fsaverage
assets and matching Python packages. Manifests retain original absolute paths
and hashes for provenance; they do not make active resume portable to another
machine. Keep code and inputs unchanged when resuming the local frozen run.

## Script 04: completed-run audit and diagnostic evaluation

Run manually from the repository root:

```bash
.venv/bin/python cross_subject_fsaverage/template_montage_no_digitization/scripts/04_evaluate_diagnostics.py
```

This stage streams the frozen inputs for SHA-256 verification without loading
EEG arrays. It checks the full planned identities and configurations, completed
status coverage, finite PASS coordinates, template HEAD/fsaverage MRI transform
consistency, and pair/hemisphere agreement with the existing 61-run audited
ground-truth table. Raw contact and native coordinate-transform files are
hash-checked solely as provenance of that diagnostic reference; they are never
used to construct a forward or fit template geometry. No MNE or ANTs functions
are imported, and no registration or inverse fit runs. A shared lock blocks
concurrent Script 03 writers during evaluation.

The new `outputs/04_evaluate_diagnostics/` directory receives:

| File | Contents |
| --- | --- |
| `01_run_level_diagnostics.csv` | All 5,124 jobs; `diagnostic_distance_mm` for PASS only; anatomical error fields remain invalid/missing |
| `02_ground_truth_diagnostic_reference.csv` | Original 61 audited individual HEAD coordinate references |
| `03_method_summary.csv` | Pooled strategy/method counts, pass rates and diagnostic statistics |
| `04_subject_summary.csv` | Strategy/method/participant summaries |
| `05_reference_summary.csv` | Single Reference summaries by parameter owner |
| `06_equal_subject_summary.csv` | Equally weighted participant medians and pass rates, with available-subject counts |
| `07_run_reference_summary.csv` | One row per test run/method/strategy, aggregating reference evaluations |
| `08_unsuccessful_jobs.csv` | Preserved failed jobs and frozen configurations |
| `09_input_integrity_inventory.csv` | SHA-256, size and role for every verified input |
| `10_anatomical_evaluation_readiness.csv` | Previous registration status and explicit common-frame mapping blocker per participant |
| `11_sub03_summary.csv` | Additional sub-03 summary; sub-03 remains in all-seven summaries |
| `12_evaluation_manifest.json` | Input/output hashes, counts, metric definition, coordinate checks and limitations |
| `script_snapshot.py.txt`, `README.md` | Exact evaluation code and output/interpretation guide |

Expected status is
`DIAGNOSTIC_EVALUATION_COMPLETE_ANATOMICAL_ERROR_BLOCKED`, with 5,119 distances,
five retained failures and **zero anatomically validated localization errors**.
No existing output is overwritten; choose a fresh `--output-root` for reruns.

`diagnostic_distance_mm` is explicitly the Euclidean difference between a
template HEAD estimate and an individual HEAD ground-truth coordinate. Shared
axis names do not establish anatomical correspondence. This is unregistered
coordinate separation and must not be presented as anatomical accuracy or
compared as a validated localization error against baseline/no-warp results.
`localization_error_mm` stays missing and `localization_error_valid` stays False.

Single Reference repeats the same EEG run across six parameter owners; the
reference and equal-participant tables accompany pooled job summaries to expose
that structure. Failures remain in denominators and never become zero errors.
The old individual-coregistration failure for sub-03 is not a reconstruction
exclusion in this common-template variant. Its separate table is additional
reporting rather than an anatomical-quality clearance.

All seven previous MRI-to-fsaverage registrations report `FAIL_REGISTRATION`.
Script 04 inventories those statuses but does not repair or accept a mapping.
Any supplementary anatomical evaluation must independently validate an
evaluation-only ground-truth mapping before calculating anatomical errors.
It is separate from the primary no-registration comparison chosen above.
No individual geometry is introduced into reconstruction by that distinction.

## Script 05: evaluation-only mapping pilot

`scripts/05_map_ground_truth.py` prepares or fits native-MRI-to-fsaverage
registrations for stimulation ground truth. **Individual MRI is privileged
evaluation information only.** The completed EEG/template forward, parameters,
montage and estimates are untouched. No stimulation locations or localization
outcomes select registration settings. Points are mapped only after MRI fitting.
No anatomical errors are computed or approved in this stage.

First audit one subject:

```bash
.venv/bin/python cross_subject_fsaverage/template_montage_no_digitization/scripts/05_map_ground_truth.py \
  --phase audit --subject sub-01 \
  --output-root cross_subject_fsaverage/template_montage_no_digitization/outputs/05_ground_truth_mapping_audit_sub01
```

If that succeeds, run the one-subject registration pilot in a different folder:

```bash
.venv/bin/python cross_subject_fsaverage/template_montage_no_digitization/scripts/05_map_ground_truth.py \
  --phase register --subject sub-01 \
  --output-root cross_subject_fsaverage/template_montage_no_digitization/outputs/05_ground_truth_mapping_sub01
```

The default subject is sub-01, covering **11 stimulation runs**. Repeating
`--subject` selects other subjects; `--all-subjects` explicitly selects all seven
in a fresh output folder. Review the pilot before requesting a full run. Neither
phase overwrites or resumes an existing directory. No command was executed
automatically while implementing the script.

Audit checks Script 04 coordinate hashes, the native bridge audit and source
hashes, native MRI headers, actual H5 transforms, contact provenance and MRI
point domains. The independent native bridge is rederived as
`Norig @ inv(Torig) @ H5_mm`. The fixed fsaverage scanner-to-surface bridge uses
`vox2ras_tkr @ inv(scanner_affine)`. Audit does not call ANTs, resample images,
render figures, load EEG arrays or import MNE.

Register uses the original SyNQuick `-t s` protocol and installed wrapper's
convergence defaults, in double precision. A positive random seed (default 1)
and thread count (default 1) are recorded explicitly; `--random-seed` and
`--threads` override these execution settings. The original 0.1 mm round-trip
gate and one-voxel interior margin remain unchanged. The ANTs executable hashes
and available conda package build metadata are inventoried.

The old failure came from the shell wrapper splitting paths containing spaces.
This script runs the wrapper from a temporary directory using whitespace-free
relative aliases for both MRI inputs and the output prefix. Point commands also
use safe transform aliases. Original files retain their paths. Wrapper exit
code zero alone is insufficient: all affine/warp/inverse-warp/warped images must
exist, and fatal exception text in the log causes a failure.

RAS millimetres are converted to LPS for ANTs point CSVs and back afterward.
The point direction/order follows the
[ANTs documentation](https://github.com/ANTsX/ANTs/wiki/Forward-and-inverse-warps-for-warping-images,-pointsets-and-Jacobians):
moving-to-fixed points use inverse affine followed by inverse warp; the reverse
uses forward warp followed by affine. Full versus staged composition is checked
to 0.001 mm as implementation integrity. The actual displacement-field domains,
native/fixed MRI interiors, original point round-trip tolerance and fsaverage
scanner/surface conversion are checked independently.

Each fresh output folder contains:

- `01_input_inventory.csv`: evaluation-only inputs and hashes.
- `02_registration_status.csv`: all seven subjects; nonselected subjects remain
  `NOT_REQUESTED` rather than being treated as completed.
- `03_mapped_ground_truth.csv`: only requested runs, in native scanner RAS,
  fsaverage scanner RAS and fsaverage surface RAS millimetres, with QA fields.
- `04_mapping_manifest.json`: requested scope, software/protocol, hashes,
  transforms, numerical gates and interpretation limits.
- `source_snapshots/`: script, original reference coordinates, bridge and audit
  provenance snapshots.
- `registrations/sub-01/` (register phase): ANTs outputs, `registration.log`,
  `command.json`, raw LPS point input/output CSVs and point-command logs.
- `qa_mri_checkerboard.png` and `qa_mapped_stimulation.png` inside the subject
  registration directory: whole-brain alignment and post-fit target-region QA.
  `image_qa.json` records descriptive intensity correlation; it is not an
  anatomical acceptance gate.

Expected audit status: `AUDIT_COMPLETE_NO_REGISTRATION`. Expected successful
registration status: `PASS_NUMERICAL_REVIEW_REQUIRED`. All mapped rows and
manifests keep `anatomical_mapping_approved=False`. Inspect whole-brain and
stimulation-region alignment before accepting any mapping; round-trip success
alone can coexist with an anatomically poor registration. Maskface/intact MRI
differences and unchecked deformation folding remain explicit limitations.
The previous failed registrations, Script 04 diagnostics and original results
are retained unchanged.

## Script 06: primary comparison with no-warp, without MRI alignment

Run from the repository root:

```bash
.venv/bin/python cross_subject_fsaverage/template_montage_no_digitization/scripts/06_compare_no_warp.py
```

This script reads only the existing primary evaluation CSVs/JSONs and the
literal palette definitions from Script 26. It does not import or execute
Script 26, open MRI images or transforms, use Script 05 results, read EEG arrays,
rerun inverses, register anatomy or select parameters again. The explicit
comparison policy is in `assumptions/primary_comparison_no_mri_alignment.json`.

All 5,124 jobs match by strategy, method, reference subject, test subject and
run. The original no-warp comparison includes only `scenario=primary` and
`condition=single_reference_no_warp`; within-subject and mixed-peak-policy
sensitivity rows are excluded. Every configuration key and parameter JSON must
match, audited test HEAD ground truth must agree, and paired successes must
have matching epoch/time/preprocessing metadata. Both stored distance metrics
are checked against their saved HEAD estimates and ground-truth coordinates.
Channel selections may differ with the geometry; this is documented rather
than incorrectly requiring identical channel identities.

Current input checks identify **5,109 paired successes**, **10 template-only
successes**, **zero no-warp-only successes**, and **five failures in both**.
All statuses remain in the completion denominators. Paired distance differences
use only identical jobs with PASS in both conditions, while all available PASS
medians are reported in separately named columns. No failed distance is zero
filled and no sensitivity result replaces a primary failure.

Difference is **template distance minus no-warp distance**, in millimetres.
Negative values mean smaller unregistered coordinate separation. They do not
prove improved anatomical accuracy. The median of paired differences is not
the difference of the two condition medians; Script 06 labels these separately.

Outputs are written to a fresh `outputs/06_compare_no_warp/` directory:

| Output | Contents |
| --- | --- |
| `tables/01_paired_jobs.csv` | Full 5,124 matched jobs, statuses, available distances and paired deltas |
| `tables/02_method_summary.csv` | LOSO/Single Reference method summaries, paired and available-case medians, failures |
| `tables/03_participant_summary.csv` | Per-participant paired summaries, including sub-03 |
| `tables/04_reference_summary.csv` | Single Reference summaries by parameter owner |
| `tables/05_equal_participant_summary.csv` | Equal-participant aggregation after reference/run balancing |
| `tables/06_unique_run_summary.csv` | Within-run paired summaries across available reference owners |
| `tables/07_input_inventory.csv` | Input roles and SHA-256 |
| `tables/08_nonpaired_jobs.csv` | All jobs failing in either condition |
| `tables/09_configuration_preprocessing_audit.csv` | Per-job configuration/ground-truth checks and paired EEG metadata checks |
| `tables/10_participant_balanced_run_summary.csv` | Participant medians after collapsing references within runs |
| `figures/01_loso_paired_distributions.png` | Paired no-warp/template distance distributions |
| `figures/02_single-reference_paired_distributions.png` | Same comparison for Single Reference |
| `figures/03_paired_median_differences.png` | Median paired differences by method/strategy |
| `figures/04_participant_paired_differences.png` | Participant profiles with Script 26 method styles |
| `figures/05_failure_counts.png` | Full planned-job failure counts in both conditions |
| `figures/06_sub03_paired_differences.png` | Explicit sub-03 comparison; sub-03 remains in all-seven summaries |
| `REPORT.md`, `comparison_manifest.json` | Comparison report, definitions, counts, provenance and limitations |
| `method_styles.csv`, `script_snapshot.py.txt` | Exact palette/styles and executed code |

Figures default to 300 dpi. `--no-figures` generates tables/report only;
`--output-root` must be fresh. No output overwriting or reconstruction resume
is performed. The expected completion status is
`PRIMARY_NO_REGISTRATION_DIAGNOSTIC_COMPARISON_COMPLETE`.

Single Reference's six parameter-owner evaluations repeat the same recording.
The script therefore supplies per-run, reference and participant summaries
alongside pooled job statistics, without p-values or independence claims.
Equal-participant summaries first take each run's median paired difference
across available references, then the participant median across runs, then give
each available participant equal weight. Coverage and failure counts remain
explicit. Both anatomy and sensor geometry differ between conditions; this
comparison does not isolate anatomy alone.

### Historical 12-method baseline context

Script 06 also includes the original Script 26 `complete_all_pass` selected
baseline by default: **732 unique results, 12 methods × 61 runs**. Override its
location with `--baseline-root`. The selected configuration keys must match the
12 saved baseline definitions. This is the same historical baseline source used
in the previous fsaverage reporting, not a newly fitted baseline.

The baseline uses individual-anatomy within-subject localization errors and
globally selected, in-sample best fixed parameters. No-warp and template values
remain unregistered HEAD-coordinate diagnostics with transferred parameters.
The three conditions therefore supply descriptive context rather than a matched
anatomical-accuracy ranking. Baseline-relative percentage improvements and
mixed-metric paired deltas are deliberately not calculated. The primary
no-warp/template paired deltas remain unchanged.

Before attaching a baseline value, Single Reference is collapsed to one median
per method/test run over the same available successful reference pairs in both
cross-model conditions. All three context distributions then use common test
runs within each method/strategy. The historical baseline is stored only once
per method/run in its source table, rather than duplicated across six reference
owners. Both all-61 baseline medians and common-run baseline medians are labeled
in context summaries. The baseline parameters are preserved separately because
they need not equal the transferred cross-model parameters.

Additional outputs:

- `tables/11_historical_baseline_unique_runs.csv`: the 732 original baseline
  results and explicit metric/selection labels.
- `tables/12_historical_baseline_configuration_definitions.csv`: all 12 saved
  baseline configurations.
- `tables/13_baseline_context_unique_runs.csv`: one context row per
  strategy/method/test run; reference pairs collapsed before baseline attachment.
- `tables/14_baseline_context_method_summary.csv`: all three condition medians
  and IQRs on common runs, plus baseline all-run medians and coverage.
- `tables/15_baseline_context_participant_summary.csv`: participant context,
  including sub-03.
- `figures/07_loso_baseline_context.png` and
  `figures/08_single-reference_baseline_context.png`: three-condition
  distributions with explicit metric labels and hatched baseline boxes.
- `figures/09_baseline_context_by_participant.png`: participant profiles for all
  12 methods across the three conditions and two transfer strategies.

The report now includes these baseline tables and **nine figures** in total.
Colors and markers still follow Script 26. The helper
`scripts/comparison_baseline_context.py` is hash-pinned and copied into the
comparison output. No reconstruction, MRI alignment or additional fitting is
required to generate this context.

## Script 07: no-warp figure presentation

The nine existing comparison images now use the exact method order, year
labels, and family grouping from `cross_subject_no_warp/scripts/09_visualize_results.py`.
This preserves its historical display-year convention without asserting those
dates are each method's first invention. Script 26 colors and marker identities
remain consistent across the comparison figures. Family bands label the method
axes, and participant plots have legends grouped by family with method years.

Images remain in the same `outputs/06_compare_no_warp/figures/` directory.
Original images and their comparison manifest are backed up in
`outputs/06_compare_no_warp/figures_before_no_warp_style/`. All numerical analysis
tables, values, and `REPORT.md` are byte-identical. `figure_method_metadata.csv`
records the presentation labels/order and `figure_style_manifest.json` records
style/code hashes and output integrity; the comparison manifest tracks the
updated image hashes. The metric labels and no-MRI primary policy are retained.

After generating a new Script 06 report, apply this presentation with:

```bash
.venv/bin/python cross_subject_fsaverage/template_montage_no_digitization/scripts/07_style_figures_like_no_warp.py
```

To regenerate already styled figures, add `--overwrite`. Rendering uses saved
tables only: no parameter selection, reconstruction, anatomy mapping or
distance/statistical recomputation. All nine images are staged before replacing
the existing images. DPI defaults to 300.

## Script 08: the Figure 08 "00" statistics table

Figure `08_single-reference_baseline_context.png` uses
`tables/13_baseline_context_unique_runs.csv`, filtered to
`strategy == Single-reference` and `common_run_available == True`.
The plotted columns are `baseline_localization_error_mm`,
`no_warp_run_diagnostic_mm`, and `template_run_diagnostic_mm`.

`tables/00_figure08_single_reference_statistics.csv` has the same columns as
no-warp's `00_master_three_strategies_statistics.csv`:
`method, strategy, N, Mean_mm, Median_mm, SD_mm, Q1_mm, Q3_mm, IQR_mm, Min_mm, Max_mm`.
It contains 36 rows: 12 methods × Baseline/No-warp/Template-fsaverage.
Here `strategy` names the three plotted conditions, all within the Single
Reference transfer view. Values are rounded to three decimal places, with
sample standard deviation (`ddof=1`) and linear quartiles.

For this figure, **N=61 unique runs per method/condition**, rather than 366
reference jobs: no-warp/template values are already the per-run medians across
the same paired successful reference evaluations. This matches the boxes in
Figure 08. `tables/00_figure08_plot_values.csv` retains all 2,196 exact plotted
values with method, condition, test subject, run and metric labels.
`tables/00_figure08_statistics_manifest.json` records source/filter definitions,
hashes, rounding and reduction policies. No existing tables or figures change.

Regenerate the export with:

```bash
.venv/bin/python cross_subject_fsaverage/template_montage_no_digitization/scripts/08_export_figure08_statistics.py --overwrite
```

The combined LOSO + Single Reference table is
`tables/00_master_three_conditions_statistics.csv`: 72 rows (12 methods ×
three conditions × two transfer strategies). `transfer_strategy` identifies
LOSO or Single-reference; `strategy` identifies Baseline, No-warp or
Template/fsaverage. It uses the exact common-run distributions from Figures
07 and 08. Most N values are 61; the LOSO Gamma-MAP common cohort has 60 runs.
Exact values and export provenance accompany the combined table.

```bash
.venv/bin/python cross_subject_fsaverage/template_montage_no_digitization/scripts/08_export_figure08_statistics.py --both-strategies --overwrite
```

## Clear template-only distance exports

`scripts/09_export_template_distances.py` verifies and exports distances for the
completed fsaverage reconstruction with fixed HydroCel template positions and
no test-subject electrode placement. It recomputes the saved unregistered HEAD
distances and checks agreement with Script 04; no reconstruction or MRI mapping
is performed. The primary metric and exploratory mapping label remain unchanged.

Inside `outputs/06_compare_no_warp/tables/`:

- `00_template_fsaverage_distances.csv`: all 5,124 jobs, 5,119 distances and five
  failures with missing distances; explicit no-test-electrode/no-alignment flags.
- `00_template_fsaverage_method_summary.csv`: all 12 methods, separately for
  LOSO and Single Reference, with counts, mean/median and quartiles (24 rows).
- `00_template_fsaverage_distance_export.json`: metric definition and hashes.

These are coordinate diagnostics, not anatomically validated localization
errors. The reference ground-truth coordinates are retained from the existing
audit; no individual electrode positions are introduced into reconstruction.
The export script refuses existing files rather than overwriting them.

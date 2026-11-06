# Fsaverage with template electrodes and no individual digitization

This separate Localize-MI variant assumes test EEG and known channel identities,
without test MRI, measured electrode positions or individual coregistration.
It transfers the frozen LOSO and Single Reference configurations to fsaverage
anatomy and a documented, common template cap. Both anatomy and sensor geometry
change relative to the earlier no-warp condition; this is not an anatomy-only
comparison.

**Scripts 01–03** are implemented. Preparation, scalp review, BEM interpolation
review and the shared forward build are complete. Script 03 passed its
12-method pilot and the full reconstruction is running. Final results and
localization-error evaluation are not yet available.

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

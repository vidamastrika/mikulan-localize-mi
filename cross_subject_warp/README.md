# LOSO and single-reference anatomical warp experiment

Branch: `experiment/loso-single-reference-warp`.

## Current stage

Scripts 01–05 are implemented. The existing native LOSO and reference-only
primary/sensitivity datasets are imported with verified input fingerprints and
explicit metadata provenance; see [native data completion](docs/NATIVE_DATA_COMPLETION.md).

The user-run coordinate bridge check passed all seven subjects. Registration
QA passed all 42 affine and 36/42 SyN pairs. Each of the six failed SyN pairs has
one sampled point above the fixed 0.1 mm roundtrip threshold in one direction.
Those pair flags remain unchanged.

Script 06 mapped evaluation and the separate 0.15 mm QA sensitivity analysis
have been run by the user. The original 0.1 mm QA flags and primary results
remain unchanged. Scripts 07 and 08 have been run by the user. A strategy-key
join error was found in Script 07's LOSO versus single-reference table and
fixed; regenerate Script 07 into a fresh review folder before visualization.
Script 09 is implemented and synthetically checked but has not been run on
experiment outputs. Terminal commands are in the opening comments and
[runnable stage instructions](scripts/README.md). No new registration fitting or inverse
reconstruction was run;
`cross_subject_no_warp/` and `cross_subject/` originals are preserved.

Start with [the next steps](docs/NEXT_STEPS.md). The detailed
[repository audit](docs/repository_audit.md) explains the experimental design,
selection rules, dependencies and coordinate concern.

## What has been copied

| Folder | Contents |
|---|---|
| `source/00_no_warp_workflow/scripts/` | All nine historical main scripts, finalizers, sensitivities and the pre-patch archive |
| `source/01_registration_resources/scripts/` | Prior affine/SyN registration and coordinate-validation code |
| `source/02_parameter_grids/` | Frozen raw grids, preparation plans and original manifest |
| `source/03_solver_code/` | Frozen solver implementation and original code manifest |
| `source/07_historical_baseline/` | Historical selected definitions, method summary and 732 run-level results |
| `source/provenance_manifest.json` | Original branch, commit, path, SHA-256 and size for each copied historical file |

Copied Python files are historical references, unchanged byte-for-byte. They
retain their original path defaults and may write into historical experiment
folders if executed. **Do not run those copies directly.** New runnable stages
will live in `scripts/` after their paths and behavior have been adapted and
reviewed. See [runnable stage instructions](scripts/README.md). The distinction is intentional: an old no-warp script is not a warp
implementation merely because it was copied into this directory.

The two large parameter grids use Git LFS. After checking out this branch, use
`git lfs pull` if they still contain small pointer files instead of CSV data.
Copies preserve the original object contents; LFS stores identical content by
object ID, so the new paths do not require distinct data objects.

## Planned nine-stage structure

| Script under `scripts/` | Output folder under `outputs/` |
|---|---|
| `01_prepare_experiment.py` | `01_prepare_experiment/` |
| `02_select_loso_parameters.py` | `02_select_loso_parameters/` |
| `03_run_localization.py` | `03_reference_based/`, followed by `03_reference_based_final/` |
| `04_select_single_reference_parameters.py` | `04_single_reference_parameters/` |
| `05_run_single_reference.py` | `05_single_reference_reconstruction/`, followed by `05_single_reference_final/` |
| `06_evaluate_localization.py` | `06_evaluate_localization/` |
| `07_compare_strategies.py` | `07_compare_strategies/` |
| `08_reference_sensitivity.py` | `08_reference_sensitivity/` |
| `09_visualize_results.py` | `09_visualize_results/` |

Retain numbered artifact names from the historical workflow and explicitly add
the registration mode to mapped evaluation rows and keys. Preserve primary and
sensitivity datasets separately. Do not silently change selection, montage,
solver or peak-extraction policies.

## What is not included

Local reconstruction outputs, MRI/EEG data, BEM solutions, affine matrices,
deformation fields and QA outputs are not uploaded with the code. Script 01
records their actual local availability and copies registration resources into
the ignored resource directory; inspect its generated provenance manifest. See
[local resource import](docs/LOCAL_RESOURCES.md).

## Registration policy

Reuse valid existing MRI-only registrations before considering any new fitting.
LOSO changes parameter selection; it does not require a new registration for
the same subject pair and unchanged MRIs. Need all 42 ordered non-self pairs to
support all-reference analyses. The LOSO primary/fallback reference subset is
covered by that larger inventory.

Affine and SyN map an already reconstructed reference source point into test
anatomy at evaluation. They share the same inverse fits. Do not warp EEG,
lead fields or solver geometry for this comparison. Keep reconstruction,
registration and evaluation status separate.

The previous code labels its head/surface-to-NIfTI RAS bridge provisional.
Validate that bridge and registration QA before claiming anatomical accuracy.
Missing local resources must remain explicit; they do not authorize expensive
reconstruction or registration during preparation.

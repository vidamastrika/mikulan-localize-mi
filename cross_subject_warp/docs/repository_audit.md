# Localize-MI cross-subject warp audit

## Scope and evidence

Primary branch: `experiment/loso-single-reference-no-warp`, commit `cf89dd2b0257e302fb5de1d5c280b5b074e67aad`.
Comparison branch: `experiment/cross-subject-anatomy`, commit `b48d8d41f09a6515f7d88f546cc524762e5556e7`.

The earlier audit inspected the wrong branch. The requested experiment is present on the specified branch. This report supersedes the earlier conclusions about unavailable experiment code.

Branches were fetched and exported to `/tmp/localize-audit/no-warp/` and `/tmp/localize-audit/anatomy/` for inspection. The working checkout was not switched. No experimental source was edited, no experiment directory was created, and no inverse reconstruction or registration was executed. This report is the only new workspace deliverable. Downloading historical CSVs is not executing their experiments.

Two grids are Git LFS objects, not ordinary Git blobs: `inputs/01_grid_results.csv` (134,537,163 bytes) and `inputs/06_23_all_next_methods_combined.csv` (66,444,055 bytes). Their objects were fetched, SHA-256 verified, and materialized only in the audit export. All 9 files listed by the grid snapshot manifest and all 31 files listed by the solver snapshot manifest match their recorded hashes.

Tracked source does not establish that local experiments completed. Counts hard-coded in finalizers below are historical expectations, not observed reconstruction results in this cloud instance.

## 1. Exact script and directory inventory

All paths below are relative to the project root on the primary branch. Actual inspection files are under `/tmp/localize-audit/no-warp/` plus the relative path.

| Main script under `cross_subject_no_warp/scripts/` | Actual default output under `cross_subject_no_warp/outputs/` | Function |
|---|---|---|
| `01_prepare_experiment.py` | `01_prepare_experiment/` | Schedule, folds, rigid-coordinate audit |
| `02_select_loso_parameters.py` | `02_select_loso_parameters/` | Training-only LOSO selection from frozen grids |
| `03_run_localization.py` | `03_reference_based/` | Patched paired within/reference reconstruction |
| `04_select_single_reference_parameters.py` | `04_single_reference_parameters/` | Reference-only parameter selection |
| `05_run_single_reference.py` | `05_single_reference_reconstruction/` | All-reference cross reconstruction |
| `06_evaluate_localization.py` | `06_evaluate_localization/` | Primary/sensitivity evaluation and coordinate validation |
| `07_compare_strategies.py` | `07_compare_strategies/` | Historical/LOSO/single-reference master comparison |
| `08_reference_sensitivity.py` | `08_reference_sensitivity/` | Variation across six references |
| `09_visualize_results.py` | `09_visualize_results/` | PNG plots and manifest |

Use executable defaults, not stale comments: Script 03's docstring mentions `03_run_localization_v2`, and Script 01's README template lists obsolete downstream folder names. Actual downstream canonical results are `03_reference_based_final` and `05_single_reference_final`.

Additional scripts inspected:

* `03_run_localization_before_patch.py`: archived pre-patch implementation. The patch changes `all_good` to physically pick good channels after average referencing, adds sparse-STC debugging, and enables separate-folder retries of original FAIL keys.
* `03_finalize_reference_based.py`: merges original PASS records with patch records for exactly the original FAIL keys; checks identities, paired EEG fingerprints and channel provenance. Output `03_reference_based_final/`.
* `03_wholebrain_gamma_sensitivity.py`: separate whole-brain peak extraction for one Gamma-MAP failure; output `03_gamma_wholebrain_sensitivity/`.
* `05_finalize_single_reference.py`: validates all single-reference attempts and selections without repairing failures; output `05_single_reference_final/`.
* `05_wholebrain_hemisphere_sensitivity.py`: retries only exact empty-hemisphere failures with whole-brain peak extraction; output `05_wholebrain_hemisphere_sensitivity/`.

The prior registration/coordinate lineage is under `cross_subject/scripts/`: 03 (registration pilot), 08 (readiness), 10–16 (image/surface/T1w/point-frame checks), 17–18 (affine SESAME/quality), 19 (SyN), 20–21 (SyN points/local anatomy), 27–32 (batch reconstruction, registration and evaluation), 34–36 (other inverse methods), and 39–40 (earlier LOSO selection/evaluation). Scripts 31, 19 and 32 are the clearest reusable registration batch, SyN command and point-evaluation implementations.

`cross_subject/` is almost identical between the two branches. Differences are Script 36 and three additional files on the anatomy branch: `36_merge_four_methods.py`, `37_compare_cross_subject_methods.py`, `38_reference_sensitivity_statistics.py`. The registration implementations 19/31/32 are identical between the audited branches.

## 2. Design and final dependency graph

The frozen preparation schedule contains exactly 61 runs: sub-01=11, sub-02=6, sub-03=7, sub-04=9, sub-05=14, sub-06=7, sub-07=7.

Methods: MNE, dSPM, sLORETA, eLORETA, Continuous-ECD, RAP-MUSIC, LCMV, MxNE, irMxNE, Gamma-MAP, Champagne, SESAME.

LOSO means parameter training on six subjects. It does NOT mean reconstruction with six reference anatomies. Script 01 chooses reference sub-01 for every test except sub-01, which uses fallback sub-02. This gives 61 run/reference cases, 84 method/fold selections, and 1,464 planned fits: 12 × 61 × two conditions (`within`, `single_reference_no_warp`). Both conditions use the same fold configuration and identical selected EEG samples.

Single-reference selection uses only the reference subject's historical runs. Seven references × 12 methods gives 84 selections. Each test run is paired with its six other references: 366 run/reference pairs, 42 ordered subject pairs, 4,392 cross fits. Script 05 does not compute its own within-subject baseline.

Dependency graph:

1. Original Script 01 needs `cross_subject/outputs/30_all_reference_sesame_batch/experiment_schedule_all_references.csv`, plus dataset forward/H5/contact files. It validates 366 schedule rows and 61 distinct runs. Only identifiers are reused, not old errors or parameters.
2. Script 02 uses frozen `source/02_parameter_grids/`, which already contains copies of the fold/case plan. A fresh snapshot would otherwise need local original-grid outputs, additional-method grids, MxNE extension results and the next-method master CSV.
3. Script 03 requires complete Script 02 selections, source manifest, verified Script 01 coordinate audit, frozen solver snapshot, dataset, and BEM solutions for Continuous-ECD.
4. Script 03 finalization requires original and patch execution records. Its resulting `03_reference_based_final/01_execution_status.csv` is the canonical LOSO input.
5. Script 04 uses the same frozen grids and selection routine; it writes the 84 selections and 366-row reference/test schedule.
6. Script 05 imports patched Script 03 and requires Script 04 selections/manifests, source snapshots, preparation audit, dataset and BEMs. Its finalizer produces canonical `05_single_reference_final/01_execution_status.csv`.
7. Script 06 requires both primary final datasets AND both sensitivity variants, even when merely evaluating the primary experiment. It also needs actual dataset ground truth and H5 transforms for evaluation mode.
8. Script 07 consumes Script 06 primary/sensitivity tables and the tracked 732-row historical baseline. Script 08 consumes Script 06 and optionally Script 07 baseline context. Script 09 consumes Script 07 master rows and Script 08 reference summaries/matrices.

### Principal numbered artifacts to retain

* 01: `01_loso_folds.csv`, `02_fold_membership.csv`, `03_single_reference_cases.csv`, `04_reconstruction_plan.csv`, `05_coordinate_audit.csv`, `06_coordinate_transforms.json`, `07_experiment_manifest.json`.
* 02: `01_loso_selected_parameters.csv`, `02_fold_selection_coverage.csv`, `03_training_configuration_audit.csv`, `04_training_primary_value_audit.csv`, `05_training_duplicate_audit.csv`, `06_parameter_fields.json`, `07_selection_manifest.json`, `08_source_manifest.json`.
* 03/05 reconstruction: `01_execution_status.csv`, `02_reconstruction_plan.csv`, `03_configuration_preflight.csv`, `04_failure_tracebacks.log` when needed, `05_resolved_configurations.json`, `06_frozen_run_manifest.json`, `07_execution_summary.csv`, `09_channel_selection_audit.csv`, `10_channel_selection_summary.csv`. Script 03 also writes `08_paired_input_audit.csv` and retry provenance `11_retry_source.json` when applicable.
* 04 mirrors Script 02's selection numbering with reference-specific names and adds `09_reference_test_schedule.csv`.
* 06: `01_input_validation.csv`, `02_coordinate_validation.csv`, `03_primary_localization.csv`, `04_sensitivity_localization.csv`, `05_method_summary.csv`, `06_failure_analysis.csv`, `07_evaluation_manifest.json`, `08_raw_head_diagnostic.csv`.
* 07: `01_master_run_level.csv`, `02_master_method_summary.csv`, `03_master_overall_summary.csv`, `04_baseline_vs_loso_within.csv`, `05_baseline_vs_loso_within_summary.csv`, `06_cross_no_warp_pair_summary.csv`, `07_validation_report.json`.
* 08: `01_reference_run_level.csv`, `02_reference_method_summary.csv`, `03_test_run_sensitivity.csv`, `04_method_sensitivity_summary.csv`, `05_reference_pairing_matrix.csv`, `06_primary_vs_sensitivity.csv`, `07_baseline_context.csv`, `08_validation_report.json`.
* 09: actual code writes `00_master_three_strategies_boxplot.png`, `01_within_boxplot.png`, `02_cross_boxplot.png`, `03_pass_rate_grouped.png`, `04_reference_sensitivity.png`, `05_reference_by_method_heatmap.png`, `06_test_reference_heatmap.png`, `07_plot_manifest.json`. PNG only, 300 dpi in plotting calls. Some top-of-file documentation omits the extra `00` figure.

## 3. Exact configuration-selection policy

The default is `primary-all-pass`, not merely a winner with successful rows:

1. Filter to training subjects BEFORE setting repair, failure screening, duplicate conflict detection or scoring. LOSO excludes the test subject; reference-only selection includes just the reference.
2. Normalize numeric/boolean values; configuration identities include controlled settings and recorded fixed time windows. Hash normalized JSON into a deterministic key.
3. Repair missing settings only where matching TRAINING rows from the same source/configuration/montage identify one unambiguous value. Primary settings may be recovered from explicitly encoded failed IDs. No held-out borrowing or guessed solver defaults.
4. Candidate keys must cover exactly the expected training subject/run set. Every row must be PASS with finite nonnegative error and complete settings. Collapse exact duplicate status/error observations; conflicting duplicates invalidate the configuration.
5. If any observed configuration at a given primary value fails completeness/reliability, block that primary value. Primary fields: ECD minimum distance; MxNE alpha; irMxNE alpha; Gamma alpha; Champagne pruning threshold; SESAME fs. Minimum-norm, RAP-MUSIC and LCMV have no extra primary-value screen.
6. Unresolved failed settings conservatively block their identifiable primary value even under optional `configuration-all-pass`. Unresolvable successful settings stop selection.
7. LOSO score is the mean of the six subject-specific median errors. Ties: pooled median, pooled mean, configuration key.
8. Reference-only score is that subject's median error. Ties reduce to mean, then key.
9. Require all 84 selections before reconstruction. Missing winners remain explicit, exit nonzero, and never fall back to global selections.

The later-method `source_experiment` allowlists explicitly exclude Gamma depth-test experiments. Allowed series are the base next-method grid; irMxNE alpha15/20 extensions; Gamma alpha, low-alpha and xyz-true extensions; Champagne pruning extension; SESAME fs extension. These must remain unchanged for a controlled warp comparison.

Historical Script 26 ranks globally selected all-PASS configurations by pooled configuration median, then primary-parameter overall median, then configuration mean (remaining exact ties follow its deterministic dataframe ordering). This is different from LOSO subject-balanced scoring. Historical baseline selections are in-sample descriptive comparators, not valid LOSO winners.

### Frozen grid contents

All montage choices are `all_good`, `128`, `64`, `32`. Reported ranges below are observed settings, not a promise of a complete Cartesian grid.

| Snapshot input | Rows | Observed settings |
|---|---:|---|
| `01_grid_results.csv` | 390,400, all PASS | Four minimum-norm methods; loose/depth 0.1…1.0 by 0.1; SNR 1/2/3/4; lambda² 1/SNR² |
| `02_grid_results.csv` | 1,464, all PASS | ECD min distance 1/2.5/5/7.5/10/15 mm |
| `03_grid_results.csv` | 244, all PASS | RAP-MUSIC one dipole |
| `04_grid_results.csv` | 10,980, all PASS | LCMV reg 0/.01/.05/.1/.2; covariance half-window 2/5/10 ms; empirical/shrunk/diagonal_fixed; max-power; unit-noise-gain-invariant |
| `05_grid_results.csv` | 48,800, all PASS | MxNE alpha 25/35; loose/depth 0.1…1; one iteration |
| `06_23_all_next_methods_combined.csv` | 132,157; 131,739 PASS, 418 FAIL | Raw four-method master; allowlist filtering applies during selection |

Allowed irMxNE rows: 122,000; alpha 15/20/25/30/35; loose/depth 0.1…1; 10 iterations; seed 0. Allowed Gamma rows: 6,241; alpha .1/.2/.25/.3/.35/.4/.45/.5/.7; loose .3/.4/.5/.7/1; depth .1/.5; xyz grouping true/false; update 1; maxit 2,000 or 10,000; tolerance 1e-4 or 1e-6. These combinations come from different source experiments and must not be recombined into an invented grid.

Champagne: 1,220 rows; alpha auto; prune true; thresholds .0001/.001/.005/.01/.05; convergence 1e-8; max_iter 2,000. SESAME: 2,684 rows; fs .03/.06/.09/.12/.15/.18/.21/.24/.27/.30/.33; 100 particles; max one dipole; lambda .25; hyper_q true; seed 0. Failed/incomplete evidence is preserved, not dropped before screening.

### Available historical selected configurations

The tracked definitions have 12 configurations, and the long table has exactly 732 unique method/subject/run rows, all PASS. Recomputed method medians match the tracked summary exactly.

| Method | Montage | Historical global winner | Median mm |
|---|---|---|---:|
| MNE | 32 | loose .6, depth .2, SNR 1 | 24.575140 |
| dSPM | 128 | loose .9, depth .6, SNR 1 | 18.679919 |
| sLORETA | 128 | loose .5, depth .6, SNR 1 | 19.894705 |
| eLORETA | 128 | loose .1, depth .1, SNR 1 | 21.380409 |
| Continuous-ECD | 128 | minimum distance 15 mm | 17.973080 |
| RAP-MUSIC | 128 | one dipole | 18.627682 |
| LCMV | 32 | reg 0; half-window 10 ms; diagonal_fixed; max-power; invariant weight norm; reduce_rank=1 | 16.401428 |
| MxNE | 128 | alpha 35; loose .7; depth 1; one iteration | 17.597183 |
| irMxNE | 128 | alpha 25; loose .6; depth 1; 10 iterations; seed 0 | 17.597183 |
| Gamma-MAP | 64 | alpha .35; loose .7; depth .5; grouped xyz; update 1; maxit 2,000; tol 1e-4 | 22.881264 |
| Champagne | 32 | alpha auto; pruning .001; prune true; convergence 1e-8; max_iter 2,000 | 24.054721 |
| SESAME | all_good | fs .12; 100 particles; max one dipole; lambda .25; hyper_q true; seed 0 | 14.196784 |

These are available historical definitions, NOT the absent 84 LOSO or 84 reference-only winners.

## 4. Preprocessing, montage and inverse implementation

Use `source/03_solver_code/src/localize_mi/{io,inverse,metrics}.py` and the frozen Script 09/19 sources. The five snapshotted package modules match the corresponding top-level `src/localize_mi/` files on the audited primary branch.

The experiment loads released epochs, not raw recordings. `load_run` reads `_epochs.npy`, `_channels.tsv`, `_epochs.tsv`, `_epochs.json`, task `_coordsystem.json`, and the test forward FIF. Sampling rate comes from channels; tmin is minus events `zero_time`. Arrays/events/channel dimensions and channel ordering are checked. Released data already had baseline correction; construct EpochsArray with `baseline=None` and do not baseline-correct again. No new filtering, resampling, epoch rejection or alignment is performed in these reconstruction scripts.

Montage policy:

* Use reference forward sensor geometry to derive nominal subsets for both LOSO paired conditions and for all single-reference cross fits.
* Greedy farthest-point selection starts at greatest Z, ties by lexicographic channel name. Each subsequent sensor maximizes its distance to the closest chosen sensor. The first 32/64/128 form nested subsets; require 256 template sensors.
* Good/bad flags always come from test EEG. Reserve all healthy nominal channels first, then replace bad nominal channels by closest unused good channel in reference geometry, ties by name. Preserve returned channel order.
* `all_good` uses all good test channels, with the patched physical channel pick.
* Set electrode positions from the current model's forward (test for within, reference for cross). Average-reference all loaded epochs before selecting the subset; MNE excludes marked bads from the reference calculation.
* Paired within/cross runs must have identical data/time hashes, channel lists, sample rate, epoch count and baseline metadata. Historical grids used fixed sub-01 nominal montage, so transferred configurations are not newly optimized for reference-specific geometry.

Noise covariance: auto estimator on [-.250,-.050] s. Evoked: average epochs, crop [-.002,+.002] s, including tmax. Configuration enrichment rejects incompatible recorded windows or ambiguous settings.

Solver behavior:

* Four minimum-norm methods use shared MNE make/apply inverse; loose/depth selected, lambda²=1/SNR², scalar `pick_ori=None`.
* MxNE/irMxNE use MNE mixed_norm, selected alpha/loose/depth/iterations; rank=None, scalar STC, residual requested; default random state 0 where not explicitly selected.
* Gamma-MAP uses MNE gamma_map with recorded alpha/loose/depth/xyz grouping/update/maxit/tol; rank=None, scalar STC, residual requested.
* LCMV computes data covariance over its selected symmetric half-window and estimator; uses selected reg/orientation/weight norm/reduce_rank plus the shared noise covariance.
* ECD uses the model subject BEM, forward MRI/head transform, selected minimum distance; RAP-MUSIC requires one dipole. Both choose time/location at maximum GOF, without a hemisphere restriction.
* Cortical STCs choose maximum absolute amplitude in the known stimulation hemisphere; SESAME chooses the corresponding posterior-map peak. Empty/nonfinite/all-zero solutions fail. MRI-frame source coordinates are explicitly converted to head frame, otherwise head frame is required.
* SESAME noise_std = selected fs × maximum absolute target EEG amplitude; noise_cov=None; recorded particles/max dips/lambda/hyper_q/seed. Retain the Particle compatibility wrapper and reject zero-source results.
* Champagne dynamically checks the invertmeeg Solver API, supplies all recorded settings, picks forward channels in evoked order, supplies noise covariance only if supported, and retains the MNE projector compatibility wrapper. Do not silently switch solver implementation or defaults.

Reconstruction returns peak coordinates in model-head mm, time and vertex, not a persisted full STC. Ground-truth coordinates are not solver arguments, but current code loads contact coordinates to derive stimulation metadata/hemisphere; it is not literally a no-ground-truth-file-access pipeline.

## 5. Failure handling and final evaluation

One writer per output directory; atomic checkpoints; resume retains both PASS and FAIL and does not automatically retry. Changed code/configuration manifests require a new output folder. Failures retain reasons/tracebacks and have no estimated coordinates. PASS means a valid solver output was extracted, not necessarily optimizer convergence.

LOSO finalizer hard-codes original 1,361 PASS/103 FAIL and patch coverage of those 103 FAIL keys, expecting final 1,463 PASS/1 FAIL. The remaining expected key is Gamma-MAP/sub-05/run-08/cross with no vertices in the requested hemisphere. Single-reference finalizer expects 4,378 PASS/14 FAIL out of 4,392 attempts. These numbers cannot be independently verified without the local output CSVs.

Sensitivity is separate from primary: LOSO replaces exactly that one Gamma peak with a whole-brain extraction; single-reference replaces exactly three empty-hemisphere irMxNE failures. No-active-dipole Gamma failures and zero-source SESAME failures remain failures. This is a mixed peak policy, not a uniform whole-brain experiment. Scripts 06–09 enforce these specific counts, so cannot accept arbitrary new outcomes unchanged.

Ground truth is the arithmetic midpoint of the two named SEEG contacts in test-native surface coordinates, in metres. Convert via inverse of test head-to-surface H5 to test head, then multiply by 1,000. Within error is Euclidean distance in test-head mm. Cross model-head coordinates are deliberately NOT treated as anatomically registered: `localization_error_mm` is blank for cross; `raw_head_separation_mm` is diagnostic only.

Script 06 produces 5,856 primary rows and 5,856 sensitivity rows. Script 07 adds 732 historical rows, yielding 12,444 master rows. It pairs historical vs LOSO-within on method/test/run and LOSO-cross vs reference-only-cross on method/reference/test/run (732 matched cross rows). Statistics are descriptive mean/median/sample SD/quartiles/IQR/min/max, not inferential tests.

Script 08 retains 4,392 rows per scenario, 84 reference/method summaries per scenario, 732 test-run/method sensitivity groups per scenario, and 504 directed-pair/method summaries per scenario. It reports sample SD, range, CV and success across six references. Its pooled-raw summary actually summarizes test-run medians. Failed cases remain in denominators but have no distance contribution. Reference anatomy and reference-selected parameters both vary; this is not an isolated anatomy effect.

## 6. Affine/SyN implementation and coordinate concern

### Registration fitting

Script 31 schedules all 42 ordered non-self pairs. Fixed = TEST MRI; moving = REFERENCE MRI. Inputs are `derivatives/epochs/sub-XX/anat/sub-XX_mod-T1w_maskface.nii`. Registration uses MRI only, no EEG or stimulation-based selection.

Affine: rigid then affine, step .1; MI weight 1, 32 bins, Regular .25; convergence [1000x500x250x0,1e-6,10]; shrink 8x4x2x1; smoothing 3x2x1x0 vox. 3D float, linear interpolation, histogram matching off, winsorization [.005,.995], center initialization `[fixed,moving,1]`, collapsed transforms.

SyN: Script 19 independently runs the same rigid+affine stages then SyN[.1,3,0], CC weight 1/radius 4, convergence [100x70x50x20,1e-6,10], same shrink/smoothing. It does not simply append deformation to the separately saved affine-only result; retain its own affine component.

Expected resources:

* Affine: `cross_subject/outputs/09_anatomical_registration/{ref}_to_{test}/{ref}_to_{test}_0GenericAffine.mat` and `{ref}_in_{test}.nii.gz`.
* SyN: `cross_subject/outputs/19_syn_registration/{ref}_to_{test}/{ref}_to_{test}_syn_0GenericAffine.mat`, `..._1Warp.nii.gz`, `..._1InverseWarp.nii.gz`, `donor_syn_in_target.nii.gz`, inverse warped image, quality JSON and logs.
* Batch status: `cross_subject/outputs/31_all_reference_registration/registration_status.csv`.

Scripts check image grids/existence and foreground QA, not proof of cortical alignment. SyN completion status explicitly says point transforms are not yet validated. All required files remain unavailable locally and untracked: zero pairs can be certified ready from this audit.

### Point mapping, exactly as implemented

Script 32 flips RAS mm to LPS mm with diag(-1,-1,1), calls antsApplyTransformsToPoints, and flips back. Its ordered `-t` lists are:

| Point direction | Affine | SyN |
|---|---|---|
| Reference to test | `[affine,1]` | `[syn_affine,1]`, `InverseWarp` |
| Test to reference | `affine` | `Warp`, `syn_affine` |

These are point-transform conventions, not the similarly named image-resampling sequence. Preserve the tested order; do not infer direction from a filename alone.

QA defaults: 200 deterministic foreground MRI samples (seed 2901), at least 75% inside the moving-image interpolation region, warped-image grid affine tolerance .02, intensity reproduction RMSE <= .015, source-point roundtrip <= .1 mm. Within-grid vs electrode midpoint discrepancy must be <= .1 mm. These test implementation consistency, not anatomical truth; intensity RMSE is not a spatial distance.

**Critical unresolved bridge:** Script 32 converts reference head to surface with H5, then applies inverse H5 to obtain a variable called reference T1w. Those operations cancel numerically. It then treats that value as NIfTI RAS physical mm for ANTs and maps the result to test surface using test H5. Script 13 checks agreement with the released T1w electrode TSV but explicitly says this does not prove correspondence to NIfTI physical coordinates. Script 32 and later methods correctly label results PASS_PROVISIONAL / PROVISIONAL_REGISTRATION_DEPENDENT.

Therefore the old registration can be reused as a transform resource and point-mapping implementation, but its head/T1w/NIfTI physical-space bridge must be established for all subjects before promoting results to validated anatomical localization errors. A forward MRI/head transform is not automatically a surface-RAS/scanner-RAS conversion. A roundtrip can pass while the input physical frame is wrong.

Ground truth in the old evaluation stays in test-native surface space. Only the predicted reference source is anatomically mapped. Do not transform test ground truth with a reference-to-test warp or tune registration to its distance. With nonlinear mappings, transforming a midpoint differs from averaging transformed contacts; preserve the native test midpoint definition.

## 7. What exists and what is missing

Available in GitHub: all nine main scripts and five additional/archive scripts; six frozen raw-grid inputs (two via LFS); frozen preparation plans; both source manifests; 31-file solver snapshot; three historical baseline CSVs; earlier registration/coordinate scripts and top-level solver/grid source.

Not tracked and not available in this cloud instance:

* `cross_subject_no_warp/outputs/` entirely, including selected LOSO/reference parameters, coordinate audits, raw/patch/final reconstruction records, sensitivity variants, evaluations and figures.
* `cross_subject/outputs/`, including the 366-row original schedule, 42-pair registration status, transform files, MRI QA, source result JSONs and prior comparison outputs.
* Localize-MI released data: epoch arrays/sidecars/events/channels, contact TSVs, forward FIFs, H5 transforms and maskface MRIs.
* ECD BEM solutions expected under `outputs/12_additional_methods_all_runs/bem` (exactly one per requested model subject).
* Historical software-version manifests from the actual reconstruction runs.

In the prepared virtualenv, MNE and h5py are present; nibabel, sesameeg, invert/Champagne and pyvista were not found by module-presence checks. ANTs commands were not found on PATH. These were not installed as part of this read-only audit, and successful old dependency setup is not assumed. Reproduce recorded package versions before numerical comparison.

Original Script 01 will not run even in plan-only mode without its old schedule CSV. The tracked 61-row case plan is sufficient to derive a proposed 366-row schedule deterministically, but that should be a documented provenance-preserving implementation choice, not represented as recovery of the original output file.

## 8. Recommended `cross_subject_warp/` design

Retain all nine exact main script filenames, output-folder basenames and numbered artifact names under a new `cross_subject_warp/` root. Retain final and sensitivity folders as separate artifacts. Add a registration-mode column (`affine`/`syn`) and explicit mapping/QA fields; include that dimension in evaluation keys and grouping. Do not leave mapped rows labeled `single_reference_no_warp`; use a cross-reference condition plus the explicit registration mode and preserve original condition as provenance where importing records.

Registration should be a post-reconstruction mapping of the already selected source point. Keep peak extraction before mapping. Do not warp lead fields, EEG sensor positions, orientation priors, the source grid, or the full source-amplitude image for the primary comparison. Do not reselect a peak after nonlinear interpolation. This preserves the inverse problem and permits affine and SyN to share identical fits.

| Script | Warp-specific responsibility |
|---|---|
| 01 | Same schedule/folds; audit model/GT frames, complete 42-pair inventory, registration resources and source provenance. Plan only; no fitting. |
| 02 | Identical frozen grids and strict LOSO selection; same 84 winners if inputs/policy match. |
| 03 | Same patched LOSO within/reference fits and fingerprints; retain native model-head peaks for later mapping. |
| 04 | Identical reference-only selection and 366-case schedule. |
| 05 | Same 4,392 reference-only fits and native peak records. |
| 06 | Validate primary/sensitivity records; map each cross peak using each registration; calculate distances in one validated test frame. Preserve within and raw diagnostic fields. |
| 07 | Compare affine/SyN on matched keys and valid metrics; preserve historical and LOSO within baselines separately. |
| 08 | Reference sensitivity stratified by affine/SyN and primary/sensitivity; retain failures and repeated-measure structure. |
| 09 | Preserve numbered plots, add explicit registration legends/facets; do not mix diagnostic no-warp distances with anatomical errors. |

Keep registration fitting in a provenance-pinned helper/resource workflow rather than changing the nine main stages. Existing transforms should be reused only after MRI hashes, ANTs command/version, direction, field/grid validity and QA match; missing transforms are blockers, not grounds to execute SyN during preparation.

If canonical no-warp outputs are provided, their inverse results may be imported into warp with checksums after verifying configurations, dataset/forward/BEM, solver/software versions, channel/data fingerprints and peak policy. Do not claim reuse validation from row counts alone. If reconstruction is repeated, it is still 1,464 + 4,392 = 5,856 primary inverse attempts, not twice that for affine/SyN. There are 5,124 cross peaks per scenario to map by each registration: 10,248 mapped cross evaluations plus 732 unchanged within evaluations. Sensitivity scenarios largely duplicate observations and are not independent samples.

### Proposed source provenance layout

Retain `source/02_parameter_grids`, `source/03_solver_code`, `source/07_historical_baseline` with original files and checksums. Preserve the old source manifests; add a separate import manifest explaining original commit/path, copy checksum and how new experiment plans relate to the frozen historical plans. A solver/grid snapshot must not silently change its preparation records while retaining old hashes.

Add `source/01_registration_resources/` (new resource directory, not a new numbered main script stage): pinned Script 19/31/32 and required coordinate/QA dependencies, pair manifest, ANTs version/commands, MRI/transform/QA hashes and frame definitions. Large transforms can remain in an explicitly audited external immutable store rather than duplicating them blindly. Record original source and actual availability for every pair.

### Recommended Script 01 outputs and readiness gates

Keep original `01`–`07` preparation artifact names. Add, for example, `08_registration_pair_inventory.csv`, `09_registration_qa.csv`, `10_source_provenance.json` and a README documenting the physical-frame chain.

1. Validate exactly seven subjects, the authoritative 61-run manifest, seven LOSO folds, primary/fallback reference policy, 366 all-reference cases, 12 methods and 42 directed registration pairs.
2. Audit all required EEG metadata/forward/contact/H5/MRI/BEM paths without executing inverse models or registration. Separate missing, unvalidated and validated resources.
3. Snapshot or reference existing historical grids/solver/baseline with hashes; retain the historical sub-01 montage limitation and known-hemisphere policy.
4. Verify head/source-frame units, H5 rigidity/inversion and forward transform agreement. Establish head/surface to actual MRI-world RAS bridges with authoritative geometry; do not equate named T1w electrode coordinates with NIfTI RAS without evidence.
5. Inventory affine and complete SyN chains for all 42 pairs, including independent affine-only and SyN-affine components, forward/inverse fields, MRI provenance and saved QA. Lightweight point/image consistency checks belong in an explicit audit phase once resources exist.
6. Use MRI-only QA and predeclared acceptance criteria; never choose transforms using localization error. Preserve provisional status where anatomical validation is missing.
7. Write per-mode readiness, failure reasons, all planned dependencies and unresolved assumptions. Plan-only must remain not-ready. No construction/reconstruction work in Script 01.

## 9. Methodological issues to carry forward

* Frame bridge is the main blocker: old registration errors are explicitly provisional, and numerical inverse consistency cannot validate physical anatomy.
* Historical grids were developed on the same seven subjects. Training-only re-selection avoids direct test-fold outcome use, but does not make the search-space history prospective validation.
* Parameters are selected on historical fixed-sub-01 montages, then transferred to reference-derived montages. Preserve and disclose this mismatch rather than silently retuning.
* Known stimulation hemisphere informs cortical peak extraction; dipole methods use maximum GOF without that constraint. This is not a uniform uninformed whole-brain benchmark.
* Failure-only whole-brain overrides create a mixed selection policy. Preserve them as sensitivity; do not promote them into primary or describe them as uniform whole-brain results.
* Hard-coded failure counts and identities are dataset-specific historical checks. Reuse unchanged only when importing exactly the validated historical records; new mapping failures must not be forced to match those counts.
* Reconstruction success, registration success and evaluation validity need separate fields. A registration failure must not rewrite a valid reconstruction as a solver failure or disappear from denominators.
* Reference sensitivity varies parameters and head model together. Six references for one EEG run, multiple runs per subject, affine/SyN, and primary/sensitivity variants are dependent observations.
* Historical globally selected baseline is optimistically in-sample. Paired comparisons require matching preprocessing, peak definition and actually observed software behavior.
* Snapshot checksums protect bytes, not package behavior. Current solver wrappers are version-sensitive, especially Champagne and SESAME.
* No full STCs are saved by the no-warp reconstruction. Post-hoc point warping is possible from execution records; whole-field warping or a new peak policy requires additional saved artifacts or reconstruction.
* Documentation includes stale folder names and incomplete plot inventory. Executable defaults and consumer paths should define the new mirrored layout.

## Conclusion

The correct branch provides enough source and historical grids to specify a faithful warp extension. The recommended experiment keeps both selection strategies and inverse reconstruction unchanged and adds affine/SyN point mapping at evaluation. The unavailable local reconstruction outputs and registration resources must be supplied or regenerated in a later authorized task. Before anatomical-error claims, resolve the provisional MRI physical-frame bridge. No experimental implementation has been started.

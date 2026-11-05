# Script 03: Reconstruct reference based LOSO conditions

**Implementation:** [`scripts/03_run_localization.py`](../../scripts/03_run_localization.py)

## Purpose and research objective
Reconstruct reference based LOSO conditions. The design covers participants sub-01–sub-07, 61 test runs per participant, and 12 inverse methods. A test subject supplies the EEG and ground truth; a reference subject supplies a head model and, in the single-reference strategy, its own parameter selection.

## Inputs and provenance
- `scripts/03_finalize_reference_based.py`
- `scripts/03_run_localization.py`
- `scripts/03_run_localization_before_patch.py`
- `scripts/03_wholebrain_gamma_sensitivity.py`

Consult the implementation for exact default paths and `argparse` defaults. Principal data are the Localize-MI derivatives, the prepared schedule, frozen grid snapshots under `source/02_parameter_grids/`, and prior stage outputs. Historical grid sources and exact hashes are recorded in source/output manifests.

## Dataset, selection, and processing

The final design uses 7 subjects × 61 runs. LOSO selection removes the test subject before configuration screening and ranking. Single-reference selection is performed independently for each reference subject on the other six subjects; this is distinct from LOSO and can change both the head model and chosen parameters. The primary reference is sub-01, with sub-02 used when sub-01 is the test participant.

The reconstruction uses the test EEG with either the test subject model (`within`) or reference model (`single_reference_no_warp`). No inter-subject coordinate adjustment, affine, SyN, or warp is applied. The primary peak policy is stimulation-hemisphere restricted for cortical methods; Continuous-ECD uses maximum GOF and SESAME uses posterior probability selection. Sensitivity outputs use whole-brain peak selection only for explicitly retried failure cases.

## Methods, parameters, and workflow

See implementation functions named `main`, `make_plan`/`make_jobs`, `select_fold`, `solve`, and evaluation-specific functions. The twelve methods are MNE, dSPM, sLORETA, eLORETA, Continuous-ECD, RAP-MUSIC, LCMV, MxNE, irMxNE, Gamma-MAP, Champagne, SESAME. Parameter schemas and selected values are preserved in the parameter CSVs and manifests; this document does not substitute guessed values for those records.

## CLI and outputs

Defaults and flags are defined in the linked script. Example from repository root:

```bash
python cross_subject_no_warp/scripts/{f.name if f else '0X_script.py'}
```

Stage output: [`../02_outputs/{outputs[n]}.md`](../02_outputs/{outputs[n]}.md). `--overwrite`, audit/plan/evaluate/reconstruct/merge phases, and path overrides are available where declared by `argparse`; consult source for exact semantics.

## Validation and failure handling

The implementation validates schedule uniqueness, expected subject/run/reference coverage, coordinate frames and rigid transforms, selected configuration coverage, primary/variant key identity, and PASS/FAIL statuses. Solver exceptions become FAIL records with failure reasons. The final LOSO reconstruction retains one failure (Gamma-MAP, test sub-05 run-08, reference sub-01); single-reference retains 14 failures. PASS denotes solver completion, not anatomical validity or optimizer convergence.

## Dependencies, provenance, and limitations

Previous numbered stage: {('none' if n==1 else f'Script {n-1:02d}')}. The current final implementation is identified by git commit `cf89dd2b0257e302fb5de1d5c280b5b074e67aad` (2026-11-01, “Complete cross-subject no-warp LOSO and single-reference experiments”). Helper code is under `source/03_solver_code/src/localize_mi/`; manifests and SHA-256 fields provide artifact provenance. Numerical coordinate audits do not establish anatomical fiducial accuracy. Cross-subject raw head-coordinate separation is diagnostic only and is not a validated anatomical localization error.

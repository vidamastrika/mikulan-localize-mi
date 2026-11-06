# Cross-subject parameter experiments

This folder contains the cross-subject parameter selection and evaluation from
`cross_subject` (Scripts 39 and 40). The experiment varies each localization
method's configuration parameters while keeping the method and its head model
setup fixed. It does not compare or cross head models.

## Contents

- `scripts/39_loso_parameter_selection.py` selects configurations from
  precomputed parameter grids using strict all-PASS reference-subject results.
- `scripts/40_loso_evaluation.py` evaluates single-reference and leave-one-
  subject-out selections and compares them with the fixed Script 26 baseline.
- `outputs/39_loso_parameter_selection/` includes selection tables and the
  archived full-grid CSVs used to support the evaluation. The canonical grid
  backup is large (about 565 MB).
- `outputs/40_loso_evaluation/` includes run-level evaluations, summaries,
  figures, and their analysis manifest.
- `sources/` contains copies of the input grids for Script 39 and the Script 26
  baseline tables used by Script 40. `sources/source_manifest.json` records
  original paths, sizes, and SHA-256 checksums.

The outputs and sources were copied; the original files remain in place. The
result manifests are preserved as provenance and therefore retain the
original input paths. From the repository root, rerun the pipeline with:

```sh
python cross_parameters/scripts/39_loso_parameter_selection.py
python cross_parameters/scripts/40_loso_evaluation.py
```

Both scripts default to the copied inputs and write to `cross_parameters/`.
Script 39 recomputes selection and rewrites its output tables and grid archives;
Script 40 then evaluates those selections and writes its summaries and figures.

The global fixed baseline is selected using all subjects, so it is descriptive
and in-sample rather than held-out validation.

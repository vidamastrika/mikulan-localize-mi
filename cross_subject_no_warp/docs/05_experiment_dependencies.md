# Experiment dependencies and provenance

## Local dependencies

- Dataset root default: `~/Documents/DATASETS SOURCE LOCALIZATION/Localize-MI/`.
- Schedule default: `cross_subject/outputs/30_all_reference_sesame_batch/experiment_schedule_all_references.csv`.
- Frozen parameter input snapshots and preparation plan: `cross_subject_no_warp/source/02_parameter_grids/`.
- Solver implementation snapshot: `cross_subject_no_warp/source/03_solver_code/`; modules include `localize_mi/io.py`, `inverse.py`, `metrics.py`, and `plotting.py`.
- BEM default: `outputs/12_additional_methods_all_runs/bem`.
- Historical baseline snapshot: `cross_subject_no_warp/source/07_historical_baseline/`.

## Reproduction graph

See the Mermaid dependency diagram in [the overview](00_experiment_overview.md). Numbered scripts consume prior outputs in order, with Script 03's finalization and Script 05's finalization auditing reconstruction status before evaluation.

## Provenance and limits

The script tree's latest recorded commit is `cf89dd2b0257e302fb5de1d5c280b5b074e67aad` (2026-11-01). Parameter and execution manifests record input hashes and configuration fields. Outputs are present locally and CSV counts were checked against their files. The external dataset itself is not included in this experiment directory; absolute local dataset files and their availability cannot be independently confirmed from this documentation pass. No reconstruction was rerun.

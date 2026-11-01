# Next steps

1. Make the existing local output folders accessible to the coding session.
   Codex CLI running in your Ubuntu project can inspect them directly. A cloud
   session sees GitHub code, not your desktop's filesystem.
2. Inventory existing affine/SyN transforms, MRI inputs and QA reports. Copy
   them without overwriting originals into this experiment's ignored local
   registration-resource area, recording hashes and original paths.
3. Implement Script 01 with the same schedule/fold/artifact conventions. Its
   first run should inspect files and report readiness only. Validate 7
   subjects, 61 EEG runs, 12 methods, 7 LOSO folds, 366 all-reference run cases,
   42 registration pairs, coordinate frames and dependencies.
4. Check the provisional MRI coordinate bridge. Existing matrix/field files
   alone do not establish a correct anatomical point mapping.
5. Adapt Scripts 02–05 while preserving strict training-only all-PASS selection,
   reference-derived montages, patched inverse behavior and failure policy.
   If canonical local reconstruction records match all required inputs and
   fingerprints, import them with provenance instead of rerunning inverse fits.
6. Add affine/SyN point mapping and QA to Script 06. Adapt comparisons and plots
   in 07–09 to distinguish registration modes and retain failures.

The registration scripts, solver code and historical grids have already been
copied into `source/` with provenance. No experiment computations are needed
until local resource inventory and coordinate validation are complete.

## Suggested request in a local Codex session

> Continue `cross_subject_warp/` on branch
> `experiment/loso-single-reference-warp`. Read its README and provenance
> manifest. First inventory my existing `cross_subject/outputs/` registration
> files, `cross_subject_no_warp/outputs/` canonical reconstruction files and
> Localize-MI dataset. Copy reusable local outputs into the new experiment
> without altering originals, and record checksums and source paths. Implement
> and run only Script 01's preparation/audit stage. Do not run registrations,
> inverse reconstructions, or change selection/solver policies.

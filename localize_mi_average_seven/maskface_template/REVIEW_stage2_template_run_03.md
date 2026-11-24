# Stage 2 construction attempt: run 03

## Outcome

Preparation `outputs/02_stage2_preparation/run_03/` completed successfully. It contains all seven whitespace-free staged MRI inputs, with source and staged hashes recorded. Construction `outputs/03_template/run_03/` began with those staged inputs and did not fail on file paths.

The construction process ran for approximately 47 minutes, then received `KeyboardInterrupt`. The manifest status is `INTERRUPTED`, with no successful return code. The process is no longer running. Three participant registered rasters and transforms are present; the fourth has a repaired input but no completed registration. There are no completed SyN iteration template backups and no `ANTs_iteration_*` backups. The log ends during registration, before a groupwise iteration completed.

The existing `group_template0.nii.gz` and `initial_group_template0.nii.gz` are preliminary initialization images, not completed average-seven templates. Do not run MRI QA or FreeSurfer on them. The subsequent `FileExistsError` messages are the scripts protecting the partial run directory from overwrite; they do not indicate that another process is running.

## Next action

Keep preparation run 03 and partial construction run 03 unchanged. To restart the full ANTs construction from its beginning, reuse completed preparation run 03 and select fresh output `outputs/03_template/run_04/`. This will repeat registration and may take substantial time. No supported resume point was reached in run 03.

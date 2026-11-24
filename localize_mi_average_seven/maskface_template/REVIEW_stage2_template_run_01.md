# Stage 2 template construction attempt: run 01

## Outcome

Run `outputs/03_template/run_01/` failed before groupwise SyN iterations completed. Preserve this directory as a failed execution record; do not use its preliminary `group_template0.nii.gz` or `initial_group_template0.nii.gz` as the Stage 2 template.

## Failure evidence

The released MRI files reside under `/home/girvi949/Documents/DATASETS SOURCE LOCALIZATION/Localize-MI/`. The installed ANTs shell pipeline did not consistently quote input paths containing spaces. The log shows N4 attempting a truncated path (`LOCALIZATION/Localize-MI/...`) and transform initialization attempting a path with spaces removed (`DATASETSSOURCELOCALIZATION/...`). It then reports that the moving image was not set and cannot find the affine transform. The N4 subprocess segfault follows the malformed input path. The top-level `All_Command_lines_OK` line is not a successful completion signal; required registration products and iterations are absent.

This is a shell path-handling failure. The recorded log does not support a conclusion about anatomy, masking, intensity calibration, registration quality or FreeSurfer suitability.

## Corrective workflow

The preparation script now copies all seven raw maskface images byte-for-byte into `outputs/02_stage2_preparation/run_02/ants_inputs/`, using whitespace-free filenames. It compares each staged file's SHA-256 with its released source, records both paths/hashes and preserves the raw image data. The ANTs command then uses only the staged paths. No fitted intensity gains are applied.

Use fresh `run_02` outputs for preparation and construction, then continue with MRI QA. The previous `run_01` attempt and its hashes remain untouched. See [STAGE2_RUN.md](STAGE2_RUN.md).

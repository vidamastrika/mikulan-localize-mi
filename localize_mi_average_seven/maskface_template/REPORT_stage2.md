# Stage 2 preparation report

## Status

Stage 2 preparation run 03 completed. Construction run 01 failed because ANTs did not parse the spaced source paths; preparation run 02 failed due to a Path-versus-hash comparison bug; construction run 03 used the corrected staged paths but was interrupted before completing a groupwise iteration. All runs are preserved. See [run 01 review](REVIEW_stage2_template_run_01.md), [preparation run 02 review](REVIEW_stage2_preparation_run_02.md), and [construction run 03 review](REVIEW_stage2_template_run_03.md). Reuse preparation run 03 and restart construction in fresh run 04 if continuing.

Stage 1 review supports an exploratory construction trial, with anatomy remaining modified and independently unverified. The original Pydeface template and all earlier outputs remain unchanged. Stage 2 outputs are assigned separate run directories under `outputs/`.

## Planned experimental comparison

The seven released maskface MRIs are the sole primary input difference from the original average-seven construction. Raw maskface inputs are used as stored; NIfTI scaling is handled by the image reader, and Stage 1 fitted intensity gains are not applied.

The construction repeats the recorded four-iteration ANTs unbiased groupwise SyN/CC procedure, with the same input order and installed script defaults checked against the original records. `-b 1` is added only to preserve per-iteration backups. The ANTs log will provide the executed interpolation, averaging and registration records. The original construction used linear interpolation, a mean of normalized intensities and a maximum input grid of 256 voxels, based on the largest input's physical space. Original ANTs executable hashes were not saved; the preparation stage records the currently used script/executables and their hashes, so exact binary identity with the historical run cannot be established.

FreeSurfer will use the same recorded 8.2.0 build and `recon-all -all`, in a new `average_seven_maskface` subject directory. The four watershed outputs will be generated separately with the historical `mri_watershed -useSRAS -surf` invocation, without `-T1` or `-atlas`. They will remain unmodified diagnostic surfaces.

## Manual execution and review

Run commands in the order documented in [STAGE2_RUN.md](STAGE2_RUN.md). Review the pre-FreeSurfer MRI QA before deciding whether to run FreeSurfer. After reconstruction, review the final overlays, logs, topology/intersection screens, conditional nesting screens and boundary cues. Each executed stage writes its own `REPORT_stage2.md`, manifest, hashes and stage outputs.

FreeSurfer completion and numerical surface checks do not establish anatomical validity. This experiment stops after anatomical QA. No BEM correction or solution, source space, EEG alignment, forward model or inverse localization is part of Stage 2.

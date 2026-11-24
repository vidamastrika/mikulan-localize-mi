# Maskface template experiment stages

## Stage 1 — reviewed feasibility

Completed in `outputs/01_input_feasibility/` and `outputs/01b_intensity_review/`. See `REVIEW_stage1_intensity_corrected.md`. The maskface inputs are technically usable for an exploratory construction trial, while superficial anatomy remains modified and unverified.

## Stage 2 — template construction and anatomical QA

Scripts are prepared for manual execution. They preserve all original and Stage 1 results and write only to new numbered directories below `maskface_template/outputs/`.

1. `scripts/02_prepare_stage2.py` verifies prior manifests, reviews, seven raw maskface files, image headers, original pipeline records, installed ANTs defaults and the pinned FreeSurfer build. It does not register or reconstruct images.
2. `scripts/03_construct_template.py --execute` runs the original four-iteration ANTs groupwise SyN/CC procedure. The only registration-script option change is `-b 1`, retaining per-iteration backups. Raw input arrays are used; no fitted Stage 1 gains are applied.
3. `scripts/04_compare_template_mri.py` runs before FreeSurfer. Review image coverage, brain/face/left-lateral signal cues, headers, registered rasters, warp screens and all native reference panels.
4. `scripts/05_run_freesurfer.py --execute` runs the pinned FreeSurfer 8.2.0 `recon-all -all` in a new `average_seven_maskface` subject directory, then generates four unmodified surfaces using the historical `mri_watershed -useSRAS -surf` invocation. It does not run a BEM solution.
5. `scripts/06_review_surfaces.py` produces high-resolution MRI overlays, surface topology and intersection screens, conditional nesting screens, layer-gap proxies and boundary intensity summaries.

Successful software execution is not anatomical validation. Review the logs, coordinate assumptions, overlays and QA tables. No surface repair, BEM correction/solution, source-space construction, EEG alignment, forward computation or inverse localization is included in Stage 2.

## Construction settings and caveats

The original command used four unbiased groupwise iterations, SyN with CC, ANTs defaults recorded in the preparation manifest, and linear interpolation as recorded in the original log. The group image was built on the largest input's physical grid with a maximum dimension of 256 voxels. Stage 2 pins current ANTs script and executable paths because original binary hashes were not retained; exact binary identity of the historical run cannot be proven. The new output retains all intermediate iteration backups, registered rasters, affine transforms, forward/inverse warps and logs.

The original FreeSurfer command was `recon-all -all` on the average image using build stamp `freesurfer-linux-rocky8_x86_64-8.2.0-20260314-d932c45`. Watershed surfaces are generated separately with `-useSRAS`, without `-T1` or `-atlas`, matching the historical surface procedure. The generated surfaces are diagnostics and remain unmodified.

Stage 1 raw maskface image values are retained. The fitted gain/offset was a display/measurement diagnostic only and is not applied to construction. FreeSurfer may internally conform the input and perform its usual intensity processing; commands and resulting MRI headers are retained.

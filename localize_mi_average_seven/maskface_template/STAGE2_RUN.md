# Stage 2 manual run order

Run commands from the repository root. Preparation run 03 completed and is reusable. Construction run 03 was interrupted; it is partial and must not be used as a template. No ANTs process is currently running. Preserve both directories and use a fresh construction output, run 04.

## 1. Verify completed preparation

Review `outputs/02_stage2_preparation/run_03/manifest.json`, `REPORT_stage2.md`, `pipeline.json`, and `inputs.json`. Confirm status is `COMPLETE` and the seven staged SHA-256 values match their released-source values. Do not rerun preparation into this existing directory.

## 2. Construct the maskface average-seven template

Construction run 01 failed because ANTs did not handle spaces in the source MRI path. Preparation run 02 had a Path-versus-hash check bug. Construction run 03 used the corrected staged inputs, ran for about 47 minutes, and was interrupted during registration; it has no completed groupwise iteration. See `REVIEW_stage2_template_run_01.md` and `REVIEW_stage2_template_run_03.md`.

```bash
.venv/bin/python localize_mi_average_seven/maskface_template/scripts/03_construct_template.py \
  --preparation-root localize_mi_average_seven/maskface_template/outputs/02_stage2_preparation/run_03 \
  --output-root localize_mi_average_seven/maskface_template/outputs/03_template/run_04 \
  --execute
```

Wait for the command to finish. A successful completion prints `COMPLETE`; an interrupted run records `INTERRUPTED` and must not be treated as a valid template. Review the construction manifest and log before continuing.

## 3. Review MRI template before FreeSurfer

```bash
MPLCONFIGDIR=/tmp/maskface-stage2-matplotlib .venv/bin/python \
  localize_mi_average_seven/maskface_template/scripts/04_compare_template_mri.py \
  --construction-root localize_mi_average_seven/maskface_template/outputs/03_template/run_04 \
  --output-root localize_mi_average_seven/maskface_template/outputs/04_template_MRI_QA/run_04
```

Review `REPORT_stage2.md`, all high-resolution panels, regional MRI cues, registered-raster cues, headers, intensity ranges, and coarse warp screens. Continue only if the MRI and QA are suitable for exploratory FreeSurfer review.

## 4. FreeSurfer and unmodified watershed surfaces

After reviewing step 3:

```bash
.venv/bin/python localize_mi_average_seven/maskface_template/scripts/05_run_freesurfer.py \
  --construction-root localize_mi_average_seven/maskface_template/outputs/03_template/run_04 \
  --mri-qa-root localize_mi_average_seven/maskface_template/outputs/04_template_MRI_QA/run_04 \
  --output-root localize_mi_average_seven/maskface_template/outputs/05_freesurfer/run_04 \
  --runtime docker --docker-command 'sudo docker' \
  --image freesurfer/freesurfer:8.2.0 \
  --license /home/girvi949/.freesurfer/license.txt \
  --reviewed-mri-qa \
  --execute
```

The pinned FreeSurfer image must already be available to Docker. A successful return code is not anatomical validation.

## 5. Generate anatomical QA and stop Stage 2

```bash
MPLCONFIGDIR=/tmp/maskface-stage2-matplotlib .venv/bin/python \
  localize_mi_average_seven/maskface_template/scripts/06_review_surfaces.py \
  --construction-root localize_mi_average_seven/maskface_template/outputs/03_template/run_04 \
  --freesurfer-root localize_mi_average_seven/maskface_template/outputs/05_freesurfer/run_04 \
  --output-root localize_mi_average_seven/maskface_template/outputs/06_surface_QA/run_04
```

Review the report, logs, figures, and CSV/JSON screens. Do not repair surfaces, create a BEM solution or source space, or run EEG alignment, forward, or inverse computations.

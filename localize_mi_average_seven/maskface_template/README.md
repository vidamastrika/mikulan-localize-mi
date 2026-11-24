# Average-seven maskface experiment

This is an isolated exploratory variant. Stage 1 feasibility is complete and reviewed. Stage 2 scripts are ready for manual execution; registration and FreeSurfer reconstruction have not been launched by this preparation.

See [Stage 1 review](REVIEW_stage1_intensity_corrected.md), [Stage 2 preparation report](REPORT_stage2.md), and the ordered [manual execution commands](STAGE2_RUN.md).

Stage 2 creates a separate four-iteration ANTs average, evaluates the MRI before FreeSurfer, reconstructs a new `average_seven_maskface` FreeSurfer subject, and generates unmodified watershed boundaries for anatomical review. Outputs are written only under `outputs/` with numbered run directories. Existing paths are refused; use a fresh run number for retries.

The workflow uses raw maskface MRI values; Stage 1 fitted intensity calibration is not applied. It preserves the original registration thresholds and does not modify original MRI/surfaces, repair surfaces, create a BEM solution or source space, perform EEG alignment, or run forward/inverse computations. FreeSurfer completion is not anatomical validation.

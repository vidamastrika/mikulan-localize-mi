# EEG alignment investigation: preparation and existing evidence

Status: **PREPARED, NOT EXECUTED**. Dataset diagnostic outputs will be created by `diagnose.py` in a new `run_01/` directory. This report records inspection of existing artifacts; it does not claim newly computed results.

The inspected branch is `experiment/localize-mi-average-seven`. Git reports the entire existing `localize_mi_average_seven/` tree as untracked. Existing files and results were preserved. This numbered diagnostic directory is separate from the earlier `04x` and `04y` outputs.

## Findings verified from existing outputs and source

- `outputs/04x_exploratory_alignment_retry1/01_alignment.csv` records average-seven passing for sub-05/06/07 (3/7), and fsaverage passing for all except sub-03 (6/7). Acceptance remains nearest-vertex median ≤10 mm and p95 ≤20 mm, with review required.
- `outputs/04y_scalp_fit_diagnosis_retry1/01_native_frame_audit.csv` records native scalp median distances of 9.76–10.77 mm and p95 of 14.62–22.68 mm. These are historical proximity measurements, not new results or independent digitization validation.
- That audit records identity native forward rotations and forward translations matching GIFTI volume centers to less than 0.001 mm. This is consistent with a scanner-RAS-style convention; the anatomical meaning of the HEAD tag remains unresolved.
- The earlier report gives average-seven scalp extents of 154.40 × 200.56 × 213.11 mm and fsaverage extents of 170.01 × 212.91 × 231.56 mm. Coverage, orientation, neck extent and facial masking affect these measurements.
- `docs/SCALP_FIT_DIAGNOSIS.md` documents that alternative rigid starts and landmark perturbations did not resolve the main failures. This diagnostic reuses nominal saved transforms and does not repeat fitting or select better starts.
- The inspected sub-01 released epochs are NPY signals plus JSON/TSV metadata. Their channel tables support label/order checks, but do not supply independent digitized sensor positions.
- `fx_bids.make_coordsys_eeg_json` reads epoch dig fiducials, writes them into the EEG sidecar and labels their coordinate system T1w. `make_electrodes_eeg_tsv` writes epoch channel locations. These functions do not establish which epoch files generated the release or explain the repeated fiducials.
- `fx_preprocess.make_dig_files` negates Polhemus X and Y and exports HPTS/FCsv records. Original records and the subsequent montage import history remain necessary to verify conventions.

## Questions the prepared script measures

The script audits all seven original forward models, released EEG electrode and coordinate sidecars, run channel tables, native scalp GIFTIs, native head-to-surface HDF5 transforms and saved template transforms. It recovers all forward sensor positions with labels and frame tags; compares HDF5 transforms with inverse forward transforms; exports forward dig records when retained; and checks repeated sidecar/TSV values across participants.

It compares native, fsaverage and average-seven scalp residuals using both full-mesh nearest-vertex distances and exact unsigned point-to-triangle distances. Only nearest-vertex metrics reproduce the unchanged numerical gate. It exports electrode residual vectors, fixed native geometric region bins, fractions above 20 mm and regional shares of squared residual. Regional concentration is descriptive; no new acceptance threshold is introduced.

Each participant receives a PNG with three views of each scalp/sensor configuration, common mm axes, a shared color scale, and labels/segments for the largest residuals. Cohort channel heatmaps show repeated regional/channel patterns. Mesh dimensions, robust spans, horizontal sections and landmark distances compare seven participants with both templates.

## Assumptions and unresolved issues

Surface positions follow the existing FreeSurfer surface-RAS/mm contract. FIF positions follow their metre/HEAD tags. Numerical consistency does not independently establish anatomical correctness. Two explicit sidecar fiducial interpretations are exported without selecting one. Template fiducials remain estimated and unreviewed; no verified native anatomical fiducials are available.

Native geometric region bins and unregistered horizontal sections are orientation and coverage dependent. They support inspection rather than definitive anatomical attribution. A coherent global or regional residual pattern can suggest a cause, but does not establish whether the cause is cap geometry, surface coverage, digitization convention or registration.

Matching released HDF5/forward metadata can share upstream provenance. Repeated TSV/sidecar coordinates may reflect shared montage or export inputs; that is a hypothesis. The actual cause and independent anatomical sensor verification remain unresolved without original digitizer records or expert review.

The repaired BEM candidate remains pending anatomical review. This work does not modify surfaces, change thresholds, scale heads, fit new transforms, calculate forwards or run inverse localization.

## Execution and deliverables

Run the commands in `README.md`. The script refuses an existing output directory. Each run writes twelve diagnostic CSVs, seven participant PNGs, two cohort PNGs, transform and surface metadata JSON files, a content-hashed `manifest.json`, and a computed `REPORT.md`. Interrupted runs retain status RUNNING; only a fully completed run receives COMPLETE.

Preparation checks passed for Python syntax, analytic triangle interior/edge/vertex/degenerate cases, proper-rigid validation and rejection of scaling. A full import check was blocked by MNE attempting to open the read-only home configuration in this managed session. No dataset analysis was executed. The geometry checks isolated the pure numerical functions and do not substitute for the manual dataset run.

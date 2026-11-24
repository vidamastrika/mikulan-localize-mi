# Manual execution

From the repository root:

```bash
cd /home/girvi949/projects/mikulan-localize-mi
.venv/bin/python localize_mi_average_seven/diagnostics/01_eeg_alignment/diagnose.py \
  --output-root localize_mi_average_seven/diagnostics/01_eeg_alignment/run_01
```

Uses the existing experiment configuration, dataset symlink, saved `04x_exploratory_alignment_retry1` transforms, average-seven repair-candidate scalp, and installed fsaverage. It verifies saved alignment input hashes and reproduces the nominal metrics before reporting results. No anatomy or existing result is written. Exact triangle distances examine every face and can take several minutes; progress prints once per participant.

If rerunning, use `run_02` (then `run_03`, etc.). Existing directories are refused, including interrupted runs. Input locations can be overridden with `--dataset`, `--alignment-root`, `--average-seven`, and `--fsaverage`; the saved alignment provenance must still match. Run `--help` for defaults.

If a headless machine needs a writable Matplotlib cache:

```bash
MPLCONFIGDIR=/tmp/average-seven-matplotlib .venv/bin/python \
  localize_mi_average_seven/diagnostics/01_eeg_alignment/diagnose.py \
  --output-root localize_mi_average_seven/diagnostics/01_eeg_alignment/run_02
```

The completed `run_01/REPORT.md` contains measured findings and caveats. `manifest.json` contains input/output SHA256 hashes and completion status. Files:

| Output | Contents |
|---|---|
| `01_coordinate_audit.csv` | Units/frame tags, labels, TSV comparison, native transform checks, forward dig records |
| `02_recovered_sensors.csv` | All 256 original sensor positions per participant |
| `03_model_distances.csv` | Native and both template distance summaries; unchanged vertex gate |
| `04_electrode_residuals.csv` | Per-electrode distances, positions and closest-triangle residual vectors |
| `05_regional_residuals.csv` | Geometric region counts, residuals and concentration measures |
| `06_scalp_dimensions.csv` | Mesh extents and robust coordinate spans |
| `07_shape_sections.csv` | Unregistered horizontal width/depth profiles |
| `08_landmark_audit.csv` | Explicit native sidecar hypotheses and estimated template landmarks |
| `09_cross_participant_identity.csv` | Pairwise fiducial/TSV identity and raw forward geometry differences |
| `10_available_coordinate_files.csv` | Scoped dataset coordinate/transform/digitization file inventory |
| `11_landmark_dimensions.csv` | Inter-landmark distances with provenance |
| `12_template_native_dimensions.csv` | Template minus native mesh extents |
| `transforms.json`, `surface_metadata.json` | Original matrices and surface metadata |
| `figures/sub-*_scalp_residuals.png` | Per-participant scalp/electrode/residual views |
| `figures/cohort_channel_residuals.png` | All participant/channel residuals in each model |
| `figures/scalp_sections.png` | Shape profiles of all nine models |

Thresholds apply to nearest-vertex distance only. Exact triangle distances supplement the analysis without changing acceptance. All numerical passes still require anatomical review.

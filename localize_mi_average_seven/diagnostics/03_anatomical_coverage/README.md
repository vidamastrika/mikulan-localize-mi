# Anatomical coverage investigation — manual execution

Run from the repository root:

```bash
.venv/bin/python localize_mi_average_seven/diagnostics/03_anatomical_coverage/diagnose.py \
  --output-root localize_mi_average_seven/diagnostics/03_anatomical_coverage/run_01
```

The script uses the existing virtual environment and existing local data only. It reads `02_scalp_geometry/REVIEW_run_02.md`, verifies input/output hashes for both completed diagnostics, verifies the construction inputs, and reproduces the original full-montage statistics. It does not overwrite any output directory. Use `run_02` for a retry; interrupted runs retain RUNNING status.

This is a diagnostic script, not a new experiment runner. No head scaling, sensor fitting, standard-montage auto-normalization, face reconstruction, surface edits, electrode removal from the primary experiment, or forward/inverse computations occur. All sensitivity statistics use the original saved transforms. The original full-montage CSV is copied verbatim into the new output.

The existing tested MRI/scalp utilities in `02_scalp_geometry/diagnose.py` and `geometry.py` are imported without invoking their main routine. Those files are hashed and must remain consistent with the preceding completed run. Required packages are already present: MNE, nibabel, NumPy, SciPy, pandas and Matplotlib. High-resolution rendering can take several minutes and use substantial memory; progress is printed per participant/model and BEM-boundary configuration.

Optional synthetic verification, without participant data or MNE import:

```bash
MPLCONFIGDIR=/tmp/anatomical-coverage-matplotlib .venv/bin/python \
  localize_mi_average_seven/diagnostics/03_anatomical_coverage/check_coverage.py
```

If Matplotlib needs a writable cache during the full run, use the same `MPLCONFIGDIR` prefix. Paths can be overridden with `--alignment`, `--geometry`, `--dataset`, `--construction`, `--average-seven`, `--fsaverage`; figures default to 220 DPI and require at least 180 DPI.

## Outputs

| Output | Contents |
|---|---|
| `01_anonymization_evidence.csv` | Numbered release documentation, processing sidecars, export code and construction-command evidence |
| `02_channel_anatomy_coverage.csv` | Per-participant/channel geometric anatomy and native-MRI-support proxies, plus existing maskface comparison |
| `03_regional_residual_contributions.csv` | Counts, median/p95 distances, >20 mm counts and squared-residual shares, separately partitioned by region and coverage |
| `04_subset_sensitivity.csv` | Unchanged full montage, native supported subsets, common subsets across all seven, and geometric-definition sensitivity; no refitting |
| `05_preserved_left_right_comparison.csv` | Supported cranial left/right residual distances and mean vector components |
| `06_mirrored_native_proxy_pairs.csv` | Descriptive mirrored-native-scalp nearest pairs; not verified physical sensor pairing |
| `07_construction_slice_support.csv` | MRI support/FOV fractions for the improved construction-stage and registered-input panels |
| `08_BEM_layer_coverage_cues.csv` | Regional MRI support/extents at sampled outer-skull, inner-skull and available brain surfaces |
| `09_residual_plane_validation.csv` | Figure locations, plane origins/bases, endpoint off-plane errors and image-support fractions |
| `10_all_electrodes_with_coverage.csv` | Original per-electrode residuals augmented with outcome-independent native coverage/region labels |
| `11_channel_coverage_consensus.csv` | Which labels/coverage states each EEG channel has across participants |
| `12_original_full_montage_results.csv` | Verbatim preserved full-montage diagnostic statistics |
| `13_existing_inner_skull_repair_comparison.csv` | Original-versus-existing-candidate inner-skull displacements; no adoption |
| `fixed_subset_definitions.json` | Exact common subsets and fixed proxy definitions |
| `anatomy_origin_assessment.json` | Documented upstream anonymization versus unresolved added construction effects |
| `montage_feasibility.json` | Pinned HydroCel resource/assumption record and equivalent-placement feasibility; no placement |
| `BEM_layer_inventory.json` | Layer sources and topology audit; no BEM solution |
| `validation_manifest.json`, `REPORT.md` | Completion status, input/output hashes, versions, measurement results, limitations and recommendations |

Focused PNG groups include:

- Existing native Pydeface versus maskface MRIs, with original scalp overlays in three anatomical planes.
- Per-participant/model facial-or-low-support, supported-left-cranial and inferior-lateral residuals: two planes containing **both** endpoints and one axial slice through the closest scalp point. Figure selection uses the largest residual only for display; it never defines the sensitivity subsets.
- Initial/intermediate/final template and retained registered-input comparisons in residual-aligned planes extending into the scalp, separately for facial and supported-left examples. Earlier construction stages/warped images are not assumed anatomically identical to the final shape-updated template.
- Nine primary native/template layered MRI views of scalp, outer skull, inner skull and available brain segmentation boundaries, including facial and cranial controls. The existing average-seven inner-skull repair is overlaid separately without adoption.

## Interpretation

Anatomical classes are **geometric proxies** in native FreeSurfer surface RAS mm, computed at original native closest scalp points. They are not established by channel numbering. Baseline rules are facial (Z<15, Y>=40, Y>=0.65|X|), inferior lateral (Z<15, |X|>=40, except facial), cranial (Z>=15), otherwise transition/uncertain. Additional fixed Y/Z rules are reported to expose classification sensitivity, not to optimize alignment.

MRI-supported coverage uses 27 inward radial/tangent probes and conservative agreement across three intensity cutoffs. These labels are uncertain because native sensor correspondence, radial directions, head-surface definitions and MRI intensity boundaries remain imperfect. MRI support at depth does not prove intact superficial scalp. Dark skull/CSF is not evidence by itself that anatomy is missing.

The original median ≤10 mm and p95 ≤20 mm thresholds apply to full-montage nearest-vertex results. Subset columns only describe whether the same numerical limits would be met on that subset with the same saved transform. They are not new accepted alignments and do not redefine the primary experiment. Both arms use identical subset labels, including explicitly frozen common subsets across all seven participants.

The release README, sidecars and export source document that Pydeface precedes template construction and that original surfaces were generated from original MRIs. Additional construction effects, BEM anatomical suitability and coordinate correctness require review. No script can recover the removed anatomy from this evidence.

Common HydroCel placement is assessed as a separate assumed experiment. The raw resource and loader are inventoried without calling MNE's standard-montage loader or applying the existing 95 mm normalization. Physical eN-to-EN correspondence remains explicitly unverified. Equivalent procedures alone do not establish equivalent sensor anatomy over a cut-away face.

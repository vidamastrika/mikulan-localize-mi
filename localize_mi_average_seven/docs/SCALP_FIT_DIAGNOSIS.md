# Scalp-fit and coordinate audit — 2026-11-16

Completed outputs: `outputs/04y_scalp_fit_diagnosis_retry1/`.
The empty initial `04y_scalp_fit_diagnosis` directory is an interrupted attempt:
a pandas Series attribute collided with the `transform` column name. The runner
now indexes that column explicitly and all diagnostics were rerun in a fresh
directory. The original exploratory transforms remain unchanged.

## Findings

- Both arm scalp surface metadata matches each arm's T1 geometry: 1-mm voxels,
  consistent dimensions/orientation/volume center. Scanner-RAS/surface-RAS
  roundtrip error is below 2e-14 mm. No unit or surface-header mismatch was found.
- Average-seven scalp extents are 154.40 x 200.56 x 213.11 mm; fsaverage extents
  are 170.01 x 212.91 x 231.56 mm in surface-RAS axes. These are descriptive mesh
  extents, affected by coverage and orientation, not an estimate of head size
  free of those factors. Shape/coverage mismatch is a plausible contributor.
- All seven native forward MRI-to-HEAD transforms have identity rotations and
  translations matching their released skin GIFTI volume centers to <0.001 mm.
  This is consistent with a scanner-RAS-style convention. It does not prove the
  convention or invalidate electrode coordinates, but a conventional
  fiducial-defined HEAD frame must not be assumed from the FIF label alone.
- Original-transform native sensor-to-native-scalp distances are not zero:
  medians 9.76–10.77 mm and p95 14.62–22.68 mm. These are proximity diagnostics,
  not digitization-error estimates or independently validated alignment.

## 224 rigid diagnostic fits

Each arm/participant used 16 predefined starts: estimated landmarks, identity,
inverse native MRI-to-HEAD, continuation of the previous fit for 40 additional
iterations, +/-10-mm translations on each MRI axis, and +/-15-degree rotations
around the mapped sensor centroid on each axis. All fits use the same rigid
40-iteration/15%-trimmed ICP and unchanged numerical thresholds. No scaling,
best-start production selection, participant deletion or formal review override
was introduced. Native inverse is a diagnostic seed, not anatomical MRI mapping.

| Participant | Average-seven passing starts | Average-seven p95 range, mm | fsaverage passing starts | fsaverage p95 range, mm |
|---|---:|---:|---:|---:|
| sub-01 | 0/16 | 23.20–29.02 | 16/16 | 7.77–10.71 |
| sub-02 | 0/16 | 22.51–34.36 | 16/16 | 11.24–14.93 |
| sub-03 | 0/16 | 36.20–42.85 | 3/16 | 19.25–24.94 |
| sub-04 | 0/16 | 33.77–39.11 | 15/16 | 13.71–21.58 |
| sub-05 | 14/16 | 13.46–21.81 | 16/16 | 10.27–14.26 |
| sub-06 | 16/16 | 15.44–19.38 | 16/16 | 7.62–11.03 |
| sub-07 | 16/16 | 13.03–17.63 | 16/16 | 7.11–11.94 |

Average-seven failures for sub-01 through sub-04 persist across tested starts;
small landmark edits or these larger rigid starts do not solve them. This does
not prove the global optimum or a unique cause. Fsaverage sub-03 and sub-04
can cross the numerical threshold between basins. The nominal fsaverage sub-03
FAIL is retained; a numerically better seed is not automatically accepted.

The inner-skull repair has no effect on the unchanged scalp used in these fits.
Numerical BEM success must not be confused with scalp correspondence.

## Available provenance and next inputs

The original preprocessing code loads `spatial/ch_info/<subject>_egi_dig.hpts`,
including three fiducial rows and 256 EEG positions. No `*.hpts`, `*egi_dig*`,
`*.elp` or `*polhemus*` files were found in the released dataset or the accessible
Documents tree. Released EEG sidecars repeat the same landmarks across subjects,
and prior repository audits document their ambiguous coordinate labels; they
cannot resolve independent landmark provenance.

No new ANTs or FreeSurfer run is justified by this audit. The next useful input
is the original digitization/export records or expert verification of their
frame convention. Keep the existing unreviewed exploratory results and failures
as such. The frozen no-warp design may legitimately retain failed alignment jobs;
forcing all subjects to pass would change the experiment. There are currently
three nominal paired numerical passes (sub-05/06/07), still pending anatomy and
alignment review. Formal forward and inverse jobs have not been launched.

The user also ran a filename search over their home directory, `/media`, `/mnt`
and `/run/media` for HPTS/EGI/ELP/Polhemus patterns; it returned no visible matches.
Because errors were redirected and ripgrep respects some ignore rules, this is
not an exhaustive absence proof. Original digitization remains unavailable in
the searched locations; repeating MRI processing cannot supply this provenance.
The current released-data-only result is an exploratory feasibility assessment:
three of seven participants pass the nominal numerical alignment criteria in
both arms, with anatomical review still pending. Retain all seven in coverage
and failure reporting. No formal full-cohort localization comparison is ready.

Figures: `08_diagnosis.png`; exact transforms and frame/header/extent tables are
saved alongside. Code: `scripts/04y_diagnose_scalp_fit.py`.

# Script 18 — forward-model ambiguity demonstration

This companion implements a separate, reproducible **forward simulation**. It does
not use the earlier recorded-EEG fit or modify any baseline result. It starts at
`sub-01`, using `run-01` only to select the baseline usable EEG channels.

## Run through Script 18

From the repository root, using the project environment:

```bash
.venv/bin/python scripts/18_make_presentation_figures.py \
  --source-ambiguity-demo --subject sub-01 --run run-01
```

Only the new demonstration runs. The baseline grids and inverse solvers are not
loaded or executed in this mode. The default search samples up to 2,048 valid
sites with NumPy `default_rng(2026)` and evaluates all unordered sampled pairs at
20, 40 and 60 mm minimum Euclidean separation. It retains ten highest absolute
Pearson correlations per threshold, with source indices breaking ties. The
presentation pair is the best at 40 mm, or the available threshold nearest 40 mm.
No similarity level is assumed in advance.

For all valid sites (more computation):

```bash
.venv/bin/python scripts/18_make_presentation_figures.py \
  --source-ambiguity-demo --subject sub-01 --run run-01 \
  --ambiguity-max-sources 8196 --ambiguity-seed 2026 \
  --output-root outputs/18_presentation_figures_full_search
```

For custom thresholds or a different output directory, run the companion directly:

```bash
.venv/bin/python scripts/script18_source_ambiguity_demo/analysis.py \
  --subject sub-01 --run run-01 --separations-mm 20 40 60 \
  --max-sources 2048 --seed 2026 --top-k 10 --dpi 400 \
  --output-dir outputs/18_presentation_figures/source_ambiguity_demo/sub-01_run-01
```

**Existing output directories are always refused**, including when Script 18's
`--overwrite` flag is supplied. Use a fresh directory for another analysis.
Nothing is computed when the companion is imported.

## Reused modelling inputs

All paths below are relative to `data/Localize-MI`:

- `derivatives/sourcemodelling/sub-01/fwd/sub-01_fwd.fif`: inspected forward
  contains a 256 × 24,588 free-orientation lead field, 8,196 cortical sites in two
  4,098-site source spaces, head-frame source and electrode positions, and an
  MRI-to-head transform. The lead field already incorporates the BEM solution.
- `derivatives/epochs/sub-01/eeg/sub-01_task-seegstim_run-01_channels.tsv`:
  236 good EEG channels and 20 bad channels; channel order matches the forward.
- `derivatives/sourcemodelling/sub-01/xfm/sub-01_from-head_to-surface.h5`:
  rigid head-to-native-surface transform, used only for anatomy rendering.
- `derivatives/sourcemodelling/sub-01/anat/sub-01_hemi-{L,R}_pial.surf.gii`:
  subject-specific cortical surfaces.

The released anatomy also includes inner-skull, outer-skull, and outer-skin
surfaces. A standalone BEM solution is not present in this released subject's
`fwd` directory and is not required. Historical `fx_source_loc.py` specifies a
three-shell BEM with conductivities (0.3, 0.006, 0.3 S/m), ico4 surfaces and oct6
source spacing; the demonstration reuses the released operator rather than
asserting a new verification of those historical construction settings.

Baseline conventions are documented in `src/localize_mi/io.py` and
`src/localize_mi/inverse.py`: use the participant-specific sensor locations stored
in forward Info rather than the released electrode TSV, whose coordinates are
shared across participants; exclude bad EEG channels and apply their average
reference. The demo computes the identical reference projection explicitly on
`G`. Other baseline analyses use reduced montages too; this demo uses the
baseline **all-good-channel** convention.

## Scientific definitions

- Convert to fixed cortical-normal orientations with MNE `surf_ori=True`,
  `force_fixed=True`, `use_cps=True`. Embedded cortical patch statistics are used
  when present, following MNE's orientation conversion.
- Compute distances between `source_rr` coordinates in the **same head frame**;
  convert metres to millimetres. Distances are Euclidean, not geodesic.
- Reject nonfinite source positions and columns, and referenced column norms
  at or below `max(maximum finite norm × 1e-12, floating-point tiny)`.
- Search absolute Pearson correlation between mean-centered scalp patterns.
  Correlation is insensitive to absolute source strength and sign.
- For display, scale positive source A current to 1 µV scalp RMS. Fit one signed
  current for source B by ordinary least squares against A. Record the current
  amplitudes, raw signed correlation, polarity reversal and aligned RMS.
- Relative residual is `||y_A - alpha*y_B||₂ / ||y_A||₂`. With these centered
  maps and unconstrained scalar alignment it equals `sqrt(1-r²)` numerically.
- No random EEG waveforms, temporal signal simulation, noise-level claim,
  recorded-response fit, or inverse-algorithm comparison is involved. The random
  seed determines only which source locations are searched. A high correlation
  does not prove exact non-uniqueness or equal potentials for equal current.

## Generated outputs

Under `outputs/18_presentation_figures/source_ambiguity_demo/sub-01_run-01/`:

```text
figures/source_ambiguity_demo.png             # white, 16:9, default 400 dpi
numerical/source_ambiguity_pairs.csv          # top ten per threshold, all metrics
numerical/source_ambiguity_channel_patterns.csv
numerical/source_ambiguity_manifest.json      # indices, channels, inputs, hashes
source_ambiguity_summary.md                   # actual results and limitations
```

The PNG has three panels: orange/blue sources on native cortex; two predicted
scalp topographies at actual electrode locations with shared scale; and orange
solid/blue dashed electrode potentials in baseline channel order. It explicitly
reports the separation, amplitude alignment, polarity, absolute correlation and
relative residual.

## Dependencies and implementation status

Requires the project's NumPy, pandas, MNE, Matplotlib, nibabel and h5py. All are
available in the current project environment, and the listed sub-01 inputs exist.
Neither epochs nor recorded EEG arrays are required. No dependency installation
or model reconstruction is needed.

Implementation has been syntax checked. The search and rendering have **not**
been executed for this new demonstration, so no numerical outcome is claimed.

If running within a read-only managed environment, MNE and Matplotlib may need
writable configuration locations. Set `_MNE_FAKE_HOME_DIR` and `MPLCONFIGDIR` to
existing task-specific temporary directories before the command; this affects
configuration caches, not modelling inputs.

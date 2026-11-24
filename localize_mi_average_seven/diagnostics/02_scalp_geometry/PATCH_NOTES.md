# Diagnostic reader fix

The first run stopped while reading an alternative FIF scalp because MNE's default `read_bem_surfaces(on_defects='raise')` rejects vertices with fewer than three neighboring triangles. The reader should preserve those defects for investigation rather than stop before auditing them.

The diagnostic FIF reader now uses `on_defects='warn'`, captures the warnings in the alternative-surface CSV and final manifest, and prints each alternative path before reading it. Finite coordinates, triangle indices, units and frame checks remain enforced. No geometry is repaired or replaced.

The explicit topology audit now counts vertices with fewer than three incident faces and excludes such meshes from conditional inside/outside sign eligibility. All distances on these meshes remain unsigned/unresolved. Alignment thresholds remain unchanged; warning-tolerant ingestion does not constitute BEM or anatomical acceptance.

Synthetic regression checks passed for warning capture with preserved coordinates/faces, low-incidence and unused-vertex topology rejection, and the existing numerical geometry/MRI tests. No dataset diagnostic was rerun automatically.

The interrupted `run_01` is preserved. Retry manually into `run_02`:

```bash
.venv/bin/python localize_mi_average_seven/diagnostics/02_scalp_geometry/diagnose.py \
  --output-root localize_mi_average_seven/diagnostics/02_scalp_geometry/run_02
```

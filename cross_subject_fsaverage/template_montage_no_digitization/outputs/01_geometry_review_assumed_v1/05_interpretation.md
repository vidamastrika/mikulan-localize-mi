# Review interpretation and next modeling requirement

The completed review confirms 191 of 256 sensors inside the inspected closed
fsaverage scalp mesh and 65 outside. There are no ambiguous classifications.
The mesh is connected, consistently oriented and without boundary/nonmanifold
edges; self-intersections were not tested. Input hashes match their recorded
values, and signed-distance signs agree with the classifications.

Triangle-boundary separation: median 6.69 mm, p95 14.81 mm, maximum 16.43 mm.
Deepest inside sensor: e15, 16.43 mm. Greatest outside gap: e165, 15.18 mm.
The original nearest-vertex gates remain passed as recorded (median 6.83 mm,
p95 14.90 mm). Numerical gate passage does not prove scalp-conforming geometry.
Fiducial residuals remain 12.54 mm at nasion and approximately 5.01 mm at each ear.
The mapping remains ASSUMED_UNVERIFIED.

## Standard MNE EEG/BEM behavior

Static inspection of installed MNE 1.12.1, `mne/forward/_compute_forward.py`,
function `_bem_specify_els` (lines 230–266), shows that EEG integration-point
locations are converted to MRI coordinates and passed to `_project_onto_surface`
to compute triangle interpolation weights on `bem['surfs'][0]`.

This is solver behavior, not evidence that the original raw montage aligns with
the scalp. It does not justify moving or scaling the stored sensors merely to
obtain a geometry PASS. A future MNE BEM run must disclose both:

1. The original, fixed template sensor locations and their geometry QA.
2. The effective BEM scalp interpolation locations and associated displacement.

The outer_skin mesh inspected here is not necessarily the exact triangulation
stored in the BEM solution. A subsequent audit must use the actual BEM scalp
surface rather than assume its projections match the present nearest-surface
points. Do not use digitization, individual MRI or localization outcomes to
choose or optimize this operation.

## Next step

Before any forward calculation, implement a separately reviewed audit of the
actual BEM surface, effective sensor interpolation locations and displacements,
with all original coordinates/results retained. This is needed to describe what
the standard solver will evaluate. It must not silently relax the original
no-coordinate-adjustment condition or promote the current geometry to a valid
anatomical alignment.

No forward model or inverse fit was computed in this review. Script 02 remains
unimplemented, and no existing readiness result was modified.

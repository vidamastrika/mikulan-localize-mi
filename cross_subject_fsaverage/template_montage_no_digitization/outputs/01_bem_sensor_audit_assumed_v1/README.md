# Actual-BEM sensor interpolation audit

01_bem_interpolation_overlay.png: original and effective locations on the actual BEM scalp.
02_bem_interpolation_displacements.png: displacement distribution and largest channels.
03_effective_bem_sensor_locations.csv: original coordinates, effective locations, triangles, barycentric weights and displacements.
04_bem_interpolation_audit.json: input/code hashes, mesh checks, statistics and solver assumptions.

Effective coordinates are derived audit data, not replacement input sensor coordinates.
MNE recomputes unconstrained triangle-plane weights; negative weights and off-surface effective points are reported, not clipped.
The standard solver interpolates scalp potentials at these locations for monopolar EEG definitions.
No BEM construction, forward matrix, inverse solution or individual registration was run.
Original placement QA and unverified identity-number assumptions are retained. No automatic forward authorization is given.

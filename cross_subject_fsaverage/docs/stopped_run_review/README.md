# Stopped fsaverage reconstruction review

## Checkpoint and scope

The reconstruction process previously identified as PID 582636 is no longer
present. The saved ledger has 411 PASS, 37 FAIL, 4,088 PENDING and 588
BLOCKED_COREGISTRATION_QA records (5,124 total). All 37 failures are SESAME
neighbourhood-construction failures, affecting test sub-01, sub-02, sub-04 and
sub-05 (11, 6, 9 and 11 fits respectively). This is not a sub-03 failure.
No existing scripts, results, coordinates, thresholds, or solver settings were
changed in this review. No forward or inverse calculation was rerun.

## Confirmed source-resolution incompatibility

The original native source space has 8,196 vertices; the precomputed fsaverage
ico-5 space has 20,484. Therefore the experiment changes source discretization
as well as anatomy. The inherited SESAME `prepare_sesame` call leaves `radius`
unspecified. `initialize_radius` chooses 0.01 metres for these coordinates.
`_compute_euclidean_neigh_matrix` imposes a hard-coded upper limit `n_max=100`
and minimum `n_min=3`, counting the source itself among its neighbours.

A lightweight cKDTree check of saved sub-01 native and fsaverage source
coordinates produced the following results. No dense all-pairs matrix or
posterior/inverse calculation was needed. Full radius checks are recorded in
`01_neighbourhood_geometry.csv`.

| Geometry | Radius (mm) | Neighbours min / median / max | Points exceeding 100 | Connected components |
|---|---:|---|---:|---:|
| Native | 10 | 8 / 35 / 86 | 0 | 1 |
| Fsaverage | 10 | 28 / 73 / 146 | 2,340 | 1 |
| Fsaverage | 9 | 23 / 56 / 118 | 135 | 1 |
| Fsaverage | 8 | 18 / 43 / 89 | 0 | 1 |

The 10 mm fsaverage graph is connected and meets the minimum-neighbour rule.
It fails the library capacity guard. The graph is Euclidean, as used by the
inherited free-orientation SESAME branch; this check does not assert geodesic
cortical connectivity or validate posterior sampling. Other test forwards use
the same template anatomy under rigid transforms, but these exact counts were
measured on sub-01 only.

## Minimal correction to review

Prefer investigating a **capacity-only SESAME compatibility adapter** while
retaining the 10 mm radius, neighbour membership, transition probabilities,
particle count, seed, priors and selected inverse parameters. The inspected
library function allocates `n_matrix` from the observed maximum neighbour count;
100 is a guard rather than its array width. Raising that guard to accommodate
the actual graph is therefore a more faithful candidate than shrinking the
radius. It still needs implementation review, equality checks on supported
small graphs, and a single pilot before any full retry. This review has not
implemented or validated such an adapter.

An explicit 8 mm radius passes the geometry checks, but changes SESAME's
proposal neighbourhood and default `neigh_std=radius/2`; it is a methodological
variant, not a transparent fix. Rebuilding a coarser fsaverage source space
would change all methods' discretization and require new forwards and inverse
fits in a separate experiment. Existing 20,484-source results must not be
silently merged with a coarser experiment.

If a compatibility adapter is implemented, preserve the current ledger as the
original attempt. Retry only the failed SESAME identities in a new output,
with new script hashes, explicit compatibility provenance and merge records.
The existing runner intentionally does not retry FAIL on resume. Its manifest
also rejects changed code; do not bypass that check. Pending work with changed
code should use a new output with explicit continuation/import provenance.

## Why it is slow

37 completed Continuous-ECD fits consumed 6,898.80 seconds (mean 186.45 seconds),
about 60% of the 11,421.09 seconds summed across all recorded attempts. Recorded
runtimes may include earlier pilots/resumes and are not one process's elapsed
time. ECD uses continuous dipole fitting with the fsaverage BEM; it does not
search the cortical source grid, so reducing source resolution alone would
not remove this dominant cost. No settings should be changed to accelerate it
without a separately labeled analysis.

Dense SESAME neighbour preparation computes `cdist(src, src)`: one float64
20,484-square matrix is about 3.36 GB, compared with 0.54 GB at 8,196 sources
(decimal GB, excluding other arrays). Source-count growth is 2.5-fold but this
allocation grows about 6.25-fold. Repeated EEG loading, covariance calculation,
checkpoint rewriting and per-fit setup also add overhead. Actual memory use
and eventual successful SESAME runtime cannot be predicted from failed fits.

## Recommended continuation

Keep the experiment stopped until the SESAME compatibility adapter and its pilot
are reviewed. Retain all current PASS/FAIL records and the unchanged sub-03
block. Do not replace 10 mm with 8 mm or regenerate source spaces automatically.
After a successful pilot, decide between continuing the current source-resolution
experiment with a documented adapter or defining an entirely new coarser-source
experiment. Preserve the fact that original versus fsaverage discretization
and sensor geometry differ when making anatomy-only claims.

Evidence: saved Script 04 forward status/source counts; Script 05 status ledger
and failure tracebacks; unchanged no-warp `Engine.sesame`/`Engine.solve`; installed
`sesameeg/mne/mne_utils.py::_prepare_neighbours_matrix` and
`sesameeg/utils/smc_utils.py::{initialize_radius,_compute_euclidean_neigh_matrix}`.

# Fsaverage experiment plan

The current objective is to reproduce the existing no-warp cross reconstruction
with fsaverage anatomy. Reuse the frozen 61-run schedule, 84 LOSO selections,
84 Single Reference selections, reference identities, channel-selection and
bad-channel-replacement policies, 12 inverse methods, preprocessing, and solver
settings. Reference identities remain parameter/channel-selection identities.
LOSO parameters are selected by held-out test subject; Single Reference parameters
are selected from the reference subject's own runs.

The anatomical substitution requires new forward calculations with fsaverage
BEM/source space and test-subject sensor geometry under the existing candidate
rigid test-head-to-fsaverage transform. This also changes sensor geometry from
the old reference geometry to test geometry; it must be stated when interpreting
an anatomy-only comparison. No new alignment, sensor relocation, scaling, MRI
registration, or affine/SyN mapping is fitted by Script 04.

## Incremental implementation

1. Existing Script 01 freezes schedules/configurations and inventories historical
   baseline, no-warp, affine, and SyN comparisons. Keep these outputs intact.
2. Existing Script 02 supplies candidate rigid transforms and numerical QA.
   Sub-03 remains blocked; the other six retain their review-required status.
3. Script 03 MRI ground-truth registration is an optional separate operation.
   It is not a prerequisite for the new forward/reconstruction workflow and
   must not run automatically. Interrupted/failed registration artifacts are
   not accepted mapping inputs.
4. New `04_make_forward.py` validates inputs and computes one 256-channel EEG
   forward per eligible test subject using existing fsaverage BEM and source
   files. It retains all 5,124 job identities, including 588 blocked sub-03 jobs.
   Audit mode performs no forward/inverse calculation. Build mode is run locally
   by the user. Every invocation requires a fresh output directory.
5. Next, implement reconstruction by reusing the no-warp engine and method
   dispatch. Preserve reference-based nominal montage selection and replacement,
   then select matching channels from the test/fsaverage forward. Continuous-ECD
   must receive the fsaverage BEM with the same test-head-to-template transform.
6. Reproduce no-warp evaluation first: compare returned model-HEAD estimates to
   test ground truth converted from its native surface frame through its own
   H5. Label these unregistered cross-anatomy values **diagnostic distances**,
   without anatomical-error validity claims. Do not automatically apply MRI
   registration or inherited affine/SyN mappings.
7. Retain primary versus whole-brain failure-sensitivity policies and report
   comparison cohorts without deleting failed/blocked cases or duplicating the
   historical within baseline across six references.

No expensive computations are executed during implementation. Subsequent scripts
will be implemented incrementally after the preceding stage is reviewed.

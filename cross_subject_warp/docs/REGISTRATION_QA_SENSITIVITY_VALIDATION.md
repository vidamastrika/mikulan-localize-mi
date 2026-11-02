# Registration-QA sensitivity implementation validation

## What was checked

The new driver is `scripts/06_registration_qa_sensitivity.py`; its policy,
provenance, field diagnostics and comparison logic are in
`scripts/qa_sensitivity/`. This is a separate exploratory branch of the
analysis. Existing Script 06 code, primary results, source tables and transform
files were preserved.

The final check used Python's `unittest` discovery across
`cross_subject_warp/tests/`: **67 passed, 0 failed, 0 skipped** (1.584 seconds).
It covers the original synthetic regression suite and the new sensitivity
tests. `py_compile` also passed for the new driver, helper modules and focused
tests.

The focused tests exercise:

- Five regressions against available historical outputs: fixed source hashes,
  saved mapped row and matched-cohort parity, diagnostic point counts, saved
  threshold classification without primary promotion, and within-subject
  agreement with no-warp results.
- Ten synthetic mechanics/edge tests: immutable independent pair/source
  thresholds; exact boundary behavior; centred affine and LPS handling;
  physical vector-field reading and approximate inverse closure; native-control
  orchestration serialization; detection of changed common-cohort errors or
  lost validity; pair-threshold relaxation while preserving the image gate;
  symmetric, nearest-face boundary probe generation; and separate loading of
  forward/inverse field values; plus field-outside exclusion with the original
  sample denominator and coverage calculation.
- The existing test suite's coordinate, selection, native-data, point-mapping,
  registration-QA and overwrite checks.

Historical data results were not edited to make tests pass. The previously
saved 0.1 mm primary cohort remains authoritative. In particular, an old
SyN failure is only eligible for a new `PASS_QA015_ONLY` label after the user
executes a complete new QA stage and every unchanged gate passes.

## Not run and limitations

The completed attribution directory was read-only inspected. The single
blocking input point maps to approximately MRI voxel `[143, 255, 184]` in a
256-cubed image; it lies on the final voxel plane, and the corresponding
field-grid affine entries differ from the MRI affine by at most about
`1.9e-9`. This supports a grid-boundary/coordinate-precision question but does
not establish ANTs boundary behavior or waive the domain requirement.

The user-run corrected boundary diagnostic found the original point outside
both fields. At inward probes of 0.01 and 0.25 voxel, both field lookups were
inside; ANTs/reader discrepancies were below 0.0005 mm and composite/staged
return differences were below 0.0011 mm. The exact outside point remains
invalid; these probes do not establish QA acceptance. The code excludes
field-outside points from scored roundtrips while retaining the 200-point
denominator and requiring 75% valid coverage. The revised audit passed all
12/12 directions; it excluded one MRI-interior sample, retained at least 93%
field-domain coverage per direction, and confirmed the six >0.1 mm samples are
in-domain. The subsequent user-run all-pair QA completed 84/84 under the 0.15 mm
pair threshold, reproduced all 84 original 0.1 mm flags, and kept the
source-point threshold at 0.1 mm. Mapped sensitivity evaluation then completed
21,960 rows with zero mapping execution errors and zero point-validation
exclusions; the 30 and 22 native reconstruction failures remain recorded for
the two peak scenarios. The cross-policy comparison has not run yet. The
historical saved samples alone classify
78/84 primary checks as passing and show 6 valid points above 0.1 mm but none
above 0.15 mm; that retrospective threshold classification is not a new QA
result. It must not be described as 84/84 validated.

The first user-run boundary diagnostic had a code defect: it read both the
forward and inverse displacement probes from the forward field. Its output at
`outputs/06_registration_qa_boundary_diagnostic/` is preserved, but its
inverse-reader comparison is invalid. The helper was corrected to load the two
fields separately, and the user's corrected rerun is preserved at
`outputs/06_registration_qa_boundary_diagnostic_review/`. A synthetic test
checks that forward and inverse fields are loaded separately.

The controlled fixtures test software mechanics only and are not evidence of
anatomical registration accuracy. A real residual may reflect multiple causes;
the available saved displacement fields may not identify how the historical
continuous inverse was estimated. If independent controls, composition or
domain checks do not pass, downstream stages are gated. The optional ITK/SimpleITK
Python backends are absent in the current environment; the implementation uses
the field reader only after its analytic controls, and reports the native
control as a required user-run gate. No package was installed.

No measured code-coverage percentage is reported. The tests cover targeted
branches, not every line, external command behavior or all real field boundary
configurations. Expensive inverse reconstruction and registration fitting are
excluded. The revised attribution, all-pair QA, mapped evaluation and
cross-policy comparison have been run by the user with the original resource
tree and `localizemi-registration` ANTs runtime. These real-data stages are not
run by the routine test suite.

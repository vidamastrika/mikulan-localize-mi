# Readiness after preparation and LOSO selection

No required source files are missing in Script 01's checked inventory. The
raw grids, solver snapshot, baseline, dataset, seven model/BEM sets, canonical
inverse tables and all 42 affine/SyN resource chains are available. This is file
availability, not a claim that the old numerical results are validated for reuse.

Script 02 uses the verified frozen grids and fresh Script 01 plans. It can run
without solving the physical coordinate bridge because it only reselects
parameters from prior within-subject errors. It preserves strict training-only
repair, duplicate handling, all-PASS screening, subject-balanced ranking and
method experiment allowlists. Affine and SyN use the same winners/inverse fits.
See the commands at the top of `scripts/02_select_loso_parameters.py`.

## Completed Script 02 run

Ran against `outputs/01_prepare_experiment_review/` with default
`primary-all-pass` reliability. All 84 method/fold selections completed and the
selected-parameter CSV is byte-for-byte identical to the historical no-warp
CSV (SHA-256 `77369c7a046e55fad23e2b13590c4106c7f0868e9abae63430e261dc6f0e02d5`).
Configuration keys, parameters, training membership, references and scores were
also compared explicitly. Historical selections were read only after selection
finished, for validation. The comparison is recorded locally as
`outputs/02_select_loso_parameters/09_historical_selection_comparison.json`;
this additional post-run check is separate from Script 02's standard 01–08
artifacts.

Three regression tests passed: exclusion of held-out failures/conflicts/missing
settings, blocking training failures/conflicting duplicates, and unchanged
historical selection functions/policy constants. Existing-output refusal and
Python compilation checks passed. No inverse or registration fitting ran.

## Evidence still needed

| Issue | Observed evidence | Remaining work |
|---|---|---|
| Contact units | All seven surface and all seven T1w coordinate JSONs declare `iEEGCoordinateUnits: m`; `fx_bids.py` exports original coordinates divided by 1,000 | Units are now documented as metres. Confirm anatomical frame/origin separately |
| Native head/surface physical bridge | User-run bridge helper passed all seven subjects using archived pial VolGeom, actual MRI headers, H5/forward transforms and released contact consistency | Use explicit matrices in `outputs/06_coordinate_bridge/02_native_physical_bridges.json`; registration mapping QA remains separate |
| Surface to named T1w coordinates | Saved stage 13 matches inverse H5 to released electrode coordinates for sub-01/sub-02, but labels itself diagnostic only | Extend geometry validation to every subject; a TSV space name is insufficient evidence of scanner/world RAS |
| Registration MRI provenance | Current input MRIs and copied resource files are hashed; saved reports name original MRI paths | Compare against recorded fitting-time MRI identity/hashes; review ANTs command and version evidence. Unverified does not mean those records are absent |
| Registration QA | Matrices, full SyN chains, warped images and saved QA exist for all 42 pairs | Validate grids, physical point directions, RAS/LPS conversion and roundtrips; foreground overlap alone does not establish cortical correspondence |
| Canonical native records | Primary and sensitivity tables are now imported with verified EEG/channel fingerprints, configurations, forward/H5/contact/BEM identities and recorded software versions; see [completion report](NATIVE_DATA_COMPLETION.md) | Exact intermediate patch executable is not archived; no independent solver rerun is claimed. Physical-coordinate validation remains separate |
| Peak-selection policy | Historical cortical results use known-hemisphere extraction; separate whole-brain sensitivity variants exist | Preserve that policy in primary results and keep sensitivity separate. Do not silently reinterpret these grids as a uniform whole-brain analysis |

The contact-unit check refines Script 01's earlier generic unresolved assumption;
its immutable reports are preserved. Sidecars live at
`DATASET/derivatives/epochs/sub-XX/ieeg/sub-XX_task-seegstim_space-{surface,T1w}_coordsystem.json`.
These specify `Other` / native MRI or surface space, which does not fully define
the scanner/world transform. The cancellation of head-to-surface followed by its
inverse in historical mapping cannot itself establish a NIfTI physical frame.

## Next work

Native LOSO and reference-only datasets are complete in the new final and
sensitivity folders. The post-completion reuse audits pass with no metadata
warnings; see [native dataset completion](NATIVE_DATA_COMPLETION.md). Preserve
remaining solver failures and do not substitute sensitivity peaks into primary.

Native geometry checks pass for all seven subjects. Registration QA passes
all 42 affine and 36/42 SyN pairs; six SyN pairs each have one sampled point above
the unchanged 0.1 mm threshold. Their flags remain failed. The user approved
continuing with retained failures: map only eligible mode/pairs, keep every
scheduled row, leave invalid errors blank, and check individual mapped points.

Script 06 mapped evaluation is now written and checked with synthetic data; it
has not been run on subject inputs. Use the terminal command in
[NEXT_STEPS.md](NEXT_STEPS.md). Native `status` stays unchanged, while
`evaluation_status`/`localization_error_valid` govern metric summaries. Matched
affine/SyN comparisons require both modes valid on the same key and must report
coverage. No fitting-time provenance was invented and no solver/registration
refitting was performed.

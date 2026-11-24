# Maskface template variant — preparation report

Status: **STAGE_1_PREPARED_NOT_EXECUTED**. Stages 2–3 are deferred until the generated feasibility findings are reviewed. Run the command in README.md to create `outputs/01_input_feasibility/run_01/REPORT.md` and its reproducibility manifest.

## Existing evidence

Read the completed anatomical-coverage review. Facial removal predates the original template construction, and an existing maskface representation provides more facial signal. The native maskface sidecars cite Milchenko and Marcus; export code copies `T1_mf.nii` into these derivatives. Original native surfaces were reconstructed from original MRIs according to the release README.

The cited method modifies surface-layer signal rather than merely preserving an untouched face. Exact release-specific algorithm variant, software version, parameters, altered-voxel masks and brain-protection settings are unavailable. See [method evidence](docs/METHOD_EVIDENCE.md), including the primary paper link. Additional coverage must not be treated as verified original anatomy.

Historical repository scripts invoke watershed on maskface inputs. This records a past intended workflow, not successful reconstruction, anatomical validity or FreeSurfer compatibility for all seven released images.

## Stage 1 preparation

The new script locates all seven MRI pairs by explicit patterns; hashes sources and prior evidence; audits qform/sform, dimensions, spacing, units and FOV; compares sampled paired voxels when grids match; and exports image/surface signal profiles in facial, lateral and cranial control regions. Native GIFTI volume geometry must match the actual Pydeface MRI before surface overlays are used. Maskface images are sampled using their own actual affines, without fitting a new transform.

Common-signal equality is measured directly. Maskface-only signal remains unverified as original-versus-filtered anatomy because no unmodified reference is used. Brain inward controls and skull/surface profiles are explicitly proxies, not tissue masks or segmentation validation.

Technical readability can support consideration of an exploratory FreeSurfer reconstruction, but Stage 1 does not run recon-all or watershed and cannot prove reconstruction success, anatomical scalp fidelity, layer nesting or source-space containment. All successful Stage 1 runs still require review and leave Stage 2 unauthorized.

## Validation and preservation

Synthetic tests passed for header/affine ambiguity, anisotropic physical MRI sampling, signal-support states, paired-grid comparisons, region rules, script import and syntax. The manual script's help command also passed. No participant analysis was run during preparation.

Original anatomy/results/diagnostics and thresholds are unchanged. No new template, surface, reconstruction, head scaling, alignment, forward or inverse computation was performed. The original settings are pinned in the separate variant configuration for potential later reuse after review.

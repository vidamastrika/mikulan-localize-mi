# Failures and sensitivity

## Primary failures

The final LOSO merged audit records 1 FAIL among 1,464 planned condition fits: Gamma-MAP, test sub-05 run-08, reference sub-01. `outputs/03_reference_based_final/07_known_failure.json` preserves the known failure. The single-reference final output records 14 FAIL among 4,392 cases; see `outputs/05_single_reference_final/03_failed_cases.csv` for case-level causes. These are solver/runtime failures, not zero localization errors.

## Whole-brain peak sensitivity

The original peak selection is restricted to the known stimulation hemisphere for cortical methods. The separate whole-brain sensitivities select across both hemispheres, retaining selected parameters and labeling the variant. The LOSO sensitivity addresses the sole known Gamma-MAP failure; the single-reference analysis identifies candidate failed cases and records its overrides in `05_override_audit.csv`. These selectively retried variants are mixed-policy datasets and should not be treated as independent complete replications.

## Interpretation

Raw no-warp cross-subject separations can change because source model anatomy/head geometry and selected parameters vary with reference. The design does not isolate those effects causally. Missing distances for failed fits are excluded from distance summaries; pass rates must be read alongside distance distributions. Numerical frame checks do not resolve anatomical alignment assumptions or contact-unit uncertainty.

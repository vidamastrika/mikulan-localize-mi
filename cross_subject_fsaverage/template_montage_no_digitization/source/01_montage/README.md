# Template montage evidence and snapshots

No correspondence is presumed for `e1..e256`. `E1..E256` name matches are reported
as **unvalidated lexical candidates**, not physical electrode matches. Do not
construct an evidence mapping from numbering similarity, localization outcomes,
test sensor positions, array indices or nearest fsaverage_1005 sensors.

The installed MNE resources will be inventoried by Script 01. Raw candidate
files are preserved separately from validated geometry. Mapping evidence must
establish the Localize-MI acquisition net model and numbering for all seven
subjects, the exact MNE template version, and its coordinate/fiducial conventions.

## Optional evidence JSON schema

Supply a JSON object with these fields (no example approval is supplied):

| Field | Type / meaning |
| --- | --- |
| `montage_kind` | Exact `GSN-HydroCel-256` or `EGI_256` name |
| `template_resource_sha256` | Hash of the installed raw template resource, reported in the availability CSV |
| `subjects_covered` | List containing exactly sub-01 through sub-07 |
| `reviewer` | Person responsible for the documentary review |
| `reviewed_utc` | Review date/time |
| `dataset_net_model` | Documented acquisition net model, not a guess from names |
| `numbering_correspondence_explanation` | Explanation linking dataset acquisition channels to physical template electrodes |
| `fiducial_interpretation` | Documented meaning of the template landmarks, including MNE-derived/approximated landmarks |
| `unit_and_scaling_interpretation` | Raw coordinate convention and justification for the fixed MNE auto normalization |
| `independent_of_subject_geometry_and_localization_outcomes` | Must be the Boolean true following review |
| `template_head_size_auto_095m_accepted` | Must be the Boolean true following review |
| `fiducial_convention_verified` | Must be the Boolean true following review |
| `mapping_csv` | Path to explicit 256-row correspondence CSV, relative to the evidence JSON or absolute |
| `mapping_sha256` | Hash of that CSV |
| `documents` | Documentary records as specified below |

Each `documents` record needs `id`, `role`, `path`, `sha256`, `citation`, and
`relevant_excerpt_or_section`. Documentary roles must cover:
`dataset_net_and_numbering`, `template_numbering`,
`fiducial_coordinate_convention`, and `template_unit_scaling`. A publication or
manual may support several roles using separate records. Paths resolve relative
to the evidence JSON; provide local documentary artifacts with authoritative
citations. URLs alone are not a substitute for the hashed local artifact.
Do not provide participant coordinate files as documentary evidence.

The mapping CSV requires exactly `dataset_channel`, `template_channel`, and
`evidence_id` columns. Dataset channels must be exactly `e1..e256`; template
labels must be a bijection onto the selected resource's exact channel names.
Every row's evidence ID must identify a supplied documentary record. Extra,
missing, duplicate or ambiguous correspondences block readiness. Explicit
correspondence can be nonidentity if documentation establishes it; the program
does not infer physical identity from numbering.

The program checks structure and file integrity, not the truth of a scientific
claim in a document. The named reviewer remains responsible for that assessment.
Passing documentary validation enables geometry inspection; it does not grant
anatomical validity or permission to begin a forward-modeling stage automatically.

# Localize-MI / EGI template compatibility: read-only audit

**Strict scientific recommendation: BLOCKED.** `GSN-HydroCel-256` is the best
supported *candidate family*, but the available evidence does not independently
validate the physical correspondence of all Localize-MI `e1`–`e256` channels.
The existing strict Script 01 BLOCKED report is retained. Existing exploratory
results remain `ASSUMED_UNVERIFIED`; this audit does not erase or validate them.
No new forward modeling or Script 02 execution is authorized by this audit.

This package is inside the existing variant, not a second experiment. Script 01
and every previous result are unchanged. No preparation/geometry audit program,
preprocessing, forward, inverse, registration or sensor fitting was executed.
The geometry figures/numbers below are explicitly reused from prior audits.
New audit code was syntax-checked only and is provided for manual execution.

## 1. Acquisition evidence and the remaining identity gap

The original paper identifies a 256-channel HydroCel CleanLeads net and NA-400
amplifier. Its acquisition software was custom C++/MATLAB built around
AmpServerPro. This establishes the acquisition family and amplifier; it does not
identify a complete net part/revision/cap-size record or the custom exporter's
physical channel lookup. [Mikulan et al., 2020](https://www.nature.com/articles/s41597-020-0467-x)

The authors' public `import_egi_sdk` transposes the saved array, separates its
last row as the trigger and assigns ascending EEG labels. It does not document
how the earlier acquisition program arranged physical sensor identities in that
saved array. The inspected local copy agrees with the public source.
[Author importer](https://github.com/iTCf/mikulan_et_al_2020/blob/master/fx_preprocess.py)
[Author BIDS exporter](https://github.com/iTCf/mikulan_et_al_2020/blob/master/fx_bids.py)

The previous metadata audit inspected all 61 released channel tables: ordered,
unique `e1`–`e256`, without an independent hardware-electrode ID column. This
supports label consistency, **not** physical identity. No measured sensor
positions, EEG waveforms or stimulation outcomes were used to infer a mapping.
See the prior documentary review and its 61-row channel metadata inventory.

`channel_mapping_evidence_audit.csv` lists all 256 proposed `eN → EN` pairs as
**unverified hypotheses**. A shared suffix or row index is never counted as
correspondence evidence. No nearest-neighbor matching is performed.

The missing evidence can be an acquisition/export channel-map record, a raw
header plus independently documented exporter lookup, or equivalent archived
source showing how each saved channel maps to a physical HydroCel sensor for
these participants. New author contact is not required by this audit, and no
contact was made. Generic SDK conventions alone do not prove this custom export.

## 2. Candidate MNE resources

The experiment is frozen to **MNE 1.12.1**. The table below reflects that
installation and existing recorded availability, not whatever a current online
example happens to install.

| Candidate | Frozen availability / labels | Frame and construction | Audit recommendation |
| --- | --- | --- | --- |
| `GSN-HydroCel-256` | 256 EEG locations, `E1`–`E256`; three named fiducials | SFP; MNE native `unknown`, then template fiducial HEAD definition; fixed auto normalization | Relevant HydroCel-family candidate; dataset mapping unverified |
| `EGI_256` | 256 EEG locations, also `E1`–`E256` | Separate CSD/spherical resource; electrode-derived fiducials | Not interchangeable merely because numbers match |
| `fsaverage_1005` | **Absent in installed 1.12.1** | Introduced in 1.13; anatomical 10-05 labels, not HydroCel EN identities | Not an EGI substitute; do not upgrade the frozen run to obtain it |
| `standard_1005` | 343 EEG locations, international anatomical labels | Legacy MRI-labelled resource; later clarified as Colin27 | Control only; not an EGI map and not identical to new `fsaverage_1005` |

The version change matters: MNE 1.13 added `fsaverage_1005` and clarified the
older `standard_1005` name as `colin27_1005`. A new fsaverage-aligned 10-05
resource does not establish which 10-05 point corresponds to an acquired EN
channel. Arbitrary renaming or nearest-neighbor selection is prohibited here.
[MNE 1.13 release notes](https://mne.tools/dev/changes/v1.13.html)

The pinned HydroCel SFP hash is
`63d3de763f3bd36bb9096989a828068edcb60fea81607ddb96022d3bd2126e1c`.
The separate EGI CSD hash is
`e2ce1ebcbe8cdd2aaf77c536faea67e5a1a2c601eb327407ee4034a9d33a078e`.
The prior audit verified the HydroCel records against FieldTrip's EGI-origin
resource. FieldTrip documents the even-count/odd-count distinction: the even
256 resource excludes the separate vertex-reference position, whereas 257
includes it. Do not add an acquired Cz channel or rename E256 to Cz.
[FieldTrip template provenance](https://www.fieldtriptoolbox.org/template/electrode/)

For `EGI_256`, the installed MNE parser derives nasion from E31 and approximates
LPA/RPA from weighted E67/E94 and E219/E190 positions. HydroCel loads the SFP's
FidNz/FidT9/FidT10 records instead. These conventions affect the HEAD definition
as well as the sensor coordinates. Different resources/fiducials must not be
silently substituted. [Version-pinned MNE parser](https://github.com/mne-tools/mne-python/blob/v1.12.1/mne/channels/_standard_montage_utils.py)

A Localize-MI-specific MNE forum discussion also reports confusion between the
older GSN and HydroCel resources. It is useful context, not a dataset hardware
record or independent export-map proof, and does not change the strict block.
[MNE discussion](https://mne.discourse.group/t/egi-256-and-hydrocel-256-montages/11677)

## 3. Units, frames and fixed transforms

MNE's loaded electrode coordinates are in metres. For these spherical templates,
`head_size='auto'` applies the predefined 95 mm normalization. That operation
is a model convention, not evidence of a raw SFP physical unit, cap size or
validated scalp fit. The CSD contains unit-sphere information; the inspected
SFP records do not themselves establish a dataset-specific absolute scale.
[MNE 1.12 API](https://mne.tools/1.12/generated/mne.channels.make_standard_montage.html)

The existing HydroCel chain is:

1. Template-native coordinates → HEAD, using only the template fiducials.
2. HEAD → fsaverage MRI/surface RAS, using MNE's supplied `fsaverage-trans.fif`.
3. FreeSurfer scalp vertices, read in mm, are converted to metres for QA.

MNE supplies a predefined fsaverage HEAD/MRI transform; the official template
forward example uses it. It is **not** an individually fitted transform or a
HydroCel-specific certificate of anatomical alignment.
[MNE template-forward example](https://mne.tools/1.12/auto_tutorials/forward/35_eeg_no_mri.html)

An additional rigid fit is **not mathematically required** to convert these
coordinate frames. Whether a fixed HydroCel-to-fsaverage placement is
scientifically adequate is a separate question that the current scalp QA does
not settle. Do not add ICP, scale search or projection to make a numerical gate
pass. A future validated, common template-only transform would be a separately
specified modeling choice, never a test-subject fit and never optimized using
stimulation outcomes. No such new transform was fitted here.

The frozen matrices and normalization are preserved in
`existing_geometry_summary.json` and the original Script 01 provenance JSON.
Existing transformed fiducial residuals are **12.54 mm nasion**, **5.01 mm LPA**
and **5.01 mm RPA**. These are convention/alignment diagnostics, not an
anatomical PASS or a reason to silently redefine landmarks.

## 4. Existing geometry QA — reused, not recomputed

These numbers refer to the prior assumed HydroCel candidate with its unchanged
fixed placement, not to independently validated EEG-channel correspondence.

| Quantity | Dense fsaverage outer scalp |
| --- | ---: |
| Nearest-vertex median | 6.827 mm |
| Nearest-vertex p95 | 14.897 mm |
| Nearest-triangle median | 6.691 mm |
| Nearest-triangle p95 | 14.805 mm |
| Maximum nearest-triangle distance | 16.426 mm |
| Inside / outside / on surface | 191 / 65 / 0 |

The dense mesh was closed, consistently oriented and connected in the existing
review. Self-intersections were not checked. The original nearest-vertex gates
(median ≤10 mm, p95 ≤20 mm) passed; those numerical gates do not validate
physical channel identity or anatomical placement. Vertex distances are upper
bounds on triangle distances. The inside/outside counts come from the prior
triangle/winding audit, not merely a nearest-vertex normal sign.

[Existing visual QA — copied without rerendering](existing_hydrocel_fsaverage_visual_QA.png)

The actual BEM scalp has a different mesh. Its saved audit counted 190 locations
inside and 66 outside. MNE's native EEG interpolation produced effective
location displacements of 6.705 mm median, 14.770 mm p95 and 16.322 mm maximum.
Fourteen channels used triangle-plane extrapolation; effective points were at
most 0.0853 mm from that BEM scalp. This is documented solver behavior, not an
input coordinate repair or proof that the original montage is anatomically
valid. Existing input coordinates were not changed. See the pinned local
`mne/forward/_compute_forward.py` and prior actual-BEM report.

Fresh candidate-by-candidate geometry and a comparison figure will be generated
**only when you run the new audit command**. They use template identities only,
without assigning verified Localize-MI identities or creating forward inputs.
The code does not project, fit or alter any candidate. Same-name EGI/HydroCel
coordinate differences are descriptive template comparisons, never physical
mapping evidence.

## 5. Experimental meaning and comparison with no-warp

| Condition | EEG | Sensor geometry | Anatomy / placement |
| --- | --- | --- | --- |
| Existing no-warp | Test EEG | Reference subject | Reference anatomy; no affine/SyN |
| Existing fsaverage | Test EEG | Test subject | fsaverage; individual rigid EEG coregistration |
| New template variant | Test EEG | One common template | fsaverage; fixed template coordinate chain; no individual digitization, MRI or sensor ICP |

The 61-run/7-subject schedule, 12 methods, 168 resolved configurations and
5,124 inverse evaluations remain frozen. Reference identities remain parameter
owners; they do not supply sensor positions in the template condition.
Nominal montage sizes and bad-channel policy are retained, but geometrically
selected channel identities can change when the geometry changes. Relative to
no-warp, **both anatomy and sensor geometry differ**; this is not an anatomy-only
experiment. The fixed 95 mm convention and cap/landmark uncertainty can affect
results even if a channel numbering map is later confirmed.

The requested primary comparison remains unregistered HEAD-coordinate distance,
matching no-warp's diagnostic definition. It is not anatomically validated
localization error. Existing MRI-based evaluation-only mappings remain separate;
this compatibility audit does not read them or use outcomes to select a montage.

## 6. Recommendation for Script 01

**No Script 01 change is justified as a validation fix yet.** Its default
verified-evidence mode already blocks missing correspondence. Preserve that
behavior, its thresholds and all frozen outputs. Do not turn the previous
exploratory assumption into a verified mapping or reuse numerical solver success
as electrode-identity evidence.

If independent dataset-linked net/numbering and landmark/scale evidence is
later demonstrated, use Script 01's existing `--mapping-evidence` schema with
new source/output roots. Any necessary change should then be limited to the
reviewed mapping/provenance and the documented fixed template transform, without
changing the frozen experimental design. No alternate candidate, version
upgrade or anatomical fit is authorized by the present evidence.

## 7. Run the compatibility audit locally

From the repository root, with the **existing** virtual environment:

```bash
.venv/bin/python cross_subject_fsaverage/template_montage_no_digitization/audits/compatibility_2026-11-09/audit_montage_compatibility.py
```

It writes only a fresh
`outputs/00_montage_compatibility_audit_2026-11-09/` within this same variant:

- candidate availability, resource hashes, names, frames, scaling and fiducials;
- 256-row mapping evidence audit, still explicitly unverified;
- template-only nearest-vertex/triangle distances, winding classifications and summaries;
- fiducial residuals and same-template-name layout differences;
- a multi-candidate fsaverage-head visual comparison;
- transform provenance, SHA-256 inputs and strict BLOCKED recommendation.

**Expected exit code 3 / BLOCKED is a scientific finding, not a program failure.**
Use a different fresh `--output-root` if that destination already exists.
No preprocessing, forward, inverse, subject coregistration, BEM construction,
MRI registration or Script 01/02 execution occurs. Review the montage evidence
and plots before any new forward-model decision.

## Package files and provenance

`EVIDENCE.json` records claims, links and missing evidence.
`candidate_static_inventory.csv` reuses the initial availability audit and notes
the version-dependent missing montage. `channel_mapping_evidence_audit.csv`
records all unverified proposals. `existing_geometry_summary.json` reuses prior
measured QA, and `inspected_existing_inputs.json` inventories exact source hashes.
The included visual is a byte-exact copy of the original scalp review.

Nature full-page/PDF access redirected to an inaccessible authentication page
in this session; the original indexed PDF methods passage and prior archived
review were available. This limitation is explicit. No acquisition details or
approval flags have been invented, no external messages were sent, and no
compatibility result was selected using localization outcomes.

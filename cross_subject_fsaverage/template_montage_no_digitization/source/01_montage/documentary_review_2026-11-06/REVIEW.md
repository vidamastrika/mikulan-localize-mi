# Documentary verification of the Localize-MI template montage

Review date: 2026-11-06. Result: **partially verified; forward modeling remains blocked**.
No preparation script, geometry QA, preprocessing, forward model, inverse fit or
registration was executed during this review. Existing audit outputs are preserved.

## Findings and evidence

| Question | Finding | Confidence / remaining gap |
| --- | --- | --- |
| Which net family was used? | The original paper's EEG recordings section identifies a 256-channel HydroCel CleanLeads acquisition with an EGI NA-400 and custom AmpServerPro-based acquisition software. | Direct primary evidence supports the HydroCel family; it does not specify a complete sensor-to-export-row lookup. |
| How are dataset EEG rows labeled? | The authors' `import_egi_sdk` transposes the MATLAB data, removes the final trigger row and assigns ascending `e1` through `e256` labels. | Direct source evidence for the public Python import step, not proof of the preceding custom C++/MATLAB acquisition export's physical channel order. |
| Do released channel identities agree? | All 61 local `channels.tsv` tables contain exactly the ordered, unique `e1` through `e256` EEG names. | Metadata only. None contains a separate hardware/sensor-ID column. Waveforms, digitization, MRI and contact positions were not read. |
| Where does MNE's HydroCel resource come from? | Its 259 label/coordinate records exactly match FieldTrip's resource; FieldTrip documents these HydroCel resources as obtained from EGI's FTP server. | Verified lineage for the installed template; this does not independently establish the dataset exporter lookup. |
| Is Cz one of the 256 EEG sensors? | FieldTrip distinguishes even-count templates without the vertex reference from odd-count versions including it. | Do not invent an acquired Cz channel or map `e256` to Cz. Dataset reference/export handling should be confirmed alongside physical numbering. |
| Can the current vendor adult file replace the installed template? | No automatic substitution. The downloaded adult resource contains the same numbered sensors plus a separate Cz, but differs in its raw coordinate records and file hash. | These are different coordinate resources. No fitted transform or normalized comparison was used to select between them. |
| What does MNE do with units and landmarks? | The installed parser treats FidNz/FidT9/FidT10 as nasion/LPA/RPA, and its standard spherical loading uses the documented fixed 95 mm convention. | Software behavior is verified. The normalization is a declared model assumption, not validation of physical raw units or fsaverage fit. |
| Is fsaverage geometry ready? | Not evaluated in this review. | The independent-mapping gate remains in force. Scalp/fiducial geometry must be reviewed separately once correspondence is documented. |

## Primary sources

1. **Localize-MI original paper**, Mikulan et al. (2020), DOI 10.1038/s41597-020-0467-x:
   https://www.nature.com/articles/s41597-020-0467-x
   Relevant section: Methods / EEG recordings. It supports the acquisition family,
   channel count and use of a custom acquisition program. No localization outcomes
   were used in this montage review.

2. **Authors' public importer**:
   https://github.com/iTCf/mikulan_et_al_2020/blob/master/fx_preprocess.py
   Relevant function: `import_egi_sdk`, repository lines 35–76. Local source was
   inspected as text; no importer was called. The public BIDS exporter preserves
   EEG channel names in `channels.tsv` and exports EEG rows without the trigger:
   https://github.com/iTCf/mikulan_et_al_2020/blob/master/fx_bids.py
   Relevant function: epoch/BIDS export. The repository README explicitly states
   that only a subset of the original preparation scripts is public.

3. **FieldTrip template provenance and reference convention**:
   https://www.fieldtriptoolbox.org/template/electrode/
   Relevant section: The EGI geodesic sensor net. Upstream data:
   https://github.com/fieldtrip/fieldtrip/blob/master/template/electrode/GSN-HydroCel-256.sfp
   `external/fieldtrip_GSN-HydroCel-256_records.sfp` is a record-normalized snapshot
   of the fetched source (line endings and trailing whitespace normalized only).
   Its SHA-256 equals the installed MNE resource exactly:
   `63d3de763f3bd36bb9096989a828068edcb60fea81607ddb96022d3bd2126e1c`.

4. **Manufacturer's current sensor-position download page**:
   https://www.magstim.com/3d-sensor-position-files-for-visualizing-data-in-third-party-software/
   It provides age/density-specific average files. The adult 256 download links to:
   https://www.egi.com/images/kb/Sensor_Position_.sfp_files/GS_3_.sfp/AdultAverageNet256_v1.sfp.zip
   The archive and its unmodified SFP member are retained in `external/`.
   The adult file has 260 records: 3 fiducials, E1–E256 and Cz. Its SHA-256 is
   `bf7e2ae893e2dd70c4241c9aa7c81ceb126fc43cff41973d0dc11554c0d22194`.
   All 259 shared records differ in coordinate text from the installed MNE file.
   This is a raw-file distinction, not a statement about equivalence after scaling.

5. **Manufacturer's coordinate-system explanation**:
   https://www.egi.com/knowledge-center/item/93-what-sensor-coordinate-system-do-net-station-and-geosource-3-1-research-use
   This describes a Cartesian convention with right/anterior/superior positive,
   an average-position option, and an origin defined using a fiducial-based sphere.
   It should not be conflated with MNE HEAD axes/origin or fsaverage MRI coordinates.

6. **Manufacturer's sensor-map and fiducial documentation**:
   https://www.egi.com/images/stories/manuals/Printed%20IFUs%20with%20New%20Notified%20Body/HC_GSN_channel_map_256_8403487-54_20181207.pdf
   The map describes numbered HydroCel sensor locations and a distinct vertex
   reference. Its stated amplifier scope is Net Amps 200/300, while Localize-MI
   reports NA-400; the map alone does not resolve the custom export correspondence.
   https://www.egi.com/images/HydroCelGSN_10-10.pdf
   The technical note describes averaged HydroCel locations and anatomical
   fiducials, including preauricular points. It does not by itself establish the
   exact version of the installed SFP or validate fsaverage scalp compatibility.

7. **Version-matched MNE documentation**:
   https://mne.tools/1.12/generated/mne.channels.make_standard_montage.html
   Installed source files in the original preparation's MNE snapshots confirm
   the standard-montage normalization and SFP fiducial parsing. EGI_256 uses
   electrode-derived fiducials and is not silently interchangeable with the
   separate HydroCel resource. Raw template coordinates are not already MNE HEAD.

8. **BESA format documentation**:
   https://wiki.besa.de/index.php?title=Working_With_Additional_Files
   Its surface-point conventions support named fiducial records. This supports
   format interpretation, not dataset-specific physical channel correspondence.

9. **Manufacturer's SDK guide**:
   https://www.egi.com/images/stories/placards/ASProSDK_v21_ugid_8409500-57_20181029.pdf
   It identifies the SDK architecture and refers to a separate network-protocol
   specification. The inspected guide does not document Localize-MI's custom
   acquisition program or a row-to-sensor permutation for its saved MATLAB files.

## What has not been established

`eN -> HydroCel EN` is a plausible, source-supported candidate convention. It is
not promoted to a validated mapping merely because the public importer assigns
those numbers to array rows. The missing link is a dataset-author acquisition
record, channel-map export or explicit confirmation that saved EEG column/row N
corresponds to physical HydroCel sensor N for all seven participants, without a
permutation, reference insertion or omitted sensor.

A separately labeled comparison against measured electrode positions could
provide a diagnostic check, but it would not replace independent correspondence
evidence. No such positions were inspected, fitted or optimized here.

The fixed 95 mm normalization and template-fiducial-to-fsaverage convention are
model choices to declare and evaluate; a document cannot prove the resulting
scalp fit. Documentary identity verification and geometry review remain separate.

## Next action

Obtain the narrow acquisition-channel confirmation in
`AUTHOR_CONFIRMATION_REQUEST.md`. This request has been drafted only, not sent.
Then complete the evidence file using documented facts and explicit model
assumptions, and run Script 01 manually into fresh source/output directories.
Do not change the original BLOCKED report, set unsupported approval flags, switch
templates based on geometry outcomes, or begin Script 02 yet.

See `REVIEW_STATUS.json` for source links, resource hashes and machine-readable
claims, and `channel_identity_metadata_audit.csv` for all 61 metadata inspections.

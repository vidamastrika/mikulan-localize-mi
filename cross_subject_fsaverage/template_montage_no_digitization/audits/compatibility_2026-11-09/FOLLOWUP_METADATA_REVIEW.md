# Follow-up: released acquisition metadata and label provenance

**Result: the strict correspondence block remains.** No new forward or experiment
stage was started, and no original file/result was changed.

The follow-up inspected metadata field names in all **61 EEG JSONs** and column
names in all **61 channel tables**. The JSONs supply description, source reference
and baseline information. The channel tables supply name, type, units, filter
limits, sampling frequency and status. Neither has an independent acquisition net
code, net revision or saved-channel-to-physical-electrode lookup.

No MAT/MFF/RAW/XML/BIN/EGI/CNT/SET/VHDR/EDF acquisition-format candidate was found
in the local released-data tree. This is a local inventory finding, not a claim
that no acquisition archive exists elsewhere. EEG arrays were not opened.

## Additional source-code finding

- `fx_preprocess.import_egi_sdk` generates ascending e1..e256 names after splitting
  EEG and trigger data. It does not establish the custom acquisition export's
  preceding physical channel lookup.
- `fx_preprocess.make_dig_files` parses source digitizer IDs but constructs output
  EEG names from an expected sequence. Therefore the resulting names are not an
  independent identity cross-check. No measured coordinates were matched or used
  to fit the montage in this review.
- `fx_bids.make_electrodes_eeg_tsv` assigns the material string HydroCel CleanLeads
  literally. This corroborates the intended cap family, but is not a hardware
  header identifying the exact acquisition net revision.
- `fx_bids.export_data_bids` inherits epoch channel names for channels.tsv rather
  than adding an independent hardware sensor ID.

These public functions are available in the [author preprocessing source](https://github.com/iTCf/mikulan_et_al_2020/blob/master/fx_preprocess.py)
and [author BIDS exporter](https://github.com/iTCf/mikulan_et_al_2020/blob/master/fx_bids.py).
The local source/metadata hashes and detailed inventory accompany this report.

## Decision

Do not claim that eN is physically HydroCel EN solely from these generated labels.
The required new evidence is an independently documented acquisition/export map
or equivalent archived hardware/export record. No author contact was made.

The previously completed assumption-based experiment remains an exploratory
analysis with the physical mapping unverified. Its results, numerical geometry
and no-MRI-alignment comparison are preserved. They cannot be promoted into a
validated montage experiment by further reruns of the same audit.

No additional experiment command is necessary now. Keep validated forward
modeling blocked; retain existing results under their explicit exploratory label.

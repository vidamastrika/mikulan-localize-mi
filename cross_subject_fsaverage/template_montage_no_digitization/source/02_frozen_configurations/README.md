# Frozen design snapshots

Script 01 will populate this folder when executed locally. It has not run yet.
Snapshots include the 61-run schedule, seven LOSO folds, 366 single-reference
run/reference identities, both 84-row selections and their manifests, the
original 5,124-job plan, all 168 resolved configurations, and source-code
references for preprocessing, peak extraction and channel-selection policies.
Every snapshot receives an original path, copied path, SHA-256 and byte size in
the output inventory. Original files are never modified.

The policy code is not executed by preparation. The old loader/Engine uses
individual forward/anatomical files and requires an EEG-only adapter in a later
stage. No participant geometry or localization outcomes are used to validate the
new montage. Selected parameters remain unchanged despite geometry substitution.

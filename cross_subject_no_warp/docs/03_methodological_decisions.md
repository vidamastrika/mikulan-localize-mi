# Methodological decisions

- **Test versus reference:** Test identity indexes EEG runs and ground truth. Reference identity selects a donor head model. They must not be conflated.
- **LOSO versus single-reference selection:** Script 02 excludes each test subject before screening/ranking and selects one configuration per method/fold. Script 04 selects separately for each reference subject using its other six participants. Reference choice can therefore change both the head model and parameters.
- **No warp:** Cross-subject coordinate values are raw head-frame separations. They are not anatomical localization errors. Only within-subject comparisons support that interpretation, subject to coordinate and ground-truth validity.
- **Peak policy:** The primary reconstruction retains the inherited known-stimulation-hemisphere restriction for cortical peaks. The whole-brain variant removes that restriction for selected failed cases only; it is a sensitivity analysis and does not replace primary results.
- **Electrode TSVs:** The experiment does not use problematic EEG electrode TSV coordinates to construct source models or select references. Coordinate preparation checks contact table schema but explicitly treats its units as an assumption; evaluation uses stimulation-contact information and performs numerical audits.
- **PASS meaning:** PASS indicates the solver returned an output. It does not demonstrate convergence, reproducibility, or anatomical validity.

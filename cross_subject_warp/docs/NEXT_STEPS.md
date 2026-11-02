# Next steps

Scripts 01–08 and the separate registration-QA threshold sensitivity work are
complete. The six SyN failures remain failures under the primary 0.1 mm rule;
the 0.15 mm analysis is separately labeled and retains the same source-point
validation. Both policies preserve all rows and invalid errors as blank.

Script 07 was run on the experiment tables. The corrected comparison is in
`outputs/07_compare_strategies_corrected/`. To rerun it from the repository root:

```bash
python cross_subject_warp/scripts/07_compare_strategies.py \
  --output cross_subject_warp/outputs/07_compare_strategies_new
```

It reads completed CSVs only and writes `outputs/07_compare_strategies/`.
To preserve a completed run, it refuses an existing output directory; select a
new location with `--output cross_subject_warp/outputs/07_compare_strategies_review`.
The tables keep primary and peak-sensitivity observations, affine and SyN, and
the 0.1 mm and 0.15 mm QA policies distinct. Raw unregistered cross-subject
head-frame separation is not treated as anatomical localization error.

Script 08 has completed successfully. While preparing visualization, a join
key error was found in Script 07: strategy labels had been included in the
LOSO-versus-single-reference match key, so those rows could not match. The
script is fixed; regenerate its tables into a fresh folder (the original output
is preserved):

```bash
python cross_subject_warp/scripts/07_compare_strategies.py \
  --output cross_subject_warp/outputs/07_compare_strategies_review
```

The corrected table contains `_merge == both` matched rows. Audit and plot
with Script 09:

```bash
python cross_subject_warp/scripts/09_visualize_results.py --phase audit \
  --comparison-root cross_subject_warp/outputs/07_compare_strategies_corrected
python cross_subject_warp/scripts/09_visualize_results.py \
  --comparison-root cross_subject_warp/outputs/07_compare_strategies_corrected \
  --output-root cross_subject_warp/outputs/09_visualize_results_new
```

Script 09 passed its experiment-table audit and ran with
`outputs/07_compare_strategies_corrected/` as input. Figures and CSVs are in
`outputs/09_visualize_results_single_reference/`. Its master figures compare the historical
within-subject baseline with matched LOSO affine/SyN errors and, separately,
single-reference affine/SyN medians across common valid references. Each has a
primary 0.1 mm panel and a separate 0.15 mm QA sensitivity panel. The
single-reference summary uses the test-run as its unit and does not count six
references as six independent runs. Script 08 summarizes six single-reference
results per method/test/run, retaining the two QA policies, registration modes
and peak scenarios. Reference anatomy and the reference-selected configuration
both vary, so the summaries describe reference-choice sensitivity rather than
isolating an anatomy effect. No new fitting or inverse reconstruction was run.

The unavailable exact intermediate patch executable and fitting-time MRI/ANTs
provenance remain documented limitations. Numerical coordinate checks and
round-trip QA do not establish cortical registration accuracy. See the
[registration QA sensitivity protocol](REGISTRATION_QA_SENSITIVITY_PROTOCOL.md),
[validation record](REGISTRATION_QA_SENSITIVITY_VALIDATION.md), and
[runnable stage instructions](../scripts/README.md).

# Template/fsaverage diagnostic evaluation

5,124 jobs: 5,119 PASS, 5 FAIL. 5,119 diagnostic distances; **zero valid anatomical localization errors**.

Distance means template HEAD minus individual HEAD Euclidean separation, without anatomical registration. Matching axis names do not establish anatomical correspondence. `localization_error_mm` remains empty and `localization_error_valid` remains False.

01: full job-level diagnostic ledger; 02: audited reference coordinates; 03: pooled method/strategy summaries; 04: participant summaries; 05: Single Reference owner summaries; 06: equal-participant summaries; 07: one summary per unique test run/method/strategy across reference owners; 08: preserved failures; 09: SHA-256 input inventory; 10: anatomical-evaluation blockers; 11: additional sub-03 summary; 12: provenance and interpretation manifest.

Failed jobs retain missing distances and count in pass-rate denominators. Sub-03 remains included in all-seven summaries. Six references are repeated evaluations of the same run. Do not compare these diagnostic values as anatomical errors against baseline/no-warp results.

Next: review the diagnostic audit, resolve and independently validate the evaluation-only mapping of stimulation locations to fsaverage, then calculate anatomical errors in a separately labeled stage. No subject anatomy, digitization or ground-truth fitting is introduced into reconstruction.

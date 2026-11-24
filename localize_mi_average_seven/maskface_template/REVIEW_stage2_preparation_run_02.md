# Stage 2 preparation attempt: run 02

Run `outputs/02_stage2_preparation/run_02/` stopped at the staged-input hash check for sub-01. The staged file itself is byte-identical to the released source: both are 67,109,216 bytes and their raw SHA-256 is `0cc07eb45b3e12c2c497845791723cccf82f6163e586ae8e9a799cac1cc1019e`. The failure was in the script: `Run.use()` returns the resolved `Path`, but the code compared that path object directly with the SHA-256 string.

The code now reads the expected digest from the run's recorded input-hash map before comparing. This changes no input data. Keep run 02 and its partial staged copy unchanged; run the corrected preparation in `run_03`, then use the matching fresh construction and QA directories listed in `STAGE2_RUN.md`.

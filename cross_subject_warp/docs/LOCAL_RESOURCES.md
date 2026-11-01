# Reusing the existing local results

These folders were not present in GitHub. Their availability and validity have
not been checked. Do not assume a folder contains all subjects or pairs.

| Existing local path | Proposed new local copy | Purpose |
|---|---|---|
| `cross_subject/outputs/09_anatomical_registration/` | `cross_subject_warp/source/01_registration_resources/outputs/09_anatomical_registration/` | Affine matrices, warped MRIs, logs and QA |
| `cross_subject/outputs/19_syn_registration/` | `cross_subject_warp/source/01_registration_resources/outputs/19_syn_registration/` | SyN affine, forward/inverse fields, warped MRIs, logs and QA |
| `cross_subject/outputs/31_all_reference_registration/` | `cross_subject_warp/source/01_registration_resources/outputs/31_all_reference_registration/` | Registration schedule/status |
| Earlier coordinate QA outputs, especially stages 13–16 and 20–21 | Same original basenames inside the registration resource `outputs/` | Evidence supporting frame/direction checks |
| `cross_subject_no_warp/outputs/` canonical files | Import later into appropriate new output stages with explicit provenance | Selected parameters, final reconstruction records and sensitivities |

Do not copy new experiment results into historical folders or reset branches to
recover ignored local outputs. Switching Git branches normally leaves ignored
outputs on disk, but check `git status` and preserve your local work first.

## Optional manual copy on Ubuntu

From your **local project root**, after checking out the new branch and checking
that all three source folders exist, this copies the registration folders only.
It does not run any registration. It refuses to use a nonempty destination, so
existing imported data will not be overwritten:

```bash
python3 - <<'PY'
from pathlib import Path
import shutil

source = Path('cross_subject/outputs')
target = Path('cross_subject_warp/source/01_registration_resources/outputs')
names = ['09_anatomical_registration', '19_syn_registration',
         '31_all_reference_registration']
for name in names:
    if not (source / name).is_dir():
        raise SystemExit(f'Missing local source folder: {source / name}')
    if (target / name).exists():
        raise SystemExit(f'Destination already exists; inspect first: {target / name}')
for name in names:
    shutil.copytree(source / name, target / name)
    print(f'Copied {source / name} -> {target / name}')
PY
```

This is a convenience copy, **not a validation**. A later Script 01 must hash
the source/copy files and validate subject identities, MRI provenance, complete
SyN chains, point directions and QA. If copying is interrupted, inspect the
partial destination instead of treating it as complete. Check disk space first;
warped MRIs and deformation fields can be large. These local copies are ignored
by Git and will not be uploaded by committing the new experiment code.

For every `{ref}_to_{test}` pair, inventory the affine-only matrix separately
from the SyN affine component. SyN needs its own `0GenericAffine.mat`,
`1Warp.nii.gz`, `1InverseWarp.nii.gz`, warped images and quality report.
Fixed MRI is test; moving MRI is reference. Keep the validated historical
point-transform order and RAS/LPS conversion; do not infer point direction from
the image-resampling transform names.

The Localize-MI dataset may remain at its existing local location; record the
actual path and input hashes rather than duplicating the entire dataset. ECD
also requires the matching model-subject BEM solutions from the earlier
additional-method outputs.

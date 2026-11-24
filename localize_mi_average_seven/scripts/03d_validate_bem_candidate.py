"""Stage and validate a watershed candidate without modifying original anatomy."""
import argparse
import hashlib
import json
from pathlib import Path

import mne
import numpy as np
from mne.bem import _extract_volume_info
from execution import validate_mesh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('candidate', type=Path)
    args = parser.parse_args()
    candidate = args.candidate.resolve()
    exp = Path(__file__).resolve().parents[1]
    original = exp/'work/freesurfer/average_seven'
    subjects = candidate/'validation_subjects'
    subject = subjects/'average_seven'
    (subject/'bem').mkdir(parents=True, exist_ok=False)
    for name in ['mri', 'surf']:
        (subject/name).symlink_to(original/name, target_is_directory=True)
    info = _extract_volume_info(str(original/'mri/T1.mgz'))
    report = dict(candidate=str(candidate), anatomy_approved=False,
                  input_sha256={}, changes_from_original={})
    try:
        for name in ['brain', 'inner_skull', 'outer_skull', 'outer_skin']:
            path = candidate/f'average_seven_{name}_surface'
            report['input_sha256'][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            rr, tris, metadata = mne.read_surface(path, read_metadata=True)
            validate_mesh(rr, tris)
            old_rr, old_tris = mne.read_surface(original/'bem'/f'{name}.surf')
            if np.array_equal(tris, old_tris):
                report['changes_from_original'][name] = dict(
                    max_vertex_displacement_mm=float(np.linalg.norm(rr-old_rr, axis=1).max()))
            metadata.update(info)
            mne.write_surface(subject/'bem'/f'{name}.surf', rr, tris,
                              volume_info=metadata, overwrite=False)
        models = mne.make_bem_model('average_seven', ico=4,
                                    conductivity=(.3, .006, .3), subjects_dir=subjects)
        report['bem_geometry'] = 'PASS'
        solution = mne.make_bem_solution(models)
        mne.write_bem_solution(candidate/'candidate-bem-sol.fif', solution, overwrite=False)
        report['bem_solution'] = 'PASS'
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
    (candidate/'validation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

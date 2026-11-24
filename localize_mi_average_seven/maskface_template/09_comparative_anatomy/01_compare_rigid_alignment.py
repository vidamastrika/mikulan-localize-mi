"""Compare three rigid EEG fits after the approved Maskface BEM build."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '08_anatomical_review_bem'))
from review_common import FSAVERAGE, ROOT, VAR, Run, sha, write_json
from diagnose import OLD_FS

EXPERIMENT = VAR.parent
sys.path.insert(0, str(EXPERIMENT / 'scripts'))
from execution import module, validate_mesh

OUT = VAR / 'outputs/09_comparative_anatomy'


def run_comparison(args, run):
    import mne
    from mne.io.constants import FIFF

    bem = run.verify(args.bem_root, stage='08_BEM_AND_SOURCES')
    if bem.get('anatomical_approval', {}).get('decision') != 'APPROVED_FOR_BEM_CONSTRUCTION':
        raise ValueError('Expected the completed, anatomically approved Maskface BEM')
    config = json.loads(run.use(EXPERIMENT / 'config/experiment.json').read_text())
    params = config['coregistration']
    helper = module(run.use(ROOT / 'cross_subject_fsaverage/scripts/02_coregister_eeg.py'),
                    'maskface_three_arm_icp')
    run.use(EXPERIMENT / 'scripts/execution.py')
    run.use(EXPERIMENT / 'scripts/common.py')

    review = json.loads(run.use(args.fiducial_review).read_text())
    if (review.get('decision') != 'APPROVED' or not str(review.get('reviewer', '')).strip()
            or review.get('bem_manifest_sha256') != sha(args.bem_root / 'manifest.json')):
        raise ValueError('Expected a fiducial review bound to this approved BEM')
    pydeface_t1 = run.use(OLD_FS / 'mri/T1.mgz')
    arms = {
        'pydeface_average_seven': (OLD_FS / 'bem/outer_skin.surf', args.pydeface_fiducials),
        'maskface_average_seven': (Path(bem['scalp']), args.maskface_fiducials),
        'fsaverage': (FSAVERAGE / 'bem/outer_skin.surf', FSAVERAGE / 'bem/fsaverage-fiducials.fif'),
    }
    sensors = {}
    for subject in config['participants']:
        path = run.use(ROOT / config['dataset'] / 'derivatives/sourcemodelling' /
                       subject / 'fwd' / f'{subject}_fwd.fif')
        channels = mne.read_forward_solution(path, verbose=False)['info']['chs']
        points = np.array([channel['loc'][:3] for channel in channels])
        if (points.shape != (256, 3) or not np.isfinite(points).all()
                or any(ch['kind'] != FIFF.FIFFV_EEG_CH
                       or ch['coord_frame'] != FIFF.FIFFV_COORD_HEAD for ch in channels)):
            raise ValueError(f'Expected 256 finite original EEG coordinates in HEAD metres: {subject}')
        sensors[subject] = points

    rows = []
    for arm, (surface_path, fid_path) in arms.items():
        surface_path, fid_path = run.use(surface_path), run.use(fid_path)
        if arm != 'fsaverage':
            expected = {'scalp_sha256': sha(surface_path), 'fiducials_sha256': sha(fid_path),
                        't1_sha256': (sha(pydeface_t1) if arm == 'pydeface_average_seven'
                                      else bem['anatomical_approval']['input_sha256'][
                                          str((Path(bem['subjects_dir']) / bem['subject'] /
                                               'mri/T1.mgz').resolve())])}
            if review.get('arms', {}).get(arm) != expected:
                raise ValueError(f'Fiducial review does not match the exact inputs for {arm}')
        scalp, faces = mne.read_surface(surface_path)
        scalp = scalp / 1000
        validate_mesh(scalp, faces)
        if not .05 < np.percentile(np.linalg.norm(scalp, axis=1), 95) < .25:
            raise ValueError(f'Unexpected scalp coordinate scale: {arm}')
        points, frame = mne.io.read_fiducials(fid_path)
        pos = {int(p['ident']): p['r'] for p in points}
        xyz = np.array(list(pos.values()))
        if (frame != FIFF.FIFFV_COORD_MRI or len(points) != 3 or set(pos) != {1, 2, 3}
                or not np.isfinite(xyz).all() or np.max(np.abs(xyz)) >= .25
                or np.linalg.matrix_rank(xyz[1:] - xyz[0]) != 2):
            raise ValueError(f'Expected three valid MRI surface RAS fiducials in metres: {arm}')
        initial = np.linalg.inv(mne.transforms.get_ras_to_neuromag_trans(pos[2], pos[1], pos[3]))
        transforms = run.out / 'transforms' / arm
        transforms.mkdir(parents=True)
        for subject, xyz in sensors.items():
            matrix, distances, iterations = helper.icp(
                xyz, scalp, initial, params['iterations'], params['trim_fraction'])
            rotation = matrix[:3, :3]
            if (not np.isfinite(matrix).all() or not np.isfinite(distances).all()
                    or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7)
                    or not np.isclose(np.linalg.det(rotation), 1, atol=1e-7)
                    or not np.allclose(matrix[3], [0, 0, 0, 1])):
                raise ValueError(f'Invalid rigid result: {subject}, {arm}')
            median, p95 = np.percentile(distances * 1000, [50, 95])
            passed = median <= params['median_limit_mm'] and p95 <= params['p95_limit_mm']
            path = transforms / f'{subject}-trans.fif'
            mne.write_trans(path, mne.transforms.Transform('head', 'mri', matrix))
            rows.append(dict(subject=subject, arm=arm, channels=len(xyz), iterations=iterations,
                             median_mm=median, p95_mm=p95,
                             status='PASS_NUMERICAL_REVIEW_REQUIRED' if passed else 'FAIL_ALIGNMENT',
                             transform=str(path.relative_to(run.out))))
    results = pd.DataFrame(rows)
    results.to_csv(run.out / '01_alignment.csv', index=False)
    paired = results.pivot(index='subject', columns='arm', values=['median_mm', 'p95_mm', 'status'])
    paired.columns = [f'{metric}_{arm}' for metric, arm in paired.columns]
    paired.to_csv(run.out / '02_three_arm_comparison.csv')
    write_json(run.out / '03_visual_review_pending.json',
               {arm: {row.subject: 'PENDING' if row.status.startswith('PASS') else 'FAIL_ALIGNMENT'
                      for row in results[results.arm == arm].itertuples()} for arm in arms})
    run.doc.update(coregistration=params, mne_version=mne.__version__, channels=256,
                   score='all original sensors to nearest scalp vertex', visual_review='PENDING')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bem-root', 'pydeface-fiducials', 'maskface-fiducials', 'fiducial-review', 'output-root'):
        parser.add_argument(f'--{name}', required=True, type=Path)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        parser.error('--execute is required after approval')
    if OUT.resolve() not in args.output_root.resolve().parents:
        parser.error(f'Choose a fresh output under {OUT}')
    with Run(args.output_root, '09_THREE_ARM_RIGID_ALIGNMENT') as run:
        run.use(Path(__file__).resolve().parents[1] / '08_anatomical_review_bem/review_common.py')
        run_comparison(args, run)


if __name__ == '__main__':
    main()

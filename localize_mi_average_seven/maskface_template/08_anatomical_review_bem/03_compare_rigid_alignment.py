"""Run the frozen rigid EEG alignment for approved maskface and fsaverage."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

from review_common import FSAVERAGE, ROOT, VAR, new_run, sha, write_json

EXPERIMENT = VAR.parent
sys.path.insert(0, str(EXPERIMENT / 'scripts'))
import execution


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('bem-root', 'fiducials', 'fiducial-review', 'output-root'):
        parser.add_argument(f'--{option}', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        parser.error('--execute is required to fit both rigid alignments')

    with new_run(args.output_root, '08_RIGID_EEG_ALIGNMENT') as run:
        import mne
        from mne.io.constants import FIFF

        paths = run.verify(args.bem_root, stage='08_BEM_AND_SOURCES')
        fid = run.use(args.fiducials)
        review = json.loads(run.use(args.fiducial_review).read_text())
        if (review.get('decision') != 'APPROVED'
                or review.get('fiducials_sha256') != sha(fid)
                or review.get('bem_manifest_sha256') != sha(args.bem_root / 'manifest.json')
                or not str(review.get('reviewer', '')).strip()):
            raise ValueError('Reviewed MRI fiducials must be matched to the completed BEM run')
        fs_fid = run.use(FSAVERAGE / 'bem/fsaverage-fiducials.fif')
        for file in (fid, fs_fid):
            points, frame = mne.io.read_fiducials(file)
            positions = np.array([p['r'] for p in points])
            if (frame != FIFF.FIFFV_COORD_MRI or len(points) != 3
                    or {int(p['ident']) for p in points} != {1, 2, 3}
                    or not np.isfinite(positions).all() or np.max(np.abs(positions)) >= .25
                    or np.linalg.matrix_rank(positions[1:] - positions[0]) != 2):
                raise ValueError(f'Expected three distinct MRI RAS fiducials in metres: {file}')

        cfg = json.loads(run.use(EXPERIMENT / 'config/experiment.json').read_text())
        run.use(Path(execution.__file__))
        run.use(EXPERIMENT / 'scripts/common.py')
        run.use(ROOT / 'cross_subject_fsaverage/scripts/02_coregister_eeg.py')
        for participant in cfg['participants']:
            path = run.use(ROOT / cfg['dataset'] / 'derivatives/sourcemodelling' / participant /
                           'fwd' / f'{participant}_fwd.fif')
            channels = mne.read_forward_solution(path, verbose=False)['info']['chs']
            if len(channels) != 256 or any(ch['kind'] != FIFF.FIFFV_EEG_CH for ch in channels):
                raise ValueError(f'Expected all 256 original EEG channels: {participant}')

        anatomy = {key: paths[key] for key in ('bem', 'src', 'scalp')}
        anatomy.update(arm='average_seven', fiducials=str(fid))
        anatomy['input_sha256'] = {path: sha(Path(path)) for key, path in anatomy.items() if key != 'arm'}
        anatomy_path = run.out / 'maskface_anatomy.json'
        write_json(anatomy_path, anatomy)
        results = []
        for arm, folder in [('average_seven', 'maskface_baseline'), ('fsaverage', 'fsaverage')]:
            execution.coregister(SimpleNamespace(
                arm=arm, fsaverage_root=FSAVERAGE, anatomy_manifest=anatomy_path,
                output_root=run.out / folder), cfg)
            manifest = run.out / folder / '02_coregistration_manifest.json'
            for path, checksum in json.loads(manifest.read_text())['input_sha256'].items():
                if run.out not in Path(path).parents:
                    run.use(path, checksum)
            results.append(pd.read_csv(run.out / folder / '01_alignment.csv')[
                ['test_subject', 'status', 'median_mm', 'p95_mm']])
        results[0].merge(results[1], on='test_subject', suffixes=('_maskface', '_fsaverage'),
                         validate='one_to_one').to_csv(run.out / '01_paired_alignment.csv', index=False)
        run.doc.update(coregistration=cfg['coregistration'], sensor_count=256, visual_review='PENDING')


if __name__ == '__main__':
    main()

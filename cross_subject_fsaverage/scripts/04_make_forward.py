#!/usr/bin/env python3
# Script 04 — Build fsaverage forward models for test EEG.
# Purpose: Replace anatomical models while retaining the frozen no-warp schedule,
#   reference identities, configurations, and later reference-based channel policy.
# Inputs: Script 01 plan/source_configurations/manifest; Script 02 audit, manifest,
#   and head-to-fsaverage FIF transforms; native forwards for test sensor geometry;
#   fsaverage BEM solution and source space already inventoried by Script 01.
# Outputs: NEW output directory with subject status, full 5,124-job availability
#   ledger, fsaverage forwards, frozen configuration copies, and hash/runtime manifest.
# Parameters: --phase audit (default) or build; --n-jobs 1; explicit MNE forward
#   defaults mindist=0 mm, meg=False, eeg=True; no scaling or registration fitting.
# Dependencies: Scripts 01 and 02; numpy, pandas, MNE. Script 03 MRI registration
#   is NOT required. sub-03 remains blocked even if a replacement QA table says PASS.
# Commands (repository root; choose a new output directory for each invocation):
#   .venv/bin/python cross_subject_fsaverage/scripts/04_make_forward.py \
#       --phase audit --output-root cross_subject_fsaverage/outputs/04_make_forward_audit
#   .venv/bin/python cross_subject_fsaverage/scripts/04_make_forward.py \
#       --phase build --output-root cross_subject_fsaverage/outputs/04_make_forward
# No inverse reconstruction is performed by either command.
"""Build test-EEG/fsaverage forwards without changing frozen reconstruction choices."""
from __future__ import annotations
import argparse
import importlib.util
import json
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / 'cross_subject_fsaverage'


def shared():
    """Reuse Script 01's template filenames, methods, and digest implementation."""
    path = EXP / 'scripts/01_prepare_experiment.py'
    spec = importlib.util.spec_from_file_location('fsaverage_preparation_shared', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=['audit', 'build'], default='audit')
    p.add_argument('--preparation-root', type=Path, default=EXP / 'outputs/01_prepare_experiment')
    p.add_argument('--coregistration-root', type=Path, default=EXP / 'outputs/02_coregister_eeg')
    p.add_argument('--fsaverage-root', type=Path, default=Path.home() / 'mne_data/MNE-fsaverage-data/fsaverage')
    p.add_argument('--output-root', type=Path, default=EXP / 'outputs/04_make_forward')
    p.add_argument('--n-jobs', type=int, default=1)
    args = p.parse_args(argv)
    if args.n_jobs < 1:
        p.error('--n-jobs must be positive')
    prep, coreg, fsroot, out = [x.expanduser().resolve() for x in
        (args.preparation_root, args.coregistration_root, args.fsaverage_root, args.output_root)]
    if out.exists():
        raise FileExistsError(f'Refusing existing output directory: {out}')
    # Prevent new files being placed inside frozen inputs.
    for protected in (prep, coreg, fsroot, EXP / 'source', EXP / 'scripts'):
        if out == protected or protected in out.parents:
            raise ValueError(f'Output must be separate from inputs: {protected}')
    old = shared()
    hashes = {}

    def require(path, expected=None):
        if not path.is_file():
            raise FileNotFoundError(path)
        value = old.digest(path)
        if expected is not None and value != expected:
            raise ValueError(f'Input hash differs: {path}')
        hashes[str(path)] = value
        return path

    plan_path = require(prep / '02_fsaverage_reconstruction_plan.csv')
    prep_path = require(prep / '05_experiment_manifest.json')
    cm_path = require(coreg / '03_coregistration_manifest.json')
    cm = json.loads(cm_path.read_text())
    require(plan_path, cm['script01_plan_sha256'])
    require(prep_path, cm['script01_manifest_sha256'])
    manifest = json.loads(prep_path.read_text())
    plan = pd.read_csv(plan_path, dtype=str)
    subjects = [f'sub-{i:02d}' for i in range(1, 8)]
    key = ['strategy', 'method', 'reference_subject', 'test_subject', 'run']
    if (len(plan) != 5124 or plan[key].isna().any().any() or plan.duplicated(key).any()
            or set(plan.test_subject) != set(subjects) or not plan.model_subject.eq('fsaverage').all()
            or set(plan.method) != set(old.METHODS)
            or plan.groupby('strategy').size().to_dict() != {'LOSO': 732, 'Single-reference': 4392}):
        raise ValueError('Frozen Script 01 schedule differs from expected design')
    # Match each copied configuration file to the original inventoried hash.
    inventory = pd.read_csv(require(prep / '01_input_inventory.csv'))
    configs = []
    for name in old.SNAPSHOT:
        item = inventory.loc[inventory.name.eq(name)]
        if len(item) != 1:
            raise ValueError(f'No unique frozen inventory entry: {name}')
        row = item.iloc[0]
        configs.append(require(prep / 'source_configurations' / (name + Path(row.path).suffix), row.sha256))
    qa = pd.read_csv(require(coreg / '01_subject_coregistration_audit.csv')).set_index('subject')
    saved_electrodes = pd.read_csv(require(coreg / '02_electrode_distances.csv'))
    if set(qa.index) != set(subjects) or not qa.index.is_unique:
        raise ValueError('Coregistration audit must contain each subject exactly once')
    resources = {r['name']: r for r in manifest['fsaverage_resources']}
    anatomy = {name: require(fsroot / old.FS_FILES[name], resources[name]['sha256'])
               for name in ['source_space', 'bem_solution']}
    require(fsroot / old.FS_FILES['head_surface'], cm['fsaverage_scalp_sha256'])
    import mne
    from mne.io.constants import FIFF
    rows, inputs = [], {}
    for subject in subjects:
        q = qa.loc[subject]
        row = dict(test_subject=subject, status='BLOCKED_COREGISTRATION_QA', reason='',
                   forward_path='', runtime_seconds=0.0, n_jobs_scheduled=int(plan.test_subject.eq(subject).sum()),
                   coregistration_status=q.status, native_forward_sha256=q.native_forward_sha256,
                   transform_sha256='', forward_sha256='')
        if subject == 'sub-03':
            row['reason'] = 'Persistent sub-03 block: unresolved EEG coregistration QA failure'
        elif (q.status != 'PASS_NUMERICAL_REVIEW_REQUIRED'
              or not np.isfinite([q.median_scalp_distance_mm, q.p95_scalp_distance_mm]).all()
              or q.median_scalp_distance_mm > 10 or q.p95_scalp_distance_mm > 20):
            row['reason'] = 'Existing numerical coregistration gate not satisfied'
        else:
            native = require(Path(q.native_forward_path), q.native_forward_sha256)
            trans_path = require(coreg / 'transforms' / f'{subject}_head_to_fsaverage-trans.fif')
            trans = mne.read_trans(trans_path)
            matrix = np.asarray(trans['trans'])
            if ((int(trans['from']), int(trans['to'])) != (int(FIFF.FIFFV_COORD_HEAD), int(FIFF.FIFFV_COORD_MRI))
                or matrix.shape != (4, 4) or not np.isfinite(matrix).all()
                or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6, rtol=0)
                or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-6, rtol=0)
                or abs(np.linalg.det(matrix[:3, :3])-1) > 1e-6):
                raise ValueError(f'{subject}: invalid rigid HEAD->fsaverage MRI transform')
            fwd = mne.read_forward_solution(native, verbose=False)
            channels = fwd['info']['chs']
            names = [c['ch_name'] for c in channels]
            positions = np.array([c['loc'][:3] for c in channels])
            if (len(names) != 256 or len(set(names)) != 256 or not np.isfinite(positions).all()
                or any(int(c['coord_frame']) != int(FIFF.FIFFV_COORD_HEAD) for c in channels)):
                raise ValueError(f'{subject}: invalid native HEAD sensor geometry')
            saved = saved_electrodes.loc[saved_electrodes.subject.eq(subject)].set_index('channel')
            if len(saved) != 256 or not saved.index.is_unique or set(saved.index) != set(names):
                raise ValueError(f'{subject}: saved electrode audit coverage differs')
            saved = saved.loc[names]
            original = saved[[f'native_head_{a}_m' for a in 'xyz']].to_numpy(float)
            mapped = saved[[f'fsaverage_mri_{a}_m' for a in 'xyz']].to_numpy(float)
            if (not np.allclose(original, positions, atol=1e-10, rtol=0)
                or not np.allclose(mapped, positions @ matrix[:3, :3].T + matrix[:3, 3], atol=1e-8, rtol=0)):
                raise ValueError(f'{subject}: transform/sensors differ from Script 02 electrode audit')
            inputs[subject] = (names, positions, trans)
            row.update(status='READY_NUMERICAL_ONLY', reason='Existing candidate alignment; anatomical accuracy unverified',
                       transform_sha256=hashes[str(trans_path)])
        rows.append(row)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'source_configurations').mkdir()
    for path in configs:
        shutil.copy2(path, out / 'source_configurations' / path.name)
    shutil.copy2(plan_path, out / '02_frozen_reconstruction_plan.csv')
    if args.phase == 'build':
        (out / 'forwards').mkdir()

    def checkpoint():
        pd.DataFrame(rows).to_csv(out / '01_subject_forward_status.csv', index=False)
        statuses = {r['test_subject']: r for r in rows}
        ledger = plan.copy()
        ledger['forward_status'] = ledger.test_subject.map(lambda s: statuses[s]['status'])
        ledger['forward_path'] = ledger.test_subject.map(lambda s: statuses[s]['forward_path'])
        ledger['forward_reason'] = ledger.test_subject.map(lambda s: statuses[s]['reason'])
        ledger.to_csv(out / '03_job_forward_availability.csv', index=False)

    checkpoint()
    for row in rows:
        if args.phase != 'build' or row['test_subject'] not in inputs:
            continue
        subject = row['test_subject']
        names, positions, trans = inputs[subject]
        started = time.perf_counter()
        try:
            # HEAD coordinates remain unchanged: the transform is applied by MNE.
            # Sampling rate is irrelevant to a static forward and not used by inverse loading.
            info = mne.create_info(names, sfreq=1.0, ch_types='eeg')
            info.set_montage(mne.channels.make_dig_montage(ch_pos=dict(zip(names, positions)), coord_frame='head'))
            fwd = mne.make_forward_solution(info, trans=trans, src=anatomy['source_space'],
                bem=anatomy['bem_solution'], meg=False, eeg=True, mindist=0.0, n_jobs=args.n_jobs)
            if (fwd['info']['ch_names'] != names or fwd['sol']['data'].shape[0] != 256
                or not np.isfinite(fwd['sol']['data']).all() or fwd['nsource'] <= 0):
                raise ValueError('Invalid forward channel order/leadfield/source count')
            actual = np.array([c['loc'][:3] for c in fwd['info']['chs']])
            if not np.allclose(actual, positions, atol=1e-12, rtol=0):
                raise ValueError('Forward sensor coordinates changed')
            path = out / 'forwards' / f'{subject}_fsaverage-fwd.fif'
            mne.write_forward_solution(path, fwd, overwrite=False)
            row.update(status='PASS_FORWARD_NUMERICAL_ONLY', forward_path=str(path),
                       forward_sha256=old.digest(path), n_sources=int(fwd['nsource']))
        except Exception as exc:
            row.update(status='FAIL_FORWARD', reason=f'{type(exc).__name__}: {exc}')
        row['runtime_seconds'] = time.perf_counter() - started
        checkpoint()
    git = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True)
    report = dict(script='04_make_forward.py', created_utc=datetime.now(timezone.utc).isoformat(),
        phase=args.phase, git_commit=git.stdout.strip() if git.returncode == 0 else None,
        script_sha256=old.digest(Path(__file__)), shared_preparation_sha256=old.digest(EXP / 'scripts/01_prepare_experiment.py'),
        input_sha256=hashes, method_parameters_changed=False, schedule_changed=False,
        forward_parameters=dict(mindist_mm=0.0, meg=False, eeg=True, n_jobs=args.n_jobs,
                                source_space=str(anatomy['source_space']), bem=str(anatomy['bem_solution'])),
        software_versions=dict(mne=mne.__version__, numpy=np.__version__, pandas=pd.__version__),
        registration_fitting_executed=False, inverse_reconstruction_executed=False,
        subject_results=rows, notes=[
            'Reference identities remain parameter/channel-selection identities; anatomy is fsaverage.',
            'Native test sensor geometry plus existing candidate rigid transform; no electrode relocation or scaling.',
            'Compared with no-warp cross fitting, sensor geometry also changes from reference to test geometry.',
            'No independent anatomical accuracy claimed by numerical PASS.',
            'Future unregistered HEAD distances must be labeled diagnostic.',
            'Original baseline, no-warp, affine and SyN inputs remain unchanged.'])
    (out / '04_forward_manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(pd.DataFrame(rows)[['test_subject', 'status', 'n_jobs_scheduled']].to_string(index=False))
    return 2 if any(r['status'] == 'FAIL_FORWARD' for r in rows) else 0


if __name__ == '__main__':
    raise SystemExit(main())

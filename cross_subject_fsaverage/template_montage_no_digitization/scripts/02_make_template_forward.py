#!/usr/bin/env python3
# RESEARCH OBJECTIVE
# Build one common fsaverage/template EEG forward for the exploratory Localize-MI
# no-digitization variant. Both anatomy and sensor geometry differ from no-warp.
# ASSUMPTIONS / AUTHORIZATION
# User authorized proceeding after Script 01's interpolation review. eN->EN
# remains ASSUMED_UNVERIFIED. Keep fixed template HEAD coordinates and predefined
# HEAD->fsaverage MRI transform. Use standard MNE 1.12.1 BEM interpolation,
# including its unmodified plane extrapolation. No anatomical PASS is asserted.
# INPUTS
# Script 01 assumed-v1 manifest, full 5,124-job schedule, template geometry and
# transform provenance; completed actual-BEM audit; hash-pinned fsaverage source
# space/BEM and all frozen configuration snapshots. No participant forward,
# electrode TSV, digitization, participant MRI, EEG or stimulation data is read.
# OUTPUTS (new output directory only)
# 01_forward_manifest.json: policy, hashes, software, status and numerical checks.
# 02_frozen_reconstruction_schedule.csv: byte-exact original 5,124-row schedule.
# 03_job_forward_availability.csv: same jobs linked to one common forward.
# 04_template_sensor_coordinates.csv: byte-exact Script 01 template geometry.
# template_head_to_fsaverage-trans.fif; forwards/template_fsaverage-fwd.fif
# (build only); source_configurations/ snapshots; script_snapshot.py.txt; README.
# PARAMETERS / VALIDATION
# --phase audit (default) checks provenance/design without computing a forward.
# --phase build computes one 256-channel free-orientation EEG forward, n-jobs=1
# default; meg=False, eeg=True, mindist=0 mm (same previous fsaverage setting).
# 7 subjects, 61 runs, 12 methods: 732 LOSO + 4,392 Single Reference = 5,124
# inverse evaluations, not 5,124 forwards. Keep every method/reference identity.
# Verify finite leadfield, channel order, HEAD/MRI transform, untouched sensor
# positions, monopolar EEG reference locations and reloaded forward integrity.
# No clipping, manual scalp projection, registration, preprocessing or inverse.
# Existing numerical/anatomical QA thresholds and prior outputs remain intact.
# DEPENDENCIES: Python >=3.10, NumPy, pandas, MNE 1.12.1; local frozen assets.
# COMPLETE COMMANDS (repository root; run manually, fresh output directory):
# .venv/bin/python cross_subject_fsaverage/template_montage_no_digitization/scripts/02_make_template_forward.py \
#   --phase audit --output-root cross_subject_fsaverage/template_montage_no_digitization/outputs/02_template_forward_audit_assumed_v1
# .venv/bin/python cross_subject_fsaverage/template_montage_no_digitization/scripts/02_make_template_forward.py \
#   --phase build --output-root cross_subject_fsaverage/template_montage_no_digitization/outputs/02_template_forward_assumed_v1
# Exit 0 success (exploratory numerical status only); 1 failure. No resume/overwrite.
"""Audit or build the common exploratory template/fsaverage EEG forward."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
import pandas as pd

VARIANT = Path(__file__).resolve().parents[1]
KEY = ['strategy', 'method', 'reference_subject', 'test_subject', 'run']
METHODS = {'MNE','dSPM','sLORETA','eLORETA','Continuous-ECD','RAP-MUSIC',
           'LCMV','MxNE','irMxNE','Gamma-MAP','Champagne','SESAME'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            value.update(chunk)
    return value.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['audit','build'], default='audit')
    parser.add_argument('--preparation-root', type=Path, default=VARIANT/'outputs/01_prepare_experiment_assumed_v1')
    parser.add_argument('--bem-audit-root', type=Path, default=VARIANT/'outputs/01_bem_sensor_audit_assumed_v1')
    parser.add_argument('--output-root', type=Path, default=VARIANT/'outputs/02_template_forward_audit_assumed_v1')
    parser.add_argument('--n-jobs', type=int, default=1)
    args = parser.parse_args(argv)
    require(args.n_jobs > 0, '--n-jobs must be positive')
    prep, audit_root, out = [p.expanduser().resolve() for p in
                              (args.preparation_root,args.bem_audit_root,args.output_root)]
    allowed = (VARIANT/'outputs').resolve()
    require(allowed in out.parents and not out.exists(), 'Use a new directory within this variant/outputs')
    for protected in (prep,audit_root):
        require(VARIANT in protected.parents, 'Input reports must belong to this variant')
        require(protected not in out.parents and out not in protected.parents, 'Output overlaps frozen inputs')
    require(importlib.metadata.version('mne') == '1.12.1', 'Interpolation was reviewed for MNE 1.12.1 only')
    hashes = {}

    def verify(path, expected=None):
        path = Path(path).resolve()
        value = digest(path)
        require(expected is None or value == expected, f'Frozen input changed: {path}')
        hashes[str(path)] = value
        return path

    manifest_path = verify(prep/'01_experiment_manifest.json')
    manifest = json.loads(manifest_path.read_text())
    review = json.loads(verify(audit_root/'04_bem_interpolation_audit.json').read_text())
    require(review['status'] == 'BEM_INTERPOLATION_AUDIT_REVIEW_REQUIRED', 'Completed BEM interpolation audit required')
    require(review['mapping_policy'] == manifest['mapping_policy'] == 'documented-assumption', 'Expected explicit exploratory assumption policy')
    require(review['mapping_status'] == ['ASSUMED_UNVERIFIED'], 'Mapping status changed; review policy')
    require(review['maximum_independent_triangle_plane_projection_error_mm'] <= .001, 'Interpolation consistency check failed')
    for path, expected in review['input_sha256'].items():
        verify(path,expected)
    require(str(manifest_path) in review['input_sha256'], 'BEM audit does not reference this preparation')
    frozen = []
    for item in manifest['source_inventory']:
        path = verify(item['copy_path'],item['sha256'])
        require(VARIANT in path.parents, 'Snapshot outside this variant')
        frozen.append((item,path))
    resources = {r['role']:r for r in manifest['fsaverage_resources']}
    anatomy = {role:verify(resources[role]['path'],resources[role]['sha256'])
               for role in ('source_space','bem_solution')}
    plan_path = verify(prep/'02_planned_reconstruction_schedule.csv')
    plan = pd.read_csv(plan_path,keep_default_na=False,dtype=str)
    require(len(plan)==5124 and not plan.duplicated(KEY).any(), 'Invalid job count/identities')
    require(not plan[KEY].eq('').any().any(), 'Empty job identity')
    require(plan.groupby('strategy').size().to_dict()=={'LOSO':732,'Single-reference':4392}, 'Strategy counts differ')
    require(set(plan.test_subject)=={f'sub-{i:02d}' for i in range(1,8)} and set(plan.method)==METHODS, 'Subjects/methods differ')
    require(len(plan[['test_subject','run']].drop_duplicates())==61, 'Expected 61 subject/run pairs')
    require(plan.groupby(['test_subject','run','strategy']).size().unstack().eq(
        pd.Series({'LOSO':12,'Single-reference':72})).all().all(), 'Incomplete method coverage per run')
    single = plan.loc[plan.strategy.eq('Single-reference')]
    require(not single.reference_subject.eq(single.test_subject).any()
            and single.groupby(['test_subject','run','method']).reference_subject.nunique().eq(6).all(),
            'Single Reference exclusion/coverage changed')
    require(plan.model_subject.eq('fsaverage').all() and plan.sensor_geometry.eq('template_only').all()
            and plan.mapping_status.eq('ASSUMED_UNVERIFIED').all(), 'Unexpected anatomy/sensor/mapping policy')
    coords_path = verify(prep/'06_sensor_scalp_geometry_qa.csv')
    coords = pd.read_csv(coords_path).set_index('dataset_channel')
    names = [f'e{i}' for i in range(1,257)]
    require(coords.index.is_unique and set(coords.index)==set(names), 'Expected bijective e1..e256 geometry')
    coords = coords.loc[names]
    positions = coords[[f'template_head_{a}_m' for a in 'xyz']].to_numpy(float)
    mapped = coords[[f'fsaverage_surface_ras_{a}_m' for a in 'xyz']].to_numpy(float)
    provenance = json.loads(verify(prep/'07_coordinate_transform_provenance.json').read_text())
    matrix = np.asarray(provenance['head_to_fsaverage_mri'],float)
    require(matrix.shape==(4,4) and np.isfinite(matrix).all() and np.isfinite(positions).all(), 'Nonfinite geometry/transform')
    require(np.allclose(matrix[3],[0,0,0,1],rtol=0,atol=1e-10)
            and np.allclose(matrix[:3,:3].T@matrix[:3,:3],np.eye(3),rtol=0,atol=1e-5)
            and abs(np.linalg.det(matrix[:3,:3])-1)<1e-5, 'Invalid rigid template transform')
    require(np.allclose(positions@matrix[:3,:3].T+matrix[:3,3],mapped,rtol=0,atol=1e-10), 'Coordinate frames disagree')
    readiness = json.loads(verify(prep/'08_readiness_report.json').read_text())
    require(readiness['status']=='EXPLORATORY_NUMERICAL_REVIEW_REQUIRED' and not readiness['blockers'], 'Preparation numerical gate failed')
    import mne
    from mne.io.constants import FIFF
    info = mne.create_info(names,sfreq=1.0,ch_types='eeg')
    info.set_montage(mne.channels.make_dig_montage(ch_pos=dict(zip(names,positions)),coord_frame='head'))

    def check_sensors(candidate):
        require(candidate['ch_names']==names, 'Channel order changed')
        actual = np.array([c['loc'][:3] for c in candidate['chs']])
        require(np.allclose(actual,positions,rtol=0,atol=1e-7), 'Stored sensor positions changed')
        require(all(int(c['coord_frame'])==int(FIFF.FIFFV_COORD_HEAD) for c in candidate['chs']), 'Sensors are not HEAD coordinates')
        for c in candidate['chs']:
            reference = np.asarray(c['loc'][3:6])
            require(np.isnan(reference).all() or np.array_equal(reference,np.zeros(3)), 'Separate reference electrode would invalidate monopolar interpolation audit')

    check_sensors(info)
    trans = mne.transforms.Transform('head','mri',matrix)
    out.mkdir(parents=True,exist_ok=False)
    shutil.copy2(Path(__file__),out/'script_snapshot.py.txt')
    shutil.copy2(plan_path,out/'02_frozen_reconstruction_schedule.csv')
    shutil.copy2(coords_path,out/'04_template_sensor_coordinates.csv')
    # Preserve relative snapshot paths to avoid basename collisions.
    for item,path in frozen:
        target = out/'source_configurations'/path.relative_to(VARIANT)
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target)
        require(digest(target)==item['sha256'], 'Configuration snapshot copy changed')
    mne.write_trans(out/'template_head_to_fsaverage-trans.fif',trans,overwrite=False,verbose=False)
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(),phase=args.phase,
        status='READY_EXPLORATORY_FORWARD_BUILD',script_sha256=digest(Path(__file__)),input_sha256=hashes,
        policy=dict(mapping='ASSUMED_UNVERIFIED',coordinates='fixed original template HEAD coordinates',
            transform='predefined fsaverage HEAD to MRI; no individual fit',
            interpolation='standard MNE 1.12.1 unconstrained triangle-plane weights; no clipping',
            user_authorization='continue after actual-BEM interpolation review',
            individual_geometry_used=False,manual_projection=False,QA_thresholds_changed=False,
            anatomical_validity_established=False,preprocessing_executed=False,inverse_executed=False),
        counts=dict(shared_forwards=1,channels=256,subjects=7,runs=61,methods=12,loso=732,single_reference=4392,total=5124),
        forward_parameters=dict(meg=False,eeg=True,mindist_mm=0.0,n_jobs=args.n_jobs),
        software_versions={p:importlib.metadata.version(p) for p in ['mne','numpy','scipy','pandas']},
        interpolation_audit_summary={k:review[k] for k in ['displacement_mm','extrapolated_electrode_count','maximum_effective_point_surface_residual_mm']},
        forward_path='',forward_sha256='',runtime_seconds=0.0,
        limitations=['Template channel correspondence remains assumed, not independently verified.',
            'Original template/scalp mismatch remains; standard BEM interpolation is not anatomical validation.',
            'Both anatomy and sensor geometry differ from no-warp.',
            'One common full forward; later subsets and bad-channel handling must use frozen template policies.',
            'No localization error has been calculated by this stage.'])

    def checkpoint():
        ledger = plan.copy()
        ledger['forward_status'] = report['status']
        ledger['forward_path'] = report['forward_path']
        ledger['forward_sha256'] = report['forward_sha256']
        ledger.to_csv(out/'03_job_forward_availability.csv',index=False)
        write_json(out/'01_forward_manifest.json',report)

    (out/'README.md').write_text('Exploratory common fsaverage/template forward\n\n'
        '01_forward_manifest.json: status, numerical checks, provenance and limitations.\n'
        '02_frozen_reconstruction_schedule.csv: unchanged 5,124-job plan.\n'
        '03_job_forward_availability.csv: per-job common-forward availability.\n'
        '04_template_sensor_coordinates.csv: unchanged Script 01 coordinates.\n'
        'source_configurations/: copied frozen configuration/policy snapshots.\n'
        'template_head_to_fsaverage-trans.fif: fixed template-to-fsaverage transform.\n'
        'forwards/template_fsaverage-fwd.fif: common forward (build phase only).\n'
        'script_snapshot.py.txt: exact implementation used.\n\n'
        'Numerical success is exploratory; physical channel mapping and anatomical accuracy remain unverified.\n')
    checkpoint()
    if args.phase=='build':
        started = time.perf_counter()
        try:
            (out/'forwards').mkdir()
            forward = mne.make_forward_solution(info,trans=trans,src=anatomy['source_space'],
                bem=anatomy['bem_solution'],meg=False,eeg=True,mindist=0.0,n_jobs=args.n_jobs)
            check_sensors(forward['info'])
            data = forward['sol']['data']
            require(forward['sol']['row_names']==names and data.shape==(256,3*forward['nsource'])
                    and forward['nsource']>0 and np.isfinite(data).all(), 'Invalid free-orientation EEG leadfield')
            path = out/'forwards/template_fsaverage-fwd.fif'
            mne.write_forward_solution(path,forward,overwrite=False,verbose=False)
            reloaded = mne.read_forward_solution(path,verbose=False)
            check_sensors(reloaded['info'])
            require(np.isfinite(reloaded['sol']['data']).all() and np.allclose(reloaded['sol']['data'],data,rtol=1e-6,atol=1e-12), 'Forward serialization check failed')
            for path_string,expected in hashes.items():
                require(digest(path_string)==expected,f'Input changed during build: {path_string}')
            report.update(status='PASS_EXPLORATORY_FORWARD_NUMERICAL_ONLY',forward_path=str(path),
                forward_sha256=digest(path),n_sources=int(forward['nsource']),
                leadfield_shape=list(data.shape),leadfield_rank_not_evaluated=True)
        except Exception:
            report.update(status='FAIL_FORWARD',error=traceback.format_exc())
            (out/'failure_traceback.txt').write_text(report['error'])
        report['runtime_seconds']=time.perf_counter()-started
        checkpoint()
    print(f'{report["status"]}: {out}')
    return 1 if report['status']=='FAIL_FORWARD' else 0


if __name__=='__main__':
    raise SystemExit(main())

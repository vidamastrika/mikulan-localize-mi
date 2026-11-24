"""Safe scaffold utilities: no template, BEM or inverse computation on import."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / 'localize_mi_average_seven'
KEY = ['strategy', 'method', 'reference_subject', 'test_subject', 'run']
FINGERPRINTS = ['configuration_key', 'actual_channels_json', 'preprocessing_sha256',
                'electrode_geometry_sha256', 'coregistration_protocol_sha256',
                'peak_selection_policy', 'reference_policy_sha256']


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def config(path):
    return json.loads(path.read_text())


def schedule(c):
    frame = pd.read_csv(ROOT / c['schedule'], keep_default_na=False)
    if not set(KEY).issubset(frame) or frame.duplicated(KEY).any():
        raise ValueError('Missing or duplicate frozen schedule keys')
    if set(frame.test_subject) != set(c['participants']):
        raise ValueError('All seven subjects including sub-03 must remain scheduled')
    return frame


def fresh(path):
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def prepare(c, out):
    import nibabel as nib
    rows = []
    for subject in c['participants']:
        path = ROOT / c['dataset'] / subject / 'anat' / f'{subject}_T1w.nii'
        row = dict(subject=subject, path=str(path), status='MISSING', reason='T1 not found')
        if path.is_file():
            image = nib.load(path)
            q, qc = image.get_qform(coded=True)
            s, sc = image.get_sform(coded=True)
            units = image.header.get_xyzt_units()[0]
            valid = (len(image.shape) == 3 and units == 'mm' and np.isfinite(image.affine).all()
                     and abs(np.linalg.det(image.affine[:3, :3])) > 1e-10 and (qc > 0 or sc > 0)
                     and not (qc > 0 and sc > 0 and not np.allclose(q, s, atol=1e-3)))
            row.update(status='PASS' if valid else 'FAIL', reason='' if valid else 'Invalid/ambiguous geometry or units',
                       shape=list(image.shape), axis_codes=nib.aff2axcodes(image.affine), units=units,
                       affine_ras_mm=image.affine.tolist(), qform_code=int(qc), sform_code=int(sc), sha256=digest(path),
                       sidecar_present=path.with_suffix('.json').is_file())
        rows.append(row)
    frozen = schedule(c)
    out = fresh(out)
    write_json(out / '01_mri_inventory.json', rows)
    pd.DataFrame([{'subject': r['subject'], 'path': r['path'], 'status': r['status'], 'reason': r['reason']} for r in rows]).to_csv(out / '02_participants.csv', index=False)
    frozen.to_csv(out / '03_frozen_schedule.csv', index=False)
    source = ROOT / c['frozen_settings']
    if source.is_dir():
        shutil.copytree(source, out / 'source_configurations')
    inputs = [ROOT / c['schedule'], *source.glob('*')] if source.is_dir() else [ROOT / c['schedule']]
    write_json(out / '04_manifest.json', dict(design=c['design'], baseline_commit=c['baseline_commit'],
               configuration=c, inputs={str(p): digest(p) for p in inputs if p.is_file()},
               dependencies={name: shutil.which(name) for name in ['antsMultivariateTemplateConstruction2.sh', 'recon-all']},
               python_dependencies={name: importlib.util.find_spec(name) is not None for name in ['nibabel', 'mne', 'numpy', 'pandas', 'h5py']},
               ready=all(r['status'] == 'PASS' for r in rows) and source.is_dir()))


def template_plan(c, out):
    """Plan only: ANTs outputs must be reviewed and indexed before downstream use."""
    out = fresh(out)
    images = [str(ROOT / c['dataset'] / s / 'anat' / f'{s}_T1w.nii') for s in c['participants']]
    import shlex
    command = ['antsMultivariateTemplateConstruction2.sh', '-d', '3', '-o', str(out / 'group_'),
               '-i', str(c['registration']['iterations']), '-t', 'SyN', '-m', c['registration']['metric'], *images]
    (out / '01_build_template.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\n' + shlex.join(command) + '\n')
    write_json(out / '02_construction_provenance.json', dict(
        participants=c['participants'], input_images=images,
        preserve=['final template header', 'input-order subject associations', 'groupwise logs',
                  'final iteration affines and deformation fields'],
        scope='Template construction only. These transformations are NOT used in localization or evaluation.',
        status='PENDING_TEMPLATE_CONSTRUCTION'))
    (out / 'README.txt').write_text('PLAN ONLY. Review Script 01 MRI geometry first, then run the shell command manually. '
        'Groupwise registration builds the anatomical average; retain transforms for construction provenance only. '
        'Downstream comparison uses rigid EEG coregistration and diagnostic HEAD distances, without MRI point mapping.\n')


def anatomy_plan(c, out):
    out = fresh(out)
    write_json(out / '01_anatomy_contract.json', dict(
        template='Reviewed final group template from stage 02',
        segmentation_command='recon-all -i TEMPLATE.nii.gz -s average_seven -all',
        bem_commands=['mne.bem.make_watershed_bem(subject="average_seven", subjects_dir=...)',
                      'mne.make_bem_model(subject="average_seven", ico=4, conductivity=(0.3,0.006,0.3), subjects_dir=...)',
                      'mne.make_bem_solution(model)',
                      'mne.setup_source_space(subject="average_seven", spacing="ico5", subjects_dir=...)'],
        required_qa=['closed oriented meshes, no self-intersections', 'inner skull strictly within outer skull within scalp',
                     'positive volumes, no degenerate triangles', 'sources inside inner skull; source/scalp units metres in MNE',
                     'template scanner RAS <-> FreeSurfer surface RAS from actual volume headers',
                     'same rigid coregistration protocol in both arms; per-electrode distances and landmark review',
                     'median <= 10 mm, p95 <= 20 mm plus visual review; sub-03 may fail and must remain scheduled'],
        forward_gate='No forward computation before all QA gates PASS; never substitute native anatomy for a failed template case',
        forward_adapter='Use 04a_coregister.py and 04b_make_forward.py; explicit anatomy paths, shared rigid protocol and reviewed gates',
        status='REQUIRES_EXECUTION_AND_REVIEW; see docs/RUN.md'))


def comparison_plan(c, out):
    frozen = schedule(c)
    out = fresh(out)
    historical = pd.read_csv(ROOT / 'cross_subject_fsaverage/outputs/02_coregister_eeg/01_subject_coregistration_audit.csv')
    historical.to_csv(out / '04_historical_alignment_audit.csv', index=False)
    prior_status = historical.set_index('subject').status.to_dict()
    records = []
    for arm in c['arms']:
        for row in frozen[KEY].to_dict('records'):
            records.append(dict(**row, arm=arm, status='PENDING', failure_reason='', alignment_status='PENDING',
                                historical_alignment_status=prior_status.get(row['test_subject'], 'UNKNOWN') if arm == 'fsaverage' else 'NEW_ANATOMY_PENDING'))
    pd.DataFrame(records).to_csv(out / '01_two_arm_schedule.csv', index=False)
    old = pd.read_csv(ROOT / c['fsaverage_results'], low_memory=False)
    audit = dict(existing_results=str(ROOT / c['fsaverage_results']), approved_for_reuse=False,
                 missing_fingerprint_columns=[k for k in FINGERPRINTS if k not in old],
                 reason='Existing diagnostic HEAD-distance endpoint matches this design. Reuse is not yet approved: '
                        'require per-job identical fingerprints and reviewed rigid coregistration QA. No MRI registration is required.',
                 native_error_valid_counts=old.get('localization_error_valid', pd.Series(dtype=str)).astype(str).value_counts().to_dict())
    write_json(out / '02_fsaverage_reuse_audit.json', audit)
    write_json(out / '03_job_contract.json', dict(keys=KEY, configuration=c,
               required_per_job=FINGERPRINTS + ['arm', 'status', 'failure_reason', 'alignment_status',
                'coordinate_frame', 'coordinate_units', 'estimated_head_x_mm', 'estimated_head_y_mm', 'estimated_head_z_mm'],
               execution='Use 04a_coregister.py, 04b_make_forward.py and 04c_reconstruct.py with --execute; frozen no-warp Engine and identical protocols in both arms',
               mapping='Return model HEAD estimates directly (convert metres to mm once); no MRI deformation or template-to-native coordinate mapping',
               original_reference='Original stimulation midpoint in test HEAD mm using the existing audited native surface-to-head H5; no MRI registration or template projection',
               endpoint='UNREGISTERED_HEAD_DISTANCE_ONLY_ANATOMICAL_ACCURACY_UNVALIDATED',
               localization_error_valid=False,
               references_independent_of_arm=True))


def evaluate(c, out, result_path, reference_path):
    """Join to planned universe; missing results and rigid alignment failures remain FAIL; distances are diagnostic."""
    expected = schedule(c)[KEY]
    results = pd.read_csv(result_path, keep_default_na=False)
    refs = pd.read_csv(reference_path, keep_default_na=False)
    result_columns = KEY + ['arm', 'status', 'failure_reason', 'alignment_status',
                            'coordinate_frame', 'coordinate_units', 'estimated_head_x_mm', 'estimated_head_y_mm', 'estimated_head_z_mm'] + FINGERPRINTS
    if not set(result_columns).issubset(results) or not set(['test_subject','run','coordinate_frame','coordinate_units','gt_head_x_mm','gt_head_y_mm','gt_head_z_mm','provenance']).issubset(refs):
        raise ValueError('Result/reference schema does not meet stage 04 contract')
    if results.duplicated(KEY + ['arm']).any() or refs.duplicated(['test_subject','run']).any():
        raise ValueError('Duplicate result or reference keys')
    if not results.status.isin(['PASS','FAIL']).all() or not results.arm.isin(c['arms']).all():
        raise ValueError('Invalid arm/status')
    universe = pd.concat([expected.assign(arm=a) for a in c['arms']], ignore_index=True)
    extras = results.merge(universe, on=KEY+['arm'], how='left', indicator=True)
    if extras._merge.eq('left_only').any():
        raise ValueError('Unplanned result keys')
    merged = universe.merge(results, on=KEY+['arm'], how='left', validate='one_to_one')
    merged = merged.merge(refs, on=['test_subject','run'], suffixes=('', '_gt'), how='left', validate='many_to_one')
    distances, statuses, reasons = [], [], []
    for _, r in merged.iterrows():
        reason = ''
        if pd.isna(r.status): reason = 'MISSING_RESULT'
        elif r.status == 'FAIL': reason = r.failure_reason or 'SOLVER_FAIL'
        elif r.alignment_status != 'PASS': reason = 'ALIGNMENT_FAIL'
        elif r.coordinate_frame != 'model_head' or r.coordinate_frame_gt != 'test_head': reason = 'FRAME_MISMATCH'
        elif r.coordinate_units != 'mm' or r.coordinate_units_gt != 'mm': reason = 'UNIT_MISMATCH'
        else:
            numeric = pd.to_numeric(pd.Series([r[k] for k in ['estimated_head_x_mm','estimated_head_y_mm','estimated_head_z_mm','gt_head_x_mm','gt_head_y_mm','gt_head_z_mm']]), errors='coerce').to_numpy(float)
            if not np.isfinite(numeric).all(): reason = 'INVALID_COORDINATES'
            elif pd.isna(r.provenance) or not r.provenance: reason = 'MISSING_ORIGINAL_REFERENCE_PROVENANCE'
        distances.append(float(np.linalg.norm(numeric[:3]-numeric[3:6])) if not reason else np.nan)
        statuses.append('FAIL' if reason else 'PASS'); reasons.append(reason)
    merged['evaluation_status'] = statuses; merged['evaluation_failure_reason'] = reasons; merged['diagnostic_distance_mm'] = distances
    merged['localization_error_valid'] = False
    merged['interpretation'] = 'UNREGISTERED_HEAD_DISTANCE_ONLY_ANATOMICAL_ACCURACY_UNVALIDATED'
    a = merged[merged.arm.eq('average_seven')]; b = merged[merged.arm.eq('fsaverage')]
    pairs = a.merge(b, on=KEY, suffixes=('_average_seven','_fsaverage'), validate='one_to_one')
    match = pd.Series(True, index=pairs.index)
    for field in FINGERPRINTS:
        left, right = pairs[field+'_average_seven'], pairs[field+'_fsaverage']
        match &= left.notna() & right.notna() & left.ne('') & left.eq(right)
    pairs['protocol_match'] = match
    pairs['both_pass'] = pairs.evaluation_status_average_seven.eq('PASS') & pairs.evaluation_status_fsaverage.eq('PASS')
    pairs['paired_eligible'] = pairs.both_pass & match
    pairs['diagnostic_difference_mm'] = (pairs.diagnostic_distance_mm_average_seven - pairs.diagnostic_distance_mm_fsaverage).where(pairs.paired_eligible)
    out = fresh(out)
    merged.to_csv(out / '01_all_outcomes.csv', index=False)
    pairs.to_csv(out / '02_paired_differences.csv', index=False)
    merged.groupby(['arm','test_subject','evaluation_status']).size().rename('jobs').reset_index().to_csv(out / '03_pass_fail_counts.csv', index=False)
    write_json(out / '04_manifest.json', dict(design=c['design'], baseline_commit=c['baseline_commit'], inputs={str(p):digest(p) for p in [result_path, reference_path]},
        planned_jobs=len(universe), paired_eligible=int(pairs.paired_eligible.sum()), comparison=c['comparison'], localization_error_valid=False, note='Positive difference means larger average-seven diagnostic distance; not validated anatomical localization error. No MRI warp used; no missing failures deleted.'))


def main(stage):
    parser = argparse.ArgumentParser(description=f'Average-seven stage {stage}; heavy computations are never launched automatically')
    parser.add_argument('--config', type=Path, default=EXP / 'config/experiment.json')
    parser.add_argument('--output-root', type=Path, default=EXP / 'outputs' / stage)
    parser.add_argument('--results', type=Path, help='Existing two-arm HEAD results CSV; not generated by the planning stages')
    parser.add_argument('--references', type=Path, help='Existing audited original stimulation references in test-HEAD millimetres')
    args = parser.parse_args()
    c = config(args.config)
    if stage == '05_evaluate_no_warp':
        if args.results is None or args.references is None: parser.error('--results and --references required')
        for label, path in [('results', args.results), ('references', args.references)]:
            if not path.is_file():
                parser.error(f'--{label} file does not exist: {path}. '
                             'The example CSV names are placeholders. Stage 05 requires actual localization results '
                             'and audited test-HEAD references. Stages 02–04 currently emit plans/contracts only; '
                             'Run the explicit execution stages in docs/RUN.md, then use 04d_export_evaluate.py to generate both CSVs and evaluate. '
                             'The execution adapters require reviewed anatomy/alignment and completed arm jobs.')
        evaluate(c, args.output_root, args.results, args.references)
    else:
        {'01_prepare_experiment':prepare, '02_build_group_template':template_plan,
         '03_prepare_bem_source':anatomy_plan, '04_plan_comparison':comparison_plan}[stage](c,args.output_root)
    print(args.output_root)

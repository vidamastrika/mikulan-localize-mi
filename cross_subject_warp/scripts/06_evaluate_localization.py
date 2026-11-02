#!/usr/bin/env python3
# SCRIPT 06 — Preflight or evaluate saved native peaks using validated MRI mappings.
# From project root; project Python plus ANTs runtime:
# micromamba run -n localizemi-registration .venv/bin/python \
#   cross_subject_warp/scripts/06_evaluate_localization.py --phase evaluate \
#   --preparation-root cross_subject_warp/outputs/01_prepare_experiment_review \
#   --output-root cross_subject_warp/outputs/06_evaluate_localization_mapped
#
# Repeat into a NEW folder (earlier reports are preserved):
# micromamba run -n localizemi-registration .venv/bin/python \
#   cross_subject_warp/scripts/06_evaluate_localization.py --phase evaluate \
#   --output-root cross_subject_warp/outputs/06_evaluate_localization_mapped_review
#
# Header/input preflight only:
# python cross_subject_warp/scripts/06_evaluate_localization.py --phase audit \
#   --output-root cross_subject_warp/outputs/06_evaluate_localization_preflight_review
# Help: python cross_subject_warp/scripts/06_evaluate_localization.py --help
#
# Evaluation exit 0: all scheduled rows processed, including retained failures.
# Evaluation exit 2: reports written with mapping-execution failures to inspect.
# Audit exit 2: preflight reports written; no mapped errors were calculated.
# Existing destinations are refused. No inverse fits or registration fits.
"""Evaluate saved native peaks with explicit affine/SyN QA and failure retention.

--phase audit retains the lightweight header/input preflight. --phase evaluate
verifies native data, all seven bridge reports and the current registration QA
provenance. Passing mode/pairs are mapped; failed mode/pairs remain rows with
blank errors. Each mapped point must pass MRI/field domains and <=0.1 mm
roundtrip. No source reselection, EEG/lead-field warping or new inverse fitting.

Within baseline appears once per scenario; each cross record has affine/SyN
rows. Preserve native status; use evaluation_status/localization_error_valid for
metrics. Matched-mode comparison requires both errors valid on the same key.
Primary and sensitivity remain separate; test ground truth is never reference-
warped. Numbered evaluation artifacts 01–08 are retained; additional QA,
provenance, matched comparison and point audits are numbered 09–13.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
EXP=ROOT/'cross_subject_warp'
KEY=['method','reference_subject','test_subject','run','condition']
PEAKS=['estimated_head_x_mm','estimated_head_y_mm','estimated_head_z_mm','peak_vertex','peak_time_s']
META=['actual_channels_json','nominal_channels_json','original_bad_channels_json',
      'eeg_data_sha256','epoch_times_sha256','montage_template_subject','n_epochs','covariance_method_selected']
PROVENANCE={'source_csv','source_csv_sha256','native_input_validation','metadata_completed',
            'metadata_completion_source','anatomical_mapping_status'}
SPECS={
 'loso_primary':('LOSO','primary','03_reference_based_final',1464,1463),
 'single_primary':('Single-reference','primary','05_single_reference_final',4392,4378),
 'loso_sensitivity':('LOSO','sensitivity','03_gamma_wholebrain_sensitivity',1464,1464),
 'single_sensitivity':('Single-reference','sensitivity','05_wholebrain_hemisphere_sensitivity',4392,4381)}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda:file.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(path):return pd.read_csv(path,dtype=str,keep_default_na=False)


def canonical(value):
    if isinstance(value,dict):return {k:canonical(v) for k,v in sorted(value.items())}
    if isinstance(value,list):return [canonical(v) for v in value]
    if isinstance(value,str) and value[:1] in ['{','[']:
        try:return canonical(json.loads(value))
        except ValueError:pass
    if isinstance(value,bool):return str(value).lower()
    try:return format(float(value),'.12g')
    except (ValueError,TypeError):return str(value)


def expected_keys(cases,methods,strategy):
    keys=set()
    subjects={f'sub-{i:02d}' for i in range(1,8)}
    for case in cases.to_dict('records'):
        references=[case['reference_subject']] if strategy=='LOSO' else sorted(subjects-{case['test_subject']})
        conditions=['within','cross_reference'] if strategy=='LOSO' else ['cross_reference']
        for method in methods:
            for ref in references:
                for condition in conditions:keys.add((method,ref,case['test_subject'],case['run'],condition))
    return keys


def validate_table(table,expected,label,configs,strategy,rows,passed):
    required=set(KEY+PEAKS+META+['model_subject','configuration_key','configuration_parameters',
        'status','failure_reason','coordinate_frame','coordinate_units','native_input_validation'])
    if not required.issubset(table):raise ValueError(f'{label}: missing columns {sorted(required-set(table))}')
    if table.duplicated(KEY).any():raise ValueError(f'{label}: duplicate keys')
    if len(table)!=rows or set(map(tuple,table[KEY].values))!=expected:raise ValueError(f'{label}: schedule/key coverage differs')
    if not table.status.isin(['PASS','FAIL']).all() or int(table.status.eq('PASS').sum())!=passed:
        raise ValueError(f'{label}: outcome counts differ')
    if not table.native_input_validation.eq('PASS_RECOMPUTED').all():raise ValueError(f'{label}: incomplete native input verification')
    for row in table.to_dict('records'):
        model=row['test_subject'] if row['condition']=='within' else row['reference_subject']
        if row['model_subject']!=model or row['montage_template_subject']!=row['reference_subject']:
            raise ValueError(f'{label}: model/montage identity differs')
        config=configs[(row['method'],row['test_subject'] if strategy=='LOSO' else row['reference_subject'])]
        if row['configuration_key']!=config['configuration_key']:
            raise ValueError(f'{label}: selected configuration differs')
        selected=json.loads(config['configuration_parameters']);actual=json.loads(row['configuration_parameters'])
        if any(k not in actual or canonical(v)!=canonical(actual[k]) for k,v in selected.items()):
            raise ValueError(f'{label}: parameter settings differ')
        if any(not row[field] for field in META):raise ValueError(f'{label}: missing input metadata')
        for field in ['eeg_data_sha256','epoch_times_sha256']:
            value=row[field]
            if len(value)!=64 or any(c not in '0123456789abcdef' for c in value.lower()):
                raise ValueError(f'{label}: invalid fingerprint')
        actual_channels=json.loads(row['actual_channels_json']);bad=json.loads(row['original_bad_channels_json'])
        nominal=json.loads(row['nominal_channels_json'])
        if not all(isinstance(v,list) and all(isinstance(c,str) for c in v) for v in [actual_channels,bad,nominal]):
            raise ValueError(f'{label}: malformed channel lists')
        if not actual_channels or not nominal or len(actual_channels)!=len(set(actual_channels)) or set(actual_channels)&set(bad):
            raise ValueError(f'{label}: invalid channel selection')
        if row['status']=='FAIL':
            if not row['failure_reason'] or any(row[field] for field in PEAKS):raise ValueError(f'{label}: malformed failure')
        else:
            point=np.array([float(row[field]) for field in PEAKS[:3]])
            if not np.isfinite(point).all() or np.linalg.norm(point)>1000:raise ValueError(f'{label}: invalid peak')
            if row['coordinate_frame']!='model_head' or row['coordinate_units']!='mm':raise ValueError(f'{label}: peak frame/units differ')
    if strategy=='LOSO':
        for _,group in table.groupby(['method','test_subject','run']):
            for field in ['actual_channels_json','nominal_channels_json','eeg_data_sha256','epoch_times_sha256']:
                if group[field].nunique()!=1:raise ValueError(f'{label}: within/reference paired {field} differs')
    return dict(dataset=label,check='NATIVE_INPUTS_AND_SCHEDULE',status='PASS',rows=len(table),
                passed=int(table.status.eq('PASS').sum()),failed=int(table.status.eq('FAIL').sum()))


def validate_variant(primary,variant,number,label):
    p=primary.set_index(KEY).sort_index();v=variant.set_index(KEY).sort_index()
    if not p.index.equals(v.index):raise ValueError(f'{label}: variant keys differ')
    if 'sensitivity_override' not in v:raise ValueError(f'{label}: override flags absent')
    flags=v.sensitivity_override.str.lower()
    if not flags.isin(['true','false']).all():raise ValueError(f'{label}: invalid override flags')
    override=flags.eq('true')
    if int(override.sum())!=number:raise ValueError(f'{label}: override count differs')
    for field in sorted(set(p)-PROVENANCE):
        if field not in v:raise ValueError(f'{label}: original field absent: {field}')
        for left,right in zip(p.loc[~override,field],v.loc[~override,field]):
            if canonical(left)!=canonical(right):raise ValueError(f'{label}: nonoverride changed: {field}')
    immutable=['configuration_key','configuration_parameters','model_subject',*META]
    for key in p.index[override]:
        a=p.loc[key];b=v.loc[key]
        if a.status!='FAIL' or b.status!='PASS':raise ValueError(f'{label}: override must be FAIL to PASS')
        if b.get('peak_selection_policy')!='whole_brain_absolute_peak' or b.get('hemisphere_constraint_applied','').lower()!='false':
            raise ValueError(f'{label}: override peak policy differs')
        for field in immutable:
            if canonical(a[field])!=canonical(b[field]):raise ValueError(f'{label}: override inputs changed: {field}')
    return dict(dataset=label,check='SENSITIVITY_OVERRIDES',status='PASS',overrides=number)


def coordinate_preflight(dataset,subject):
    """Numerical/header checks only; never promote native space names to RAS proof."""
    row=dict(subject=subject,numerical_status='FAIL',physical_bridge_status='UNRESOLVED',reason='')
    paths=[]
    try:
        import h5py
        import nibabel as nib
        h5=dataset/'derivatives/sourcemodelling'/subject/'xfm'/f'{subject}_from-head_to-surface.h5'
        mri=dataset/'derivatives/epochs'/subject/'anat'/f'{subject}_mod-T1w_maskface.nii'
        paths.extend([h5,mri])
        with h5py.File(h5,'r') as file:matrix=np.asarray(file['trans'],float)
        if matrix.shape!=(4,4) or not np.isfinite(matrix).all():raise ValueError('Invalid H5 matrix')
        if not np.allclose(matrix[3],[0,0,0,1],atol=1e-5,rtol=0):raise ValueError('Invalid homogeneous row')
        rotation=matrix[:3,:3]
        if not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-5,rtol=0) or abs(np.linalg.det(rotation)-1)>1e-5:
            raise ValueError('H5 is not a proper rigid transform')
        image=nib.load(str(mri))
        if not np.isfinite(image.affine).all() or abs(np.linalg.det(image.affine[:3,:3]))<1e-12:
            raise ValueError('Invalid NIfTI world affine')
        for space in ['surface','T1w']:
            path=dataset/'derivatives/epochs'/subject/'ieeg'/f'{subject}_task-seegstim_space-{space}_coordsystem.json'
            paths.append(path);doc=json.loads(path.read_text())
            if doc.get('iEEGCoordinateUnits')!='m':raise ValueError(f'{space}: contact units differ from metres')
        row.update(numerical_status='PASS_NUMERICAL_ONLY',h5_sha256=sha(h5),mri_sha256=sha(mri),
            nifti_shape=json.dumps(list(image.shape)),nifti_axis_codes=''.join(nib.aff2axcodes(image.affine)),
            nifti_world_affine=json.dumps(image.affine.tolist()),nifti_spatial_unit=image.header.get_xyzt_units()[0],
            qform_code=int(image.header['qform_code']),sform_code=int(image.header['sform_code']),
            reason='Numerical checks do not establish head/surface to NIfTI physical RAS correspondence')
    except Exception as exc:row['reason']=f'{type(exc).__name__}: {exc}'
    return row,paths


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['audit','evaluate'],default='audit')
    p.add_argument('--input-root',type=Path,default=EXP/'outputs')
    p.add_argument('--preparation-root',type=Path,default=EXP/'outputs/01_prepare_experiment_review')
    p.add_argument('--resource-root',type=Path,default=EXP/'source/01_registration_resources/outputs')
    p.add_argument('--bridge-root',type=Path,default=EXP/'outputs/06_coordinate_bridge')
    p.add_argument('--qa-root',type=Path,default=EXP/'outputs/06_registration_qa')
    p.add_argument('--output-root',type=Path,default=EXP/'outputs/06_evaluate_localization')
    args=p.parse_args()
    for name,value in vars(args).items():
        if isinstance(value,Path):setattr(args,name,value.expanduser().resolve())
    if args.output_root.exists() or args.output_root.is_symlink():raise FileExistsError('Output exists; choose a NEW --output-root')
    if args.phase=='evaluate':
        spec=importlib.util.spec_from_file_location('mapped_evaluation_impl',Path(__file__).with_name('06_mapped_evaluation.py'))
        impl=importlib.util.module_from_spec(spec);sys.modules[spec.name]=impl;spec.loader.exec_module(impl)
        return impl.run(args,sys.modules[__name__])
    if args.output_root.exists() or args.output_root.is_symlink():raise FileExistsError('Output exists; choose a NEW --output-root')
    protected_inputs=[args.input_root/spec[2] for spec in SPECS.values()]
    protected_inputs += [args.input_root/name for name in ['02_select_loso_parameters','04_single_reference_parameters','03_05_native_completion_audit']]
    for protected in [args.preparation_root,args.resource_root,args.bridge_root,args.qa_root,EXP/'source',EXP/'scripts',ROOT/'cross_subject_no_warp',ROOT/'cross_subject',*protected_inputs]:
        if args.output_root==protected or protected in args.output_root.parents or args.output_root in protected.parents:
            raise ValueError(f'Output overlaps protected input: {protected}')
    provenance={}
    def track(path):
        path=Path(path)
        provenance[str(path)]=dict(path=str(path),sha256=sha(path),size_bytes=path.stat().st_size)
    source=EXP/'source/02_parameter_grids'
    spec=importlib.util.spec_from_file_location('evaluation_selection_checks',EXP/'scripts/02_select_loso_parameters.py')
    shared=importlib.util.module_from_spec(spec);sys.modules[spec.name]=shared;spec.loader.exec_module(shared)
    _,doc=shared.snapshot(argparse.Namespace(source_root=source))
    shared.verify_preparation(args.preparation_root,source,doc)
    _,cases=shared.load_plan(source,doc)
    selection={}
    for strategy,folder,file,hash_field in [('LOSO','02_select_loso_parameters','01_loso_selected_parameters.csv','selected_parameters_sha256'),
        ('Single-reference','04_single_reference_parameters','01_single_reference_selected_parameters.csv','selected_csv_sha256')]:
        path=args.input_root/folder/file;manifest=args.input_root/folder/'07_selection_manifest.json'
        state=json.loads(manifest.read_text());track(path);track(manifest)
        if state.get('status')!='COMPLETE' or state.get('actual_selections')!=84 or sha(path)!=state[hash_field]:raise ValueError('Incomplete/modified selections')
        table=read(path);subject='test_subject' if strategy=='LOSO' else 'reference_subject'
        if len(table)!=84 or table.duplicated(['method',subject]).any():raise ValueError('Selection coverage differs')
        selection[strategy]={(r['method'],r[subject]):r for r in table.to_dict('records')}
    completion=args.input_root/'03_05_native_completion_audit/05_completion_manifest.json'
    state=json.loads(completion.read_text());track(completion)
    if state.get('status')!='NATIVE_ATTEMPT_DATA_COMPLETE':raise ValueError('Native completion has not passed')
    data={};checks=[]
    for label,(strategy,scenario,folder,total,passing) in SPECS.items():
        path=args.input_root/folder/'01_execution_status.csv';report=args.input_root/folder/'05_validation_report.json'
        detail=json.loads(report.read_text());track(path);track(report)
        if sha(path)!=detail['completed_csv_sha256'] or sha(path)!=state['datasets'][folder]['completed_csv_sha256']:
            raise ValueError(f'Completed dataset changed: {path}')
        data[label]=read(path)
        keys=expected_keys(cases,shared.METHODS,strategy)
        checks.append(validate_table(data[label],keys,label,selection[strategy],strategy,total,passing))
    checks.append(validate_variant(data['loso_primary'],data['loso_sensitivity'],1,'LOSO'))
    checks.append(validate_variant(data['single_primary'],data['single_sensitivity'],3,'Single-reference'))
    source_doc=args.preparation_root/'10_source_provenance.json';track(source_doc)
    dataset=Path(json.loads(source_doc.read_text())['locations']['dataset'])
    if args.output_root==dataset or dataset in args.output_root.parents or args.output_root in dataset.parents:raise ValueError('Output overlaps dataset')
    coordinates=[]
    for subject in sorted({f'sub-{i:02d}' for i in range(1,8)}):
        row,paths=coordinate_preflight(dataset,subject);coordinates.append(row)
        for path in paths:
            if path.is_file():track(path)
    imported=json.loads(source_doc.read_text())['imported_files']
    imported_by_path={str(Path(r['copied_path']).resolve()):r for r in imported}
    registrations=[]
    for reference in sorted({f'sub-{i:02d}' for i in range(1,8)}):
        for test in sorted({f'sub-{i:02d}' for i in range(1,8)}-{reference}):
            pair=f'{reference}_to_{test}'
            for mode,folder in [('affine','09_anatomical_registration'),('syn','19_syn_registration')]:
                base=args.resource_root/folder/pair
                names=[f'{pair}_0GenericAffine.mat',f'{reference}_in_{test}.nii.gz'] if mode=='affine' else [
                    f'{pair}_syn_0GenericAffine.mat',f'{pair}_syn_1Warp.nii.gz',f'{pair}_syn_1InverseWarp.nii.gz',
                    'donor_syn_in_target.nii.gz','target_syn_in_donor.nii.gz','syn_registration_quality.json','ants_registration.log']
                missing=[];unrecorded=[];size_changed=[]
                for name in names:
                    path=base/name;entry=imported_by_path.get(str(path.resolve()))
                    if not path.is_file():missing.append(name)
                    elif entry is None:unrecorded.append(name)
                    elif path.stat().st_size!=entry['size_bytes']:size_changed.append(name)
                registrations.append(dict(reference_subject=reference,test_subject=test,registration_mode=mode,
                    files_present=not missing,import_provenance_present=not unrecorded,recorded_sizes_match=not size_changed,
                    missing_files=';'.join(missing),unrecorded_files=';'.join(unrecorded),changed_size_files=';'.join(size_changed),
                    current_transform_checksums_verified=False,physical_point_mapping_validated=False,
                    readiness='BLOCKED_PENDING_GEOMETRY_AND_QA'))
    blockers=[dict(code='PHYSICAL_FRAME_NOT_CERTIFIED_BY_HEADER_PREFLIGHT',detail='Header-only audit does not certify physical bridges; evaluation verifies the separate seven-subject bridge reports.'),
        dict(code='REGISTRATION_QA_NOT_CONSUMED_BY_PREFLIGHT',detail='Evaluation reads full pair QA, preserves failed mode/pairs and checks current source hashes.'),
        dict(code='CURRENT_TRANSFORM_HASH_VALIDATION_PENDING',detail='This preflight checks existence, imported provenance and sizes; transform contents were hashed at import, not rehashed here.'),
        dict(code='PREFLIGHT_ONLY_NO_MAPPING_REQUESTED',detail='Audit does not consume bridge/registration QA as mapping authorization. Use --phase evaluate for the full provenance and per-point checks.')]
    if any(row['numerical_status']!='PASS_NUMERICAL_ONLY' for row in coordinates):blockers.append(dict(code='NUMERICAL_COORDINATE_CHECK_FAILED',detail='Inspect 02_coordinate_validation.csv.'))
    if any(not r['files_present'] or not r['import_provenance_present'] or not r['recorded_sizes_match'] for r in registrations):
        blockers.append(dict(code='REGISTRATION_RESOURCE_CHECK_FAILED',detail='Inspect 09_registration_readiness.csv.'))
    track(Path(__file__))
    manifest=dict(status='PREFLIGHT_COMPLETE_EVALUATION_BLOCKED',phase='audit',native_inputs_validated=True,
        primary_rows=5856,sensitivity_rows=5856,planned_registration_pairs=42,registration_modes=['affine','syn'],
        numerical_coordinate_subjects_passed=sum(r['numerical_status']=='PASS_NUMERICAL_ONLY' for r in coordinates),
        anatomical_evaluation_ready=False,localization_errors_calculated=False,point_mapping_executed=False,
        inverse_reconstruction_executed=False,registration_fitting_executed=False,
        dataset=str(dataset),blockers=blockers,script_sha256=sha(__file__))
    args.output_root.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.evaluation_preflight_',dir=args.output_root.parent) as temp:
        stage=Path(temp)/'results';stage.mkdir()
        pd.DataFrame(checks).to_csv(stage/'01_input_validation.csv',index=False)
        pd.DataFrame(coordinates).to_csv(stage/'02_coordinate_validation.csv',index=False)
        (stage/'07_evaluation_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        pd.DataFrame(registrations).to_csv(stage/'09_registration_readiness.csv',index=False)
        pd.DataFrame(blockers).to_csv(stage/'10_readiness_blockers.csv',index=False)
        (stage/'11_source_provenance.json').write_text(json.dumps(list(provenance.values()),indent=2)+'\n')
        (stage/'README.txt').write_text(__doc__+'\nNo localization-error tables were generated.\n')
        if args.output_root.exists():raise FileExistsError(args.output_root)
        stage.rename(args.output_root)
    print('NATIVE INPUT AUDIT PASSED; anatomical evaluation remains BLOCKED.')
    print('Reports:',args.output_root)
    print('Exit 2 indicates unresolved readiness gates; no localization errors were calculated.')
    return 2


if __name__=='__main__':
    with tempfile.TemporaryDirectory(prefix='mapped-evaluation-config-') as config:
        os.environ['_MNE_FAKE_HOME_DIR']=config
        os.environ['MPLCONFIGDIR']=config
        raise SystemExit(main())

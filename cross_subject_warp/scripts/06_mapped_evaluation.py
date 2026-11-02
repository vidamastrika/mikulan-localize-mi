#!/usr/bin/env python3
# Internal Script 06 evaluation implementation. Run the public entry point:
# micromamba run -n localizemi-registration .venv/bin/python \
#   cross_subject_warp/scripts/06_evaluate_localization.py --phase evaluate \
#   --preparation-root cross_subject_warp/outputs/01_prepare_experiment_review \
#   --output-root cross_subject_warp/outputs/06_evaluate_localization_mapped
# Existing outputs are refused. Evaluation maps saved peaks only; no fitting.
"""Mapped evaluation with retained reconstruction, pair-QA and point-QA failures.

Within baselines occur once; each cross record has separate affine/SyN rows.
Never map a failed pair, reselect a source peak or transform test ground truth
through a reference warp. Summaries retain all scheduled rows in denominators;
matched-mode comparisons require both errors to be valid on the exact same key.
"""
from __future__ import annotations
import importlib.util
import importlib.metadata
import json
from pathlib import Path
import sys
import tempfile

import nibabel as nib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
EXP=ROOT/'cross_subject_warp'
SUBJECTS=[f'sub-{i:02d}' for i in range(1,8)]
PAIR_KEY=['reference_subject','test_subject','registration_mode']
MATRICES=['surface_to_scanner_ras_mm','scanner_ras_to_surface_mm','head_to_scanner_ras_mm','scanner_ras_to_head_mm']


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    obj=importlib.util.module_from_spec(spec);sys.modules[name]=obj;spec.loader.exec_module(obj)
    return obj


def apply(matrix,points):
    matrix=np.asarray(matrix,float)
    return np.asarray(points,float)@matrix[:3,:3].T+matrix[:3,3]


def inside(points,image):
    points=np.asarray(points,float).reshape(-1,3)
    vox=nib.affines.apply_affine(np.linalg.inv(image.affine),points)
    return np.isfinite(points).all(axis=1)&np.all((vox>=1)&(vox<np.asarray(image.shape[:3])-2),axis=1)


def validate_pair_qa(table,manifest,qa):
    expected={(r,t,m) for r in SUBJECTS for t in SUBJECTS if r!=t for m in ['affine','syn']}
    if table.duplicated(PAIR_KEY).any() or set(map(tuple,table[PAIR_KEY].values))!=expected:
        raise ValueError('Pair QA coverage differs')
    if not table.status.isin(['PASS_NUMERICAL_MAPPING_QA','FAIL']).all():raise ValueError('Unknown pair QA status')
    if int(table.status.eq('PASS_NUMERICAL_MAPPING_QA').sum())!=manifest['mode_pair_checks_passed']:
        raise ValueError('QA manifest/table counts disagree')
    fields={'max_roundtrip_mm':qa.MAX_ROUNDTRIP_MM,'max_image_rmse':qa.MAX_RMSE,
            'min_coverage':qa.MIN_COVERAGE,'samples':qa.SAMPLES,'foreground_seed':2901,'grid_tolerance_mm':qa.GRID_TOLERANCE_MM}
    if any(manifest.get(k)!=v for k,v in fields.items()):raise ValueError('Registration QA thresholds/sampling differ')
    for row in table.to_dict('records'):
        if row['status']=='FAIL':
            if not row['reason']:raise ValueError('Pair failure reason absent')
            continue
        limits={'grid_max_error_mm':qa.GRID_TOLERANCE_MM,'image_reproduction_rmse':qa.MAX_RMSE,
                'reference_roundtrip_max_mm':qa.MAX_ROUNDTRIP_MM,'test_roundtrip_max_mm':qa.MAX_ROUNDTRIP_MM}
        if row['registration_mode']=='syn':limits['inverse_image_reproduction_rmse']=qa.MAX_RMSE
        for field,limit in limits.items():
            value=float(row[field])
            if not np.isfinite(value) or value<0 or value>limit:raise ValueError(f'Pair PASS contradicts {field}')
        for field in ['reference_coverage','test_coverage']:
            if not qa.MIN_COVERAGE<=float(row[field])<=1:raise ValueError('Invalid pair coverage')
        if not qa.MIN_COVERAGE*qa.SAMPLES<=float(row['image_valid_samples'])<=qa.SAMPLES:
            raise ValueError('Invalid image QA coverage')
    return {tuple(row[k] for k in PAIR_KEY):row for row in table.to_dict('records')}


def map_group(points,mode,files,mapping,exe,reference_image,test_image,load_image,max_roundtrip):
    """Map unique reference scanner-RAS peaks; reject any required field-domain miss."""
    points=np.asarray(points,float)
    results=[dict(mapping_status='POINT_DOMAIN_FAIL',reason='Source point outside reference MRI interior',roundtrip_mm=None,mapped=None) for _ in points]
    valid=inside(points,reference_image)
    terms=mapping.terms(files)
    prefix='aff' if mode=='affine' else 'syn'
    if mode=='syn' and valid.any():
        indices=np.flatnonzero(valid)
        intermediate=mapping.point_map(exe,points[valid],[f'[{files["syn_aff"]},1]'])
        good=inside(intermediate,load_image(files['invwarp']))
        for i in indices[~good]:results[i]['reason']='Inverse-affine point outside inverse-field domain'
        valid[indices[~good]]=False
    indices=np.flatnonzero(valid)
    if not len(indices):return results
    mapped=mapping.point_map(exe,points[valid],terms[f'{prefix}_r2t'])
    in_test=inside(mapped,test_image)
    if mode=='syn':in_test &= inside(mapped,load_image(files['warp']))
    for i in indices[~in_test]:results[i]['reason']='Mapped point outside test MRI/forward-field interior'
    selected=indices[in_test]
    if not len(selected):return results
    accepted=mapped[in_test]
    returned=mapping.point_map(exe,accepted,terms[f'{prefix}_t2r'])
    errors=np.linalg.norm(returned-points[selected],axis=1)
    back_inside=inside(returned,reference_image)
    for i,point,error,domain_ok in zip(selected,accepted,errors,back_inside):
        if not domain_ok or not np.isfinite(error):
            results[i]=dict(mapping_status='POINT_DOMAIN_FAIL',reason='Returned point outside reference MRI/invalid',roundtrip_mm=float(error) if np.isfinite(error) else None,mapped=None)
        elif error>max_roundtrip:
            results[i]=dict(mapping_status='POINT_ROUNDTRIP_FAIL',reason=f'Source-point roundtrip {error:.9g} mm exceeds {max_roundtrip}',roundtrip_mm=float(error),mapped=None)
        else:results[i]=dict(mapping_status='PASS_MAPPED',reason='',roundtrip_mm=float(error),mapped=np.asarray(point,float))
    return results


def evaluate_records(data,specs,truth,bridges,pair_qa,qa,mapping,exe,resource_root,load_image):
    rows=[];raw=[];groups={}
    for label,table in data.items():
        strategy,scenario=specs[label][:2]
        for original in table.to_dict('records'):
            gt=truth[(original['test_subject'],original['run'])]
            if original.get('stimulation_pair')!=gt['stimulation_pair'] or original.get('hemisphere')!=gt['hemisphere']:
                raise ValueError('Recorded stimulation metadata differs from released ground truth')
            passed=original['status']=='PASS'
            head=np.array([float(original[f'estimated_head_{a}_mm']) for a in 'xyz']) if passed else None
            true_head=np.asarray(gt['head_mm'],float);true_ras=np.asarray(gt['ras_mm'],float)
            within=original['condition']=='within'
            if not within:
                raw.append(dict(strategy=strategy,scenario=scenario,**{k:original[k] for k in ['method','reference_subject','test_subject','run','condition']},
                    reconstruction_status=original['status'],raw_head_separation_mm=float(np.linalg.norm(head-true_head)) if passed else None,
                    valid_localization_error=False,interpretation='UNREGISTERED_HEAD_DIAGNOSTIC_ONLY'))
            for mode in ['within'] if within else ['affine','syn']:
                row=dict(original,strategy=strategy,scenario=scenario,registration_mode=mode,reconstruction_status=original['status'],
                    evaluation_status='RECONSTRUCTION_FAIL',evaluation_failure_reason=original['failure_reason'],
                    mapping_status='NOT_ATTEMPTED_RECONSTRUCTION_FAIL',pair_qa_status='NOT_APPLICABLE' if within else pair_qa[(original['reference_subject'],original['test_subject'],mode)]['status'],
                    localization_error_valid=False,localization_error_mm=None,source_point_roundtrip_mm=None,
                    metric_frame='test_head_mm' if within else 'test_scanner_ras_mm',
                    interpretation='WITHIN_TEST_HEAD' if within else 'REGISTRATION_DEPENDENT_NUMERIC_QA_FILTERED',
                    original_anatomical_mapping_status=original.get('anatomical_mapping_status',''),
                    anatomical_mapping_status='NOT_APPLICABLE_WITHIN' if within else 'NOT_MAPPED')
                for a,j in zip('xyz',range(3)):
                    row[f'gt_test_head_{a}_mm']=float(true_head[j]);row[f'gt_test_ras_{a}_mm']=float(true_ras[j])
                    row[f'mapped_test_ras_{a}_mm']=None;row[f'mapped_test_surface_{a}_mm']=None
                rows.append(row)
                if not passed:continue
                if within:
                    row.update(evaluation_status='PASS_WITHIN',evaluation_failure_reason='',mapping_status='WITHIN_NATIVE',
                               localization_error_valid=True,localization_error_mm=float(np.linalg.norm(head-true_head)))
                    continue
                pair=pair_qa[(original['reference_subject'],original['test_subject'],mode)]
                if pair['status']!='PASS_NUMERICAL_MAPPING_QA':
                    row.update(evaluation_status='PAIR_QA_FAIL',evaluation_failure_reason=pair['reason'],mapping_status='NOT_ATTEMPTED_QA_FAIL')
                    continue
                scanner=apply(bridges[original['reference_subject']]['head_to_scanner_ras_mm'],head.reshape(1,3))[0]
                row['_source_scanner']=scanner
                groups.setdefault((original['reference_subject'],original['test_subject'],mode),[]).append(row)
    for (reference,test,mode),group in sorted(groups.items()):
        print(f'Mapping saved native peaks: {reference} -> {test} {mode} ({len(group)} records)',flush=True)
        points=np.stack([row['_source_scanner'] for row in group])
        unique,inverse=np.unique(points,axis=0,return_inverse=True)
        files=qa.files_for_pair(resource_root,reference,test)
        try:
            mapped=map_group(unique,mode,files,mapping,exe,
                load_image(Path(pair_qa[(reference,test,mode)]['moving_mri'])),
                load_image(Path(pair_qa[(reference,test,mode)]['fixed_mri'])),load_image,qa.MAX_ROUNDTRIP_MM)
        except Exception as exc:
            mapped=[dict(mapping_status='MAPPING_EXECUTION_FAIL',reason=f'{type(exc).__name__}: {exc}',roundtrip_mm=None,mapped=None) for _ in unique]
        for row,index in zip(group,inverse):
            item=mapped[index];row.pop('_source_scanner')
            row.update(mapping_status=item['mapping_status'],evaluation_status=item['mapping_status'],
                       evaluation_failure_reason=item['reason'],source_point_roundtrip_mm=item['roundtrip_mm'])
            if item['mapping_status']=='PASS_MAPPED':
                point=item['mapped'];target=np.array([row[f'gt_test_ras_{a}_mm'] for a in 'xyz'])
                surface=apply(bridges[test]['scanner_ras_to_surface_mm'],point.reshape(1,3))[0]
                row.update(localization_error_valid=True,localization_error_mm=float(np.linalg.norm(point-target)),
                           anatomical_mapping_status='MAPPED_NUMERICAL_QA_PASS')
                for a,j in zip('xyz',range(3)):
                    row[f'mapped_test_ras_{a}_mm']=float(point[j]);row[f'mapped_test_surface_{a}_mm']=float(surface[j])
    return pd.DataFrame(rows),pd.DataFrame(raw)


def summaries_and_matches(rows,key):
    fields=['scenario','strategy','method','condition','registration_mode']
    summary=[]
    for values,group in rows.groupby(fields,sort=True):
        errors=pd.to_numeric(group.loc[group.localization_error_valid,'localization_error_mm'],errors='raise')
        summary.append(dict(zip(fields,values),scheduled=len(group),reconstruction_passed=int(group.reconstruction_status.eq('PASS').sum()),
            reconstruction_failed=int(group.reconstruction_status.eq('FAIL').sum()),valid_localizations=len(errors),
            pair_qa_excluded=int(group.evaluation_status.eq('PAIR_QA_FAIL').sum()),
            point_qa_failed=int(group.evaluation_status.isin(['POINT_DOMAIN_FAIL','POINT_ROUNDTRIP_FAIL']).sum()),
            mapping_execution_failed=int(group.evaluation_status.eq('MAPPING_EXECUTION_FAIL').sum()),
            median_error_mm=errors.median() if len(errors) else None,mean_error_mm=errors.mean() if len(errors) else None,
            sd_error_mm=errors.std(ddof=1) if len(errors)>1 else None,q25_error_mm=errors.quantile(.25) if len(errors) else None,
            q75_error_mm=errors.quantile(.75) if len(errors) else None))
    for row in summary:
        accounted=row['reconstruction_failed']+row['valid_localizations']+row['pair_qa_excluded']+row['point_qa_failed']+row['mapping_execution_failed']
        if accounted!=row['scheduled']:raise ValueError('Evaluation categories do not cover all scheduled rows')
    match_keys=['strategy','scenario',*key]
    cross=rows[rows.condition.eq('cross_reference')]
    aff=cross[cross.registration_mode.eq('affine')&cross.localization_error_valid]
    syn=cross[cross.registration_mode.eq('syn')&cross.localization_error_valid]
    if aff.duplicated(match_keys).any() or syn.duplicated(match_keys).any():raise ValueError('Duplicate matched-mode keys')
    matches=aff[match_keys+['localization_error_mm']].merge(syn[match_keys+['localization_error_mm']],
        on=match_keys,how='inner',suffixes=('_affine','_syn'),validate='one_to_one')
    matches['syn_minus_affine_mm']=matches.localization_error_mm_syn-matches.localization_error_mm_affine
    return pd.DataFrame(summary),matches


def run(args,base):
    if args.output_root.exists() or args.output_root.is_symlink():raise FileExistsError('Output exists; choose a NEW --output-root')
    protected=[args.preparation_root,args.resource_root,args.bridge_root,args.qa_root,EXP/'source',EXP/'scripts',ROOT/'cross_subject',ROOT/'cross_subject_no_warp']
    protected += [args.input_root/s[2] for s in base.SPECS.values()]
    protected += [args.input_root/n for n in ['02_select_loso_parameters','04_single_reference_parameters','03_05_native_completion_audit']]
    for path in protected:
        if args.output_root==path or path in args.output_root.parents or args.output_root in path.parents:raise ValueError(f'Output overlaps protected input: {path}')
    ledger={}
    def track(path,expected=None):
        path=Path(path).resolve();value=base.sha(path)
        if expected is not None and value!=expected:raise ValueError(f'Input checksum differs: {path}')
        ledger[str(path)]=dict(path=str(path),sha256=value,size_bytes=path.stat().st_size)
    source=EXP/'source/02_parameter_grids';code=EXP/'source/03_solver_code'
    shared=module(EXP/'scripts/02_select_loso_parameters.py','mapped_selection_checks')
    _,doc=shared.snapshot(type('Args',(),{'source_root':source})())
    shared.verify_preparation(args.preparation_root,source,doc)
    _,cases=shared.load_plan(source,doc)
    qa=module(EXP/'scripts/06_validate_registration_resources.py','mapped_registration_checks')
    mapping,mapping_path=qa.historical_mapping()
    qa_manifest_path=args.qa_root/'05_registration_qa_manifest.json';qa_manifest=json.loads(qa_manifest_path.read_text())
    qa_table_path=args.qa_root/'01_pair_mode_qa.csv';qa_table=base.read(qa_table_path)
    pair_qa=validate_pair_qa(qa_table,qa_manifest,qa)
    bridge_path=args.bridge_root/'02_native_physical_bridges.json';bridges=json.loads(bridge_path.read_text())
    bridge_manifest_path=args.bridge_root/'05_bridge_manifest.json';bridge_manifest=json.loads(bridge_manifest_path.read_text())
    if bridge_manifest.get('status')!='ALL_NATIVE_BRIDGES_PASS' or bridge_manifest.get('subjects_passed')!=7 or set(bridges)!=set(SUBJECTS):
        raise ValueError('All native bridges must pass')
    for subject,bridge in bridges.items():
        if bridge.get('input_units')!='mm' or bridge.get('output_units')!='mm':raise ValueError('Bridge units differ')
        for field in MATRICES:
            m=np.asarray(bridge[field],float)
            if m.shape!=(4,4) or not np.isfinite(m).all() or not np.allclose(m[3],[0,0,0,1],atol=1e-8,rtol=0):raise ValueError('Invalid native bridge')
            r=m[:3,:3]
            if not np.allclose(r.T@r,np.eye(3),atol=1e-5,rtol=0) or abs(np.linalg.det(r)-1)>1e-5:raise ValueError('Bridge is not rigid')
        for forward,back in [('head_to_scanner_ras_mm','scanner_ras_to_head_mm'),('surface_to_scanner_ras_mm','scanner_ras_to_surface_mm')]:
            if not np.allclose(np.asarray(bridge[forward])@np.asarray(bridge[back]),np.eye(4),atol=1e-6,rtol=0):raise ValueError('Bridge inverse differs')
    qa_provenance_path=args.qa_root/'04_source_provenance.json'
    prior={str(Path(item['path']).resolve()):item for item in json.loads(qa_provenance_path.read_text())}
    for path in [bridge_path,bridge_manifest_path,mapping_path,EXP/'scripts/06_validate_registration_resources.py']:
        entry=prior.get(str(path.resolve()))
        if entry is None:raise ValueError(f'QA provenance missing input: {path}')
        track(path,entry['sha256'])
    print('Verifying current registration QA source checksums (large resources)...',flush=True)
    for i,(path,item) in enumerate(prior.items(),1):
        track(path,item['sha256'])
        if i%100==0:print(f'  Verified {i}/{len(prior)} source files',flush=True)
    exe=qa.locate_ants(Path(qa_manifest['ants_executable']))
    track(exe,qa_manifest['ants_executable_sha256'])
    for reference in SUBJECTS:
        for test in SUBJECTS:
            if reference==test:continue
            for path in qa.files_for_pair(args.resource_root,reference,test).values():
                if str(path.resolve()) not in prior:raise ValueError('Registration resources differ from QA root')
    source_doc=args.preparation_root/'10_source_provenance.json';track(source_doc)
    dataset=Path(json.loads(source_doc.read_text())['locations']['dataset']).resolve()
    if args.output_root==dataset or dataset in args.output_root.parents or args.output_root in dataset.parents:raise ValueError('Output overlaps dataset')
    configs={}
    for strategy,folder,filename,hashfield,subject in [('LOSO','02_select_loso_parameters','01_loso_selected_parameters.csv','selected_parameters_sha256','test_subject'),
        ('Single-reference','04_single_reference_parameters','01_single_reference_selected_parameters.csv','selected_csv_sha256','reference_subject')]:
        path=args.input_root/folder/filename;state_path=args.input_root/folder/'07_selection_manifest.json'
        state=json.loads(state_path.read_text());track(state_path);track(path,state[hashfield])
        selection=base.read(path)
        if state.get('actual_selections')!=84 or state.get('reliability')!='primary-all-pass' or state.get('source_manifest_sha256')!=base.sha(source/'source_manifest.json'):
            raise ValueError('Evaluation requires complete strict primary selections from the pinned grids')
        if state.get('status')!='COMPLETE' or len(selection)!=84 or selection.duplicated(['method',subject]).any():raise ValueError('Selection incomplete')
        configs[strategy]={(r['method'],r[subject]):r for r in selection.to_dict('records')}
    completion_path=args.input_root/'03_05_native_completion_audit/05_completion_manifest.json'
    completion=json.loads(completion_path.read_text());track(completion_path)
    if completion.get('status')!='NATIVE_ATTEMPT_DATA_COMPLETE':raise ValueError('Native completion not passed')
    data={};checks=[]
    for label,(strategy,scenario,folder,total,passing) in base.SPECS.items():
        path=args.input_root/folder/'01_execution_status.csv';report=args.input_root/folder/'05_validation_report.json'
        detail=json.loads(report.read_text());track(report);track(path,detail['completed_csv_sha256'])
        if base.sha(path)!=completion['datasets'][folder]['completed_csv_sha256']:raise ValueError('Completion checksum differs')
        data[label]=base.read(path)
        checks.append(base.validate_table(data[label],base.expected_keys(cases,shared.METHODS,strategy),label,configs[strategy],strategy,total,passing))
    checks.append(base.validate_variant(data['loso_primary'],data['loso_sensitivity'],1,'LOSO'))
    checks.append(base.validate_variant(data['single_primary'],data['single_sensitivity'],3,'Single-reference'))
    # Load only the pinned ground-truth metadata functions, never raw EEG or inverse code.
    pinned_source=json.loads((EXP/'source/provenance_manifest.json').read_text())
    for item in pinned_source['files']:
        if item['path'].startswith('source/03_solver_code/'):
            track(EXP/item['path'],item['sha256'])
    pinned=json.loads((code/'code_manifest.json').read_text());metric_path=code/'src/localize_mi/metrics.py'
    entry=next(item for item in pinned['files'] if item['path']=='src/localize_mi/metrics.py')
    track(metric_path,entry['sha256']);metric=module(metric_path,'mapped_ground_truth_metadata')
    native_input_provenance=args.input_root/'03_05_native_completion_audit/03_input_provenance.json'
    native_sources={str(Path(item['path']).resolve()):item for item in json.loads(native_input_provenance.read_text())}
    track(native_input_provenance)
    truth={};coordinate_rows=[]
    for case in cases.to_dict('records'):
        subject=case['test_subject'];run=case['run'];directory=dataset/'derivatives/epochs'/subject
        meta_path=directory/'eeg'/f'{subject}_task-seegstim_{run}_epochs.json'
        contact_path=directory/'ieeg'/f'{subject}_task-seegstim_space-surface_electrodes.tsv'
        for path in [meta_path,contact_path]:
            expected=native_sources.get(str(path.resolve()))
            if expected is None:raise ValueError(f'Ground truth input lacks native completion provenance: {path}')
            track(path,expected['sha256'])
        stim=metric.load_stimulation_info(json.loads(meta_path.read_text())['Description'],contact_path)
        surface=np.asarray(stim.midpoint,float).reshape(3)*1000
        if not np.isfinite(surface).all() or np.linalg.norm(surface)>1000:raise ValueError('Invalid native contact midpoint')
        ras=apply(bridges[subject]['surface_to_scanner_ras_mm'],surface.reshape(1,3))[0]
        head=apply(bridges[subject]['scanner_ras_to_head_mm'],ras.reshape(1,3))[0]
        truth[(subject,run)]=dict(surface_mm=surface,ras_mm=ras,head_mm=head,stimulation_pair=stim.pair,hemisphere=stim.hemisphere)
        row=dict(test_subject=subject,run=run,status='PASS_NATIVE_GEOMETRY',stimulation_pair=stim.pair,hemisphere=stim.hemisphere,
            contact_sha256=base.sha(contact_path),metadata_sha256=base.sha(meta_path))
        for frame,xyz in [('surface',surface),('ras',ras),('head',head)]:
            for axis,j in zip('xyz',range(3)):row[f'gt_test_{frame}_{axis}_mm']=float(xyz[j])
        coordinate_rows.append(row)
    images={}
    def load_image(path):
        path=Path(path).resolve()
        if path not in images:images[path]=nib.load(str(path))
        return images[path]
    rows,diagnostic=evaluate_records(data,base.SPECS,truth,bridges,pair_qa,qa,mapping,str(exe),args.resource_root,load_image)
    evaluation_key=['strategy','scenario',*base.KEY,'registration_mode']
    if rows.duplicated(evaluation_key).any():raise ValueError('Duplicate mapped evaluation rows')
    primary=rows[rows.scenario.eq('primary')].copy();sensitivity=rows[rows.scenario.eq('sensitivity')].copy()
    if len(primary)!=10980 or len(sensitivity)!=10980:raise ValueError('Expected 10,980 evaluation rows per scenario')
    if len(diagnostic)!=10248:raise ValueError('Raw diagnostic coverage differs')
    summary,matches=summaries_and_matches(rows,base.KEY)
    failure=rows.loc[~rows.localization_error_valid,['strategy','scenario',*base.KEY,'registration_mode',
        'reconstruction_status','pair_qa_status','evaluation_status','failure_reason','evaluation_failure_reason']].copy()
    point_audit=rows[rows.condition.eq('cross_reference')][['strategy','scenario',*base.KEY,'registration_mode',
        'reconstruction_status','pair_qa_status','mapping_status','source_point_roundtrip_mm','evaluation_failure_reason']].copy()
    for path in [qa_manifest_path,qa_table_path,qa_provenance_path,Path(__file__),Path(base.__file__),
                 source/'source_manifest.json',code/'code_manifest.json']:track(path)
    print('Rechecking input checksums before publishing evaluation...',flush=True)
    for path,item in ledger.items():
        if base.sha(path)!=item['sha256']:raise ValueError(f'Input changed during evaluation: {path}')
    execution_failures=int(rows.evaluation_status.eq('MAPPING_EXECUTION_FAIL').sum())
    versions={package:importlib.metadata.version(package) for package in ['numpy','pandas','nibabel','scipy','mne']}
    manifest=dict(status='EVALUATION_COMPLETE_WITH_RETAINED_FAILURES' if not execution_failures else 'EVALUATION_HAS_MAPPING_EXECUTION_FAILURES',
        phase='evaluate',analysis_versions=versions,native_inputs_validated=True,native_primary_rows=5856,native_sensitivity_rows=5856,
        primary_evaluation_rows=len(primary),sensitivity_evaluation_rows=len(sensitivity),
        within_rows_per_scenario=732,cross_rows_per_mode_per_scenario=5124,
        registration_modes=['affine','syn'],within_mode='within',
        pair_qa_passed=int(qa_table.status.eq('PASS_NUMERICAL_MAPPING_QA').sum()),pair_qa_failed=int(qa_table.status.eq('FAIL').sum()),
        failed_pair_policy='Retain both scenarios and denominators; never map failed mode/pairs; leave their errors blank.',
        point_qa_policy='Interior MRI and required displacement-field domains; roundtrip <=0.1 mm.',
        source_peak_policy='Frozen native peaks; never reselect after mapping.',ground_truth_policy='Native test contact midpoint transformed only by its own bridge.',
        matched_mode_comparison='Exact key; both affine and SyN localization_error_valid. Coverage reported separately.',
        inverse_reconstruction_executed=False,registration_fitting_executed=False,point_mapping_executed=True,
        localization_errors_calculated=True,anatomical_accuracy_established=False,
        partial_pair_qa_policy_explicit=True,validation_policy='Only passing mode/pairs and passing individual source points contribute errors; all attempts remain in denominators.',
        mapping_execution_failures=execution_failures,evaluation_status_counts=rows.evaluation_status.value_counts().to_dict(),
        limitations=qa_manifest['limitations'],script_sha256=base.sha(base.__file__),implementation_sha256=base.sha(__file__))
    args.output_root.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.mapped_evaluation_',dir=args.output_root.parent) as temp:
        stage=Path(temp)/'results';stage.mkdir()
        pd.DataFrame(checks).to_csv(stage/'01_input_validation.csv',index=False)
        pd.DataFrame(coordinate_rows).to_csv(stage/'02_coordinate_validation.csv',index=False)
        primary.to_csv(stage/'03_primary_localization.csv',index=False)
        sensitivity.to_csv(stage/'04_sensitivity_localization.csv',index=False)
        summary.to_csv(stage/'05_method_summary.csv',index=False)
        failure.to_csv(stage/'06_failure_analysis.csv',index=False)
        (stage/'07_evaluation_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        diagnostic.to_csv(stage/'08_raw_head_diagnostic.csv',index=False)
        qa_table.to_csv(stage/'09_registration_readiness.csv',index=False)
        qa_table[qa_table.status.eq('FAIL')].to_csv(stage/'10_retained_pair_qa_failures.csv',index=False)
        (stage/'11_source_provenance.json').write_text(json.dumps(list(ledger.values()),indent=2)+'\n')
        matches.to_csv(stage/'12_matched_affine_syn.csv',index=False)
        point_audit.to_csv(stage/'13_source_point_mapping_audit.csv',index=False)
        (stage/'README.txt').write_text(__doc__+'\nNative status is preserved; use localization_error_valid/evaluation_status for evaluation.\n')
        if args.output_root.exists():raise FileExistsError(args.output_root)
        stage.rename(args.output_root)
    print('Evaluation written:',args.output_root)
    print(f'Rows: {len(primary)} primary; {len(sensitivity)} sensitivity. Valid errors: {int(rows.localization_error_valid.sum())}.')
    print(rows.evaluation_status.value_counts().to_string())
    print('Pair QA failures and failed reconstructions are retained with blank localization errors.')
    return 2 if execution_failures else 0

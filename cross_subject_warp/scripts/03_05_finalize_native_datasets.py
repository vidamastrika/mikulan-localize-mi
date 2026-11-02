#!/usr/bin/env python3
# Complete/import native LOSO and reference-only records; no inverse fitting.
# From the project root, with .venv active:
# python cross_subject_warp/scripts/03_05_finalize_native_datasets.py \
#   --preparation-root cross_subject_warp/outputs/01_prepare_experiment_review
#
# Fresh audit/output tree for a repeat (existing destinations are refused):
# python cross_subject_warp/scripts/03_05_finalize_native_datasets.py \
#   --preparation-root cross_subject_warp/outputs/01_prepare_experiment_review \
#   --output-base /tmp/localize-native-completion-review
#
# Recreates only EEG referencing/channel selection/fingerprints. No covariance
# estimation, inverse solve, registration, peak selection or anatomical mapping.
# Failed peaks stay blank; primary and sensitivity datasets stay separate.
"""Validate native inputs and import historical attempts with explicit metadata provenance.

Completeness refers to scheduled attempts and input metadata, not universal
solver success. Recomputed metadata is labelled as reconstructed input evidence,
never represented as a newly successful historical solver execution.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
import shutil
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
EXP=ROOT/'cross_subject_warp'
KEY=['method','reference_subject','test_subject','run','condition']
PEAKS=['estimated_head_x_mm','estimated_head_y_mm','estimated_head_z_mm','peak_vertex','peak_time_s']
META=['stimulation_pair','hemisphere','channels','actual_channels_json','nominal_channels_json',
      'replacement_details_json','original_bad_channels_json','reference_channel_scope',
      'eeg_data_sha256','epoch_times_sha256','sampling_frequency_hz','highpass_hz','lowpass_hz',
      'baseline_json','montage_template_subject','n_epochs']
SPECS={
 '03_reference_based_final':('loso','primary','03_reference_based_final/01_execution_status.csv',1464,1463),
 '05_single_reference_final':('reference','primary','05_single_reference_final/01_execution_status.csv',4392,4378),
 '03_gamma_wholebrain_sensitivity':('loso','sensitivity','03_gamma_wholebrain_sensitivity/01_execution_status_wholebrain_variant.csv',1464,1464),
 '05_wholebrain_hemisphere_sensitivity':('reference','sensitivity','05_wholebrain_hemisphere_sensitivity/04_execution_status_variant.csv',4392,4381)}


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda:file.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    obj=importlib.util.module_from_spec(spec);sys.modules[name]=obj;spec.loader.exec_module(obj)
    return obj


def canonical(value):
    """Compare JSON/numeric metadata while retaining the original stored spelling."""
    if isinstance(value,dict):return {k:canonical(v) for k,v in sorted(value.items())}
    if isinstance(value,list):return [canonical(v) for v in value]
    if isinstance(value,str) and value[:1] in ['{','[']:
        try:return canonical(json.loads(value))
        except ValueError:pass
    if isinstance(value,bool):return str(value).lower()
    try:return format(float(value),'.12g')
    except (ValueError,TypeError):return str(value)


def read(path):
    return pd.read_csv(path,dtype=str,keep_default_na=False)


def same_record(left,right,fields):
    wrong=[f for f in fields if canonical(left[f])!=canonical(right[f])]
    if wrong:raise ValueError(f'Historical lineage differs: {wrong}; {tuple(left.get(k) for k in KEY)}')


def expected_key(job):return tuple(str(job[k]) for k in KEY)


def complete_metadata(row,metadata,evidence):
    """Fill only absent input metadata; never change status, peaks or failure reason."""
    updated=dict(row);changes=[]
    for field,value in metadata.items():
        value=str(value)
        if row.get(field,'')!='':
            if canonical(row[field])!=canonical(value):
                raise ValueError(f'Input metadata mismatch: {field}, {expected_key(row)}')
        else:
            updated[field]=value
            changes.append(dict(**{k:row[k] for k in KEY},field=field,original_value='',
                                completed_value=value,evidence=evidence))
    for field in ['status','failure_reason']+PEAKS:
        if updated.get(field,'')!=row.get(field,''):raise AssertionError('Attempt outcome changed')
    return updated,changes


def verify_variant(primary,variant,expected_overrides,supplement,annotated_whole_brain=False):
    p={expected_key(r):r for r in primary.to_dict('records')}
    v={expected_key(r):r for r in variant.to_dict('records')}
    s={expected_key(r):r for r in supplement.to_dict('records')}
    if set(p)!=set(v):raise ValueError('Sensitivity keys differ from primary')
    changed=[]
    fields=sorted((set(primary)&set(variant)) - {'record_source'})
    for key,row in v.items():
        differences=[f for f in fields if canonical(row[f])!=canonical(p[key][f])]
        if differences:
            if p[key]['status']!='FAIL' or row['status']!='PASS' or key not in s:
                raise ValueError(f'Unexpected sensitivity replacement: {key}')
            excluded={'record_source','peak_selection_policy','sensitivity_override'}
            if annotated_whole_brain:
                if row.get('hemisphere_constraint_applied','').lower()!='false' or row.get('selection_criterion')!='maximum_absolute_amplitude_whole_brain':
                    raise ValueError('Invalid whole-brain sensitivity policy annotation')
                excluded.update(['hemisphere_constraint_applied','selection_criterion'])
            common=sorted((set(row)&set(s[key]))-excluded)
            same_record(row,s[key],common)
            changed.append(key)
    if len(changed)!=expected_overrides:raise ValueError(f'Expected {expected_overrides} sensitivity replacements, got {len(changed)}')
    return changed


def recompute_inputs(engine,test,run,requests):
    """Same frozen loader/reference/selection, without Engine.prepare's covariance fit."""
    loaded=engine.load_run(engine.args.dataset,test,task='seegstim',run=run)
    referenced=engine.inverse.apply_average_reference(loaded.epochs)
    good=[c for c in loaded.epochs.ch_names if c not in loaded.epochs.info['bads']]
    stim=engine.load_stim(loaded.metadata['Description'],engine.args.dataset/'derivatives/epochs'/test/'ieeg'/f'{test}_task-seegstim_space-surface_electrodes.tsv')
    results={};fingerprints={}
    for reference,montage in sorted(requests):
        if reference not in engine.nominal:
            engine.nominal[reference]=engine.grid09.fixed_spatial_subsets(engine.forward(reference))
        names,xyz,nominal=engine.nominal[reference]
        if montage=='all_good':actual=good;replacement=[]
        else:actual,replacement=engine.grid09.replace_bad_nominal(nominal[montage],names,xyz,good)
        selected=referenced.copy().pick(actual)
        if selected.info['bads']:raise ValueError('Bad channel survived selection')
        channel_key=tuple(actual)
        if channel_key not in fingerprints:
            values=np.ascontiguousarray(selected.get_data(copy=True),dtype=np.float64)
            fingerprints[channel_key]=(hashlib.sha256(values.tobytes()).hexdigest(),
                hashlib.sha256(np.ascontiguousarray(selected.times).tobytes()).hexdigest())
        data_hash,time_hash=fingerprints[channel_key]
        results[(reference,montage)]=dict(stimulation_pair=str(stim.pair),hemisphere=str(stim.hemisphere),
            channels=str(len(actual)),actual_channels_json=json.dumps(actual),
            nominal_channels_json=json.dumps(good if montage=='all_good' else nominal[montage]),
            replacement_details_json=json.dumps(replacement,sort_keys=True),
            original_bad_channels_json=json.dumps(list(loaded.epochs.info['bads'])),
            reference_channel_scope='all loaded channels; MNE average-ref excludes marked bads',
            eeg_data_sha256=data_hash,epoch_times_sha256=time_hash,
            sampling_frequency_hz=str(float(selected.info['sfreq'])),highpass_hz=str(float(selected.info['highpass'])),
            lowpass_hz=str(float(selected.info['lowpass'])),baseline_json=json.dumps(selected.baseline),
            montage_template_subject=reference,n_epochs=str(len(selected)))
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preparation-root',type=Path,default=EXP/'outputs/01_prepare_experiment_review')
    parser.add_argument('--historical-root',type=Path,default=ROOT/'cross_subject_no_warp/outputs')
    parser.add_argument('--output-base',type=Path,default=EXP/'outputs')
    args=parser.parse_args()
    for key,value in vars(args).items():setattr(args,key,value.expanduser().resolve())
    destinations=[args.output_base/name for name in [*SPECS,'03_05_native_completion_audit']]
    for path in destinations:
        if path.exists() or path.is_symlink():raise FileExistsError(f'Existing destination: {path}')
        for protected in [args.historical_root,args.preparation_root,EXP/'source',EXP/'scripts']:
            if path==protected or protected in path.parents or path in protected.parents:
                raise ValueError(f'Output overlaps input: {protected}')
    source=EXP/'source/02_parameter_grids';code=EXP/'source/03_solver_code'
    solver=module(EXP/'scripts/03_run_localization.py','native_completion_solver03')
    single=module(EXP/'scripts/05_run_single_reference.py','native_completion_single05')
    shared=module(EXP/'scripts/02_select_loso_parameters.py','native_completion_selection02')
    shared.snapshot(SimpleNamespace(source_root=source))
    solver.snapshot_code(ROOT,code)
    preparation=json.loads((args.preparation_root/'10_source_provenance.json').read_text())
    dataset=Path(preparation['locations']['dataset'])
    for path in destinations:
        if path==dataset or dataset in path.parents or path in dataset.parents:
            raise ValueError('Output overlaps dataset')
    a=SimpleNamespace(selection_root=EXP/'outputs/02_select_loso_parameters',grid_source_root=source,
        preparation_root=args.preparation_root,dataset=dataset,code_source_root=code,
        bem_root=ROOT/'outputs/12_additional_methods_all_runs/bem',montage_template='reference',quiet=True,
        source_root=source,script02=EXP/'scripts/02_select_loso_parameters.py',
        script04=EXP/'scripts/04_select_single_reference_parameters.py')
    selected,cases,_,doc,state=solver.load_inputs(a)
    shared.verify_preparation(args.preparation_root,source,doc)
    configs_loso,errors=solver.enrich_configs(selected,source,doc)
    if errors:raise ValueError(errors)
    loso_jobs=solver.make_jobs(selected,cases)
    a.selection_root=EXP/'outputs/04_single_reference_parameters'
    ref_selected,schedule,_,reference_state=single.verify_selection(a,solver)
    configs_ref,errors=single.resolve_configurations(ref_selected,solver,source,doc)
    if errors:raise ValueError(errors)
    reference_jobs=single.build_jobs(ref_selected,schedule)
    solver.check_audit(a,source,doc)
    historical=args.historical_root
    ledger={}
    def track(path):
        path=Path(path).resolve()
        if path not in ledger:ledger[path]=dict(path=str(path),sha256=digest(path),size_bytes=path.stat().st_size)
        return ledger[path]['sha256']
    versions={p:importlib.metadata.version(p) for p in ['mne','numpy','scipy','pandas','sesameeg','invertmeeg']}
    # Verify recorded run/source/selection identities before touching result copies.
    manifests={}
    old_audit=read(historical/'01_prepare_experiment/05_coordinate_audit.csv')
    for row in old_audit.to_dict('records'):
        for field,hashfield in [('forward_path','forward_sha256'),('head_to_surface_path','h5_sha256'),('contacts_path','contacts_sha256')]:
            if track(Path(row[field]))!=row[hashfield]:raise ValueError(f'Historical model input changed: {row[field]}')
    for folder in ['03_reference_based','03_reference_based_patch','03_gamma_wholebrain_sensitivity','05_single_reference_reconstruction']:
        path=historical/folder/'06_frozen_run_manifest.json';track(path)
        m=json.loads(path.read_text());manifests[folder]=m
        if m['versions']!=versions:raise ValueError(f'Software versions differ: {folder}')
        if Path(m['dataset']).resolve()!=dataset.resolve():raise ValueError('Historical dataset path differs')
        if folder.startswith('03'):
            if m['selection_sha256']!=digest(EXP/'outputs/02_select_loso_parameters/01_loso_selected_parameters.csv'):raise ValueError('LOSO selection differs')
            if m['code_manifest_sha256']!=track(code/'code_manifest.json'):raise ValueError('Solver snapshot differs')
            if m['coordinate_audit_sha256']!=track(historical/'01_prepare_experiment/05_coordinate_audit.csv'):raise ValueError('Historical coordinate audit differs')
            if m['nominal_montage']!='reference':raise ValueError('Montage policy differs')
        else:
            if m['selected_csv_sha256']!=digest(a.selection_root/'01_single_reference_selected_parameters.csv'):raise ValueError('Reference selection differs')
            if m['schedule_sha256']!=digest(a.selection_root/'09_reference_test_schedule.csv'):raise ValueError('Reference schedule differs')
            if m['montage_template']!='reference':raise ValueError('Montage policy differs')
    tables={};source_paths={};lineage=[]
    for name,(strategy,scenario,relative,total,passing) in SPECS.items():
        path=historical/relative;track(path);table=read(path)
        table['source_condition']=table.condition
        table['condition']=table.condition.replace({'single_reference_no_warp':'cross_reference'})
        if table.duplicated(KEY).any():raise ValueError(f'Duplicate keys: {name}')
        jobs=loso_jobs if strategy=='loso' else reference_jobs
        expected={expected_key(j):j for j in jobs}
        if len(table)!=total or set(map(tuple,table[KEY].values))!=set(expected):raise ValueError(f'Coverage differs: {name}')
        if table.status.eq('PASS').sum()!=passing or not table.status.isin(['PASS','FAIL']).all():raise ValueError(f'Status coverage differs: {name}')
        configs=configs_loso if strategy=='loso' else configs_ref
        for row in table.to_dict('records'):
            job=expected[expected_key(row)]
            for field in ['fold_id','model_subject','configuration_key']:
                if row[field]!=job[field]:raise ValueError(f'Job identity mismatch: {field}')
            cfg=configs[(row['method'],row['test_subject'] if strategy=='loso' else row['reference_subject'])]['parameters']
            if canonical(json.loads(row['configuration_parameters']))!=canonical(cfg):raise ValueError('Effective configuration differs')
            if row['status']=='FAIL':
                if not row['failure_reason'] or any(row.get(f,'') for f in PEAKS):raise ValueError('Invalid failed attempt')
            else:
                if row['coordinate_frame']!='model_head' or row['coordinate_units']!='mm':raise ValueError('Invalid peak frame')
                point=np.array([float(row[f]) for f in PEAKS[:3]])
                if not np.isfinite(point).all() or np.linalg.norm(point)>1000:raise ValueError('Invalid native peak')
                if row['method']=='Continuous-ECD':
                    if track(Path(row['bem_path']))!=row['bem_sha256']:raise ValueError('BEM hash mismatch')
        tables[name]=table;source_paths[name]=path
    # Every retained primary attempt must trace to its actual source attempt.
    original=read(historical/'03_reference_based/01_execution_status.csv')
    retry=read(historical/'03_reference_based_patch/01_execution_status.csv')
    original_path=historical/'03_reference_based/01_execution_status.csv'
    retry_path=historical/'03_reference_based_patch/01_execution_status.csv'
    if manifests['03_reference_based_patch']['retry_source_sha256']!=track(original_path):raise ValueError('Retry lineage changed')
    failed_keys=set(map(tuple,original.loc[original.status.eq('FAIL'),KEY].values))
    if set(map(tuple,retry[KEY].values))!=failed_keys:raise ValueError('Retry coverage differs from original failures')
    originals={tuple(r[k] for k in KEY):r for r in original.to_dict('records')}
    retries={tuple(r[k] for k in KEY):r for r in retry.to_dict('records')}
    for row in read(source_paths['03_reference_based_final']).to_dict('records'):
        key=tuple(row[k] for k in KEY);origin=originals[key]
        raw=retries[key] if origin['status']=='FAIL' else origin
        path=retry_path if origin['status']=='FAIL' else original_path
        same_record(row,raw,sorted((set(row)&set(raw))-{'record_source'}))
        lineage.append(dict(strategy='loso',method=row['method'],reference_subject=row['reference_subject'],
                            test_subject=row['test_subject'],run=row['run'],source_condition=row['condition'],
                            original_attempt_path=str(path),original_attempt_sha256=track(path)))
    reference_raw=historical/'05_single_reference_reconstruction/01_execution_status.csv'
    raw=read(reference_raw).set_index(KEY)
    for row in read(source_paths['05_single_reference_final']).to_dict('records'):
        record=raw.loc[tuple(row[k] for k in KEY)].to_dict()
        fields=sorted((set(row)&set(record))-{'record_source'})
        same_record(row,record,fields)
        lineage.append(dict(strategy='reference',**{k:row[k] for k in KEY[:-1]},source_condition=row['condition'],
                            original_attempt_path=str(reference_raw),original_attempt_sha256=track(reference_raw)))
    def normalized(path):
        track(path);t=read(path);t['condition']=t.condition.replace({'single_reference_no_warp':'cross_reference'});return t
    variant_provenance=historical/'03_gamma_wholebrain_sensitivity/14_variant_provenance.json'
    track(variant_provenance);provenance_doc=json.loads(variant_provenance.read_text())
    if provenance_doc['primary_sha256']!=track(source_paths['03_reference_based_final']):raise ValueError('LOSO variant primary hash mismatch')
    if provenance_doc['supplement_sha256']!=track(historical/'03_gamma_wholebrain_sensitivity/01_execution_status.csv'):raise ValueError('LOSO variant supplement hash mismatch')
    if provenance_doc['variant_sha256']!=track(source_paths['03_gamma_wholebrain_sensitivity']):raise ValueError('LOSO variant hash mismatch')
    overrides_loso=verify_variant(tables['03_reference_based_final'],tables['03_gamma_wholebrain_sensitivity'],1,
        normalized(historical/'03_gamma_wholebrain_sensitivity/01_execution_status.csv'),annotated_whole_brain=True)
    overrides_ref=verify_variant(tables['05_single_reference_final'],tables['05_wholebrain_hemisphere_sensitivity'],3,
        normalized(historical/'05_wholebrain_hemisphere_sensitivity/02_retry_status.csv'))
    # A failed attempt's covariance-selection metadata can be recovered only from
    # unanimous successful attempts with the identical independently verified
    # preparation recipe. The numerical inverse outcome is never borrowed.
    donors={}
    for name,table in tables.items():
        for row in table.loc[table.status.eq('PASS')].to_dict('records'):
            montage=str(json.loads(row['configuration_parameters'])['montage'])
            group=(row['test_subject'],row['run'],row['reference_subject'],montage)
            donors.setdefault(group,[]).append((name,row))
    changes=[];checks=[];completed={};native_groups={}
    engine=solver.Engine(a)
    for subject in [f'sub-{i:02d}' for i in range(1,8)]:engine.forward(subject)
    def cached_forward(path,*args,**kwargs):
        subject=Path(path).name.split('_')[0]
        return engine.cache[subject]
    # Fail closed if any future edit tries to invoke the inverse entry point.
    with patch.object(solver.Engine,'solve',side_effect=AssertionError('Inverse fitting forbidden during completion')), \
         patch.object(solver.Engine,'prepare',side_effect=AssertionError('Covariance preparation forbidden during completion')), \
         patch.object(engine.mne,'read_forward_solution',side_effect=cached_forward):
        for test,run in sorted({(r.test_subject,r.run) for r in tables['03_reference_based_final'].itertuples()}):
            print(f'Validating released EEG inputs: {test}/{run}',flush=True)
            run_rows=[]
            for name,table in tables.items():
                for index,row in table.loc[table.test_subject.eq(test)&table.run.eq(run)].iterrows():
                    run_rows.append((name,index,row.to_dict()))
            requests={(r['reference_subject'],str(json.loads(r['configuration_parameters'])['montage'])) for _,_,r in run_rows}
            recomputed=recompute_inputs(engine,test,run,requests)
            eeg_dir=dataset/'derivatives/epochs'/test/'eeg';base=f'{test}_task-seegstim_{run}'
            for suffix in ['epochs.npy','epochs.json','epochs.tsv','channels.tsv']:track(eeg_dir/f'{base}_{suffix}')
            for name,index,row in run_rows:
                montage=str(json.loads(row['configuration_parameters'])['montage'])
                group=(test,run,row['reference_subject'],montage)
                meta=recomputed[(row['reference_subject'],montage)]
                evidence='RECOMPUTED_RELEASED_EEG_FROZEN_REFERENCE_AND_CHANNEL_SELECTION'
                updated,recovered=complete_metadata(row,meta,evidence)
                matches=donors[group]
                cov_values={r['covariance_method_selected'] for _,r in matches if r.get('covariance_method_selected','')}
                if len(cov_values)!=1:raise ValueError(f'Ambiguous covariance metadata: {group}')
                if not row.get('covariance_method_selected',''):
                    updated['covariance_method_selected']=next(iter(cov_values))
                    recovered.append(dict(**{k:row[k] for k in KEY},field='covariance_method_selected',original_value='',
                        completed_value=updated['covariance_method_selected'],evidence='UNANIMOUS_MATCHING_PREPARATION_DONORS'))
                for item in recovered:item.update(dataset=name,source_csv_sha256=track(source_paths[name]))
                changes.extend(recovered)
                updated.update(source_csv=str(source_paths[name]),source_csv_sha256=track(source_paths[name]),
                    native_input_validation='PASS_RECOMPUTED',metadata_completed=bool(recovered),
                    metadata_completion_source=evidence if recovered else 'ORIGINAL_RECORD_VERIFIED',
                    anatomical_mapping_status='NOT_MAPPED_PHYSICAL_BRIDGE_UNRESOLVED')
                completed.setdefault(name,[]).append(updated)
                checks.append(dict(dataset=name,**{k:row[k] for k in KEY},status=row['status'],
                    eeg_fingerprint_matches=True,time_fingerprint_matches=True,channel_selection_matches=True,
                    fields_completed=len(recovered)))
                native_groups[group]=dict(test_subject=test,run=run,reference_subject=row['reference_subject'],montage=montage,
                    eeg_data_sha256=meta['eeg_data_sha256'],epoch_times_sha256=meta['epoch_times_sha256'],
                    covariance_metadata_donors=[dict(dataset=n,method=r['method'],source_csv_sha256=track(source_paths[n])) for n,r in matches])
    # Snapshot all supporting historical audit/config/manifest evidence verbatim.
    evidence_files=[]
    for folder in ['01_prepare_experiment','02_select_loso_parameters','03_reference_based','03_reference_based_patch',
                   '03_reference_based_final','03_gamma_wholebrain_sensitivity','04_single_reference_parameters',
                   '05_single_reference_reconstruction','05_single_reference_final','05_wholebrain_hemisphere_sensitivity']:
        for path in sorted((historical/folder).iterdir()):
            if path.is_file() and not path.name.startswith('.') and path.suffix in ['.json','.csv','.txt','.log']:
                evidence_files.append(path);track(path)
    for path in [EXP/'outputs/02_select_loso_parameters/01_loso_selected_parameters.csv',
                 a.selection_root/'01_single_reference_selected_parameters.csv',a.selection_root/'09_reference_test_schedule.csv',
                 args.preparation_root/'07_experiment_manifest.json',args.preparation_root/'10_source_provenance.json',
                 code/'code_manifest.json',source/'source_manifest.json',Path(__file__)]:track(path)
    for item in json.loads((code/'code_manifest.json').read_text())['files']:
        track(code/item['path'])
    for item in doc['files']:track(source/item['copy'])
    # Check stability again before publishing completed tables.
    for path,item in ledger.items():
        if digest(path)!=item['sha256']:raise ValueError(f'Input changed during completion: {path}')
    args.output_base.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.native_completion_',dir=args.output_base) as temp:
        stage=Path(temp)
        report=dict(status='NATIVE_ATTEMPT_DATA_COMPLETE',inverse_reconstruction_executed=False,
            registration_fitting_executed=False,anatomical_evaluation_ready=False,
            historical_solver_results_recomputed=False,installed_versions_match_recorded=True,
            versions=versions,native_input_groups_verified=len(native_groups),datasets={},
            limitations=['Existing solver peaks imported verbatim, not numerically re-solved.',
                'Input metadata filled after independent released-EEG fingerprint/channel verification.',
                'Covariance method labels for failed rows come from unanimous matching preparation donors; covariance was not refitted.',
                'Historical run manifests retain their own original script hashes; exact old patch executable may not be archived.',
                'Native data completeness does not establish the head/surface-to-NIfTI physical RAS bridge or registration QA.'])
        for name,records in completed.items():
            destination=stage/name;destination.mkdir()
            frame=pd.DataFrame(records).sort_values(KEY).reset_index(drop=True)
            frame.to_csv(destination/'01_execution_status.csv',index=False)
            frame.groupby(['method','condition','status']).size().reset_index(name='count').to_csv(destination/'02_method_condition_summary.csv',index=False)
            pd.DataFrame([r for r in checks if r['dataset']==name]).to_csv(destination/'03_native_input_audit.csv',index=False)
            pd.DataFrame([r for r in changes if r['dataset']==name],columns=['dataset',*KEY,'field','original_value','completed_value','evidence','source_csv_sha256']).to_csv(destination/'04_metadata_completion_provenance.csv',index=False)
            detail=dict(status='NATIVE_ATTEMPT_DATA_COMPLETE',rows=len(frame),pass_count=int(frame.status.eq('PASS').sum()),
                fail_count=int(frame.status.eq('FAIL').sum()),metadata_completed_rows=int(frame.metadata_completed.sum()),
                primary_or_sensitivity=SPECS[name][1],source_csv=str(source_paths[name]),source_sha256=track(source_paths[name]),
                completed_csv_sha256=digest(destination/'01_execution_status.csv'),anatomical_evaluation_ready=False,
                inverse_reconstruction_executed=False,registration_fitting_executed=False)
            (destination/'05_validation_report.json').write_text(json.dumps(detail,indent=2)+'\n')
            (destination/'README.txt').write_text(__doc__+'\nOriginal outcomes and peak values are retained; added input metadata has field-level provenance.\n')
            report['datasets'][name]=detail
        audit=stage/'03_05_native_completion_audit';audit.mkdir();(audit/'historical_evidence').mkdir()
        for path in evidence_files:
            dest=audit/'historical_evidence'/path.relative_to(historical);dest.parent.mkdir(parents=True,exist_ok=True)
            with path.open('rb') as src,dest.open('xb') as dst:
                shutil.copyfileobj(src,dst)
            if digest(dest)!=track(path):raise ValueError('Evidence copy differs')
        pd.DataFrame(lineage).to_csv(audit/'01_attempt_lineage.csv',index=False)
        pd.DataFrame(checks).to_csv(audit/'02_native_input_audit.csv',index=False)
        (audit/'03_input_provenance.json').write_text(json.dumps(list(ledger.values()),indent=2)+'\n')
        (audit/'04_preparation_group_provenance.json').write_text(json.dumps(list(native_groups.values()),indent=2)+'\n')
        (audit/'05_completion_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
        (audit/'06_sensitivity_overrides.json').write_text(json.dumps(dict(loso=[list(k) for k in overrides_loso],single_reference=[list(k) for k in overrides_ref]),indent=2)+'\n')
        for destination in destinations:
            if destination.exists():raise FileExistsError(destination)
        for destination in destinations:(stage/destination.name).rename(destination)
    print(json.dumps(report,indent=2))
    return 0


if __name__=='__main__':
    with tempfile.TemporaryDirectory(prefix='native-completion-config-') as config:
        os.environ['_MNE_FAKE_HOME_DIR']=config
        os.environ['MPLCONFIGDIR']=config
        raise SystemExit(main())

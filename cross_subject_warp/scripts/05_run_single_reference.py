#!/usr/bin/env python3
# SCRIPT 05 — Audit or reconstruct native single-reference source peaks.
# Run from the project root with .venv active. Script 04 must be COMPLETE (84).
#
# NEXT STEP: audit existing primary canonical records; no fitting or import:
# python cross_subject_warp/scripts/05_run_single_reference.py \
#   --phase reuse-audit \
#   --preparation-root cross_subject_warp/outputs/01_prepare_experiment_review
#
# Plan/configuration preflight only; no EEG loading or inverse fitting:
# python cross_subject_warp/scripts/05_run_single_reference.py \
#   --phase plan \
#   --preparation-root cross_subject_warp/outputs/01_prepare_experiment_review
#
# Repeat an audit into a NEW folder, preserving previous reports:
# python cross_subject_warp/scripts/05_run_single_reference.py \
#   --phase reuse-audit \
#   --preparation-root cross_subject_warp/outputs/01_prepare_experiment_review \
#   --output-root cross_subject_warp/outputs/05_single_reference_reuse_audit_review
#
# OPTIONAL NEW FITS, only if equivalent native peaks cannot be reused:
# python cross_subject_warp/scripts/05_run_single_reference.py \
#   --phase reconstruct \
#   --preparation-root cross_subject_warp/outputs/01_prepare_experiment_review \
#   --method MNE --reference-subject sub-01 \
#   --test-subject sub-03 --run run-01 --limit 1 --quiet
# Resume unchanged reconstruction with --phase reconstruct --resume --quiet,
# using the SAME preparation, output root and remaining options.
#
# Help only: python cross_subject_warp/scripts/05_run_single_reference.py --help
# Exit 2 after reuse-audit means mismatched records or missing failure metadata.
# Even exit 0 does not validate reuse or anatomical point mapping.
"""Single-reference native peaks shared by later affine/SyN point evaluation.

84 reference-only configurations from Script 04 x six other subjects' runs:
4,392 cross fits planned, no within baseline. Parameters and montage geometry
come from the reference; EEG and bad-channel flags come from the test subject.
Uses the unchanged patched Script 03 Engine. Average reference precedes channel
selection; cortical peaks retain the known-hemisphere policy.

Default reuse-audit compares existing primary records to the planned identities,
resolved parameters, channel provenance and native peaks. It does not import
records or fit inverses. Metadata checks do not independently reproduce raw EEG
fingerprints or prove fitting-time software/model identity. Preserve failures;
whole-brain sensitivity remains separate. No registration fitting, anatomical
mapping, lead-field warping or localization-error calculation in this stage.

Plan/audit folders must be new. Reconstruction explicitly requires
--phase reconstruct; resume preserves both PASS and FAIL without automatic retry.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback
import importlib.metadata
import pandas as pd
import numpy as np
import re

ROOT=Path(__file__).resolve().parents[2]
EXP=ROOT/'cross_subject_warp'

METHODS=['MNE','dSPM','sLORETA','eLORETA','Continuous-ECD','RAP-MUSIC','LCMV','MxNE','irMxNE','Gamma-MAP','Champagne','SESAME']
KEYS=['method','reference_subject','test_subject','run','condition']

def load_module(path):
    spec=importlib.util.spec_from_file_location('single_reference_solver03',str(path))
    mod=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=mod
    spec.loader.exec_module(mod)
    return mod

def verify_selection(args,solver):
    root=args.selection_root
    sel=root/'01_single_reference_selected_parameters.csv'
    schedule=root/'09_reference_test_schedule.csv'
    manifest=root/'07_selection_manifest.json'
    state=json.loads(manifest.read_text())
    if state.get('status')!='COMPLETE' or state.get('actual_selections')!=84:
        raise ValueError('Script 04 is not COMPLETE with 84 selections')
    for field,path in [('reference_test_schedule_sha256',schedule),('selected_csv_sha256',sel),('source_manifest_sha256',args.source_root/'source_manifest.json')]:
        if solver.digest(path)!=state[field]:raise ValueError(f'Script 04 input checksum mismatch: {path}')
    if solver.digest(args.script02)!=state['script02_sha256'] or solver.digest(args.script04)!=state['script04_sha256']:
        raise ValueError('Selection scripts differ from frozen Script 04 manifest')
    if state.get('intended_reconstruction_montage_template')!='reference subject':
        raise ValueError('Script 04 montage policy mismatch')
    source_doc=json.loads((args.source_root/'source_manifest.json').read_text())
    for item in source_doc['files']:
        if solver.digest(args.source_root/item['copy'])!=item['sha256']:
            raise ValueError(f'Frozen source grid modified: {item["copy"]}')
    preparation=state.get('preparation',{})
    if not preparation.get('plan_matches_frozen'):
        raise ValueError('Script04 manifest lacks verified warp preparation provenance')
    for item in preparation.get('files',[]):
        file=args.preparation_root/Path(item['path']).name
        if solver.digest(file)!=item['sha256']:
            raise ValueError(f'Preparation differs from Script04 inputs: {file}')
    selection=pd.read_csv(sel,dtype=str,keep_default_na=False)
    schedule_df=pd.read_csv(schedule,dtype=str,keep_default_na=False)
    subjects=[f'sub-{i:02d}' for i in range(1,8)]
    combos={(m,r) for m in METHODS for r in subjects}
    if len(selection)!=84 or selection.duplicated(['method','reference_subject']).any() or set(zip(selection.method,selection.reference_subject))!=combos:
        raise ValueError('Incorrect 84 method/reference selections')
    if not selection.selection_status.eq('SELECTED').all():raise ValueError('Selection status not SELECTED')
    if not selection.training_subjects.eq(selection.reference_subject).all():raise ValueError('Reference-only training violated')
    if not selection.configuration_all_pass.str.lower().eq('true').all() or not selection.eligible.str.lower().eq('true').all():
        raise ValueError('Non-eligible configuration selected')
    required=['reference_subject','test_subject','run']
    if len(schedule_df)!=366 or schedule_df.duplicated(required).any():raise ValueError('Expected 366 unique reference/test runs')
    if (schedule_df.reference_subject==schedule_df.test_subject).any():raise ValueError('Self reference in cross schedule')
    if not set(schedule_df.reference_subject).issubset(subjects) or not set(schedule_df.test_subject).issubset(subjects):raise ValueError('Unexpected subject')
    cases=pd.read_csv(args.source_root/source_doc['preparation']['03_single_reference_cases.csv'],dtype=str)
    expected={(ref,row.test_subject,row.run) for ref in subjects for row in cases.itertuples() if ref!=row.test_subject}
    if len(cases)!=61 or set(map(tuple,schedule_df[required].values))!=expected:
        raise ValueError('Schedule differs from 61 runs x six other references')
    return selection,schedule_df,source_doc,state

def resolve_configurations(selection,solver,source,doc):
    # Script 03 enrich_configs() expects method, test_subject, training_subjects.
    # Set its selection key to REFERENCE, because that subject supplied the
    # historical tuning data. No test subject enters parameter enrichment.
    shaped=selection.copy()
    shaped['test_subject']=shaped['reference_subject']
    cfgs,errors=solver.enrich_configs(shaped,source,doc)
    if len(cfgs)+len(errors)!=84:raise ValueError('Unexpected configuration resolution count')
    return cfgs,errors

def build_jobs(selection,schedule):
    cfg_key={(r.method,r.reference_subject):r.configuration_key for r in selection.itertuples()}
    jobs=[]
    for m in METHODS:
        for p in schedule.sort_values(['reference_subject','test_subject','run']).itertuples():
            jobs.append(dict(fold_id=p.reference_subject,reference_subject=p.reference_subject,
                test_subject=p.test_subject,run=p.run,method=m,
                condition='cross_reference',source_condition='single_reference_no_warp',model_subject=p.reference_subject,
                configuration_key=cfg_key[(m,p.reference_subject)],parameter_selection='single_reference'))
    if len(jobs)!=4392 or len({tuple(j[k] for k in KEYS) for j in jobs})!=4392:raise ValueError('Bad reconstruction job coverage')
    return jobs

def channel_audits(frame,solver,out):
    columns=['method','reference_subject','test_subject','run','status','condition','montage_template_subject',
             'channels','actual_channels_json','nominal_channels_json','replacement_details_json',
             'original_bad_channels_json','eeg_data_sha256','epoch_times_sha256',
             'sampling_frequency_hz','n_epochs','configuration_key','failure_reason']
    rows=frame.reindex(columns=columns,fill_value='')
    solver.atomic_csv(out/'09_channel_selection_audit.csv',rows)
    summary=[]
    for r in rows.to_dict('records'):
        def arr(k):
            try:return json.loads(r[k]) if isinstance(r[k],str) and r[k] else []
            except (ValueError,TypeError):return []
        actual=arr('actual_channels_json');bad=arr('original_bad_channels_json');replace=arr('replacement_details_json')
        summary.append(dict(method=r['method'],reference_subject=r['reference_subject'],test_subject=r['test_subject'],
            run=r['run'],status=r['status'],montage_template_subject=r['montage_template_subject'],
            selected_channels=len(actual),bad_channels=len(bad),replacements=len(replace),
            duplicate_selected_channels=len(actual)!=len(set(actual)),
            selected_bad_channels=len(set(actual)&set(bad)),
            eeg_data_sha256=r['eeg_data_sha256'],failure_reason=r['failure_reason']))
    solver.atomic_csv(out/'10_channel_selection_summary.csv',pd.DataFrame(summary))

def audit_reuse(args,solver,full,requested,configs):
    """Record-level audit only: no Engine, raw EEG loading, import or fitting."""
    path=args.canonical_root/'01_execution_status.csv'
    table=pd.read_csv(path,dtype=str,keep_default_na=False)
    metadata=['montage_template_subject','channels','actual_channels_json','nominal_channels_json',
              'original_bad_channels_json','eeg_data_sha256','epoch_times_sha256']
    xyz=['estimated_head_x_mm','estimated_head_y_mm','estimated_head_z_mm']
    required=set(KEYS+metadata+xyz+['fold_id','model_subject','configuration_key',
        'configuration_parameters','status','failure_reason','coordinate_frame','coordinate_units',
        'hemisphere_constraint_applied'])
    if not required.issubset(table):raise ValueError(f'Canonical columns missing: {sorted(required-set(table))}')
    table['source_condition']=table.condition
    table['condition']=table.condition.replace({'single_reference_no_warp':'cross_reference'})
    if table.duplicated(KEYS).any():raise ValueError('Duplicate canonical keys')
    records={tuple(row[k] for k in KEYS):row for row in table.to_dict('records')}
    allowed={tuple(job[k] for k in KEYS) for job in full}
    requested_keys={tuple(job[k] for k in KEYS) for job in requested}
    missing=allowed-set(records);unexpected=set(records)-allowed
    rows=[]
    for job in requested:
        record=records.get(tuple(job[k] for k in KEYS));issues=[];incomplete=[]
        if record is None:issues.append('missing canonical record')
        else:
            for field in ['fold_id','model_subject','configuration_key']:
                if record[field]!=job[field]:issues.append(f'{field} differs')
            expected=configs[(job['method'],job['reference_subject'])]['parameters']
            try:
                observed=json.loads(record['configuration_parameters'])
                if not isinstance(observed,dict):raise ValueError('expected object')
                if set(observed)!=set(expected):issues.append('configuration field set differs')
                for field,value in expected.items():
                    if field not in observed or solver.norm(observed[field])!=solver.norm(value):
                        issues.append(f'configuration differs: {field}')
            except (ValueError,TypeError):issues.append('invalid configuration JSON')
            if record['status']=='FAIL':
                if not record['failure_reason'].strip():issues.append('failure reason absent')
                if any(record[field].strip() for field in xyz):issues.append('failed record contains peak coordinates')
                for field in metadata:
                    if not record[field].strip():incomplete.append(f'{field} absent on failed attempt')
                if record['montage_template_subject'] and record['montage_template_subject']!=job['reference_subject']:
                    issues.append('montage template differs')
            elif record['status']=='PASS':
                if record['montage_template_subject']!=job['reference_subject']:issues.append('montage template differs')
                if record['coordinate_frame']!='model_head' or record['coordinate_units']!='mm':
                    issues.append('native head frame/units differ')
                try:
                    peak=np.array([float(record[field]) for field in xyz])
                    if not np.isfinite(peak).all() or np.linalg.norm(peak)>1000:issues.append('invalid native head peak')
                except (ValueError,TypeError):issues.append('missing/nonnumeric native peak')
                for field in ['eeg_data_sha256','epoch_times_sha256']:
                    if not re.fullmatch('[0-9a-fA-F]{64}',record[field]):issues.append(f'invalid {field}')
                parsed={}
                for field in ['actual_channels_json','nominal_channels_json','original_bad_channels_json']:
                    try:
                        value=json.loads(record[field])
                        if not isinstance(value,list) or not all(isinstance(c,str) for c in value):
                            raise ValueError('expected channel list')
                        if len(value)!=len(set(value)):raise ValueError('duplicate channels')
                        if field!='original_bad_channels_json' and not value:raise ValueError('empty channels')
                        parsed[field]=value
                    except (ValueError,TypeError):issues.append(f'invalid {field}')
                if 'actual_channels_json' in parsed:
                    actual=parsed['actual_channels_json']
                    try:
                        count=float(record['channels'])
                        if count!=len(actual):issues.append('recorded channel count differs')
                    except (ValueError,TypeError):issues.append('invalid recorded channel count')
                    if set(actual)&set(parsed.get('original_bad_channels_json',[])):
                        issues.append('selected channels include marked bads')
                if job['method'] not in ['Continuous-ECD','RAP-MUSIC'] and record['hemisphere_constraint_applied'].lower()!='true':
                    issues.append('primary hemisphere peak policy differs')
            else:issues.append('unknown canonical status')
        check='MISMATCH' if issues else ('INCOMPLETE_FAILURE_METADATA' if incomplete else 'MATCHED_RECORD_ONLY')
        rows.append(dict(**{k:v for k,v in job.items() if k!='source_condition'},
            source_condition=record['source_condition'] if record else job['source_condition'],
            canonical_status=record['status'] if record else 'MISSING',record_check=check,
            issues='; '.join(issues),incomplete_metadata='; '.join(incomplete),source_csv=str(path),
            reuse_status='BLOCKED_RECORD_MISMATCH' if issues else 'PENDING_INDEPENDENT_PROVENANCE_VALIDATION'))
    sources=[]
    for file in sorted(args.canonical_root.iterdir()):
        if file.is_file() and not file.name.startswith('.'):
            sources.append(dict(path=str(file),sha256=solver.digest(file),size_bytes=file.stat().st_size))
    report=dict(status='AUDIT_ONLY_NOT_IMPORTED',n_planned=len(full),n_requested=len(requested),
        n_canonical_rows=len(table),canonical_status_counts=table.status.value_counts().to_dict(),
        record_mismatches=sum(row['record_check']=='MISMATCH' for row in rows)+len(unexpected)+len(missing-requested_keys),
        incomplete_failure_metadata=sum(row['record_check']=='INCOMPLETE_FAILURE_METADATA' for row in rows),
        missing_keys=[list(key) for key in sorted(missing)],unexpected_keys=[list(key) for key in sorted(unexpected)],
        sources=sources,script05_sha256=solver.digest(__file__),script03_sha256=solver.digest(args.script03),
        selection_sha256=solver.digest(args.selection_root/'01_single_reference_selected_parameters.csv'),
        inverse_reconstruction_executed=False,registration_fitting_executed=False,peaks_imported=False,
        anatomical_evaluation_ready=False,
        unresolved=['Raw EEG/channel fingerprints have not been independently recomputed.',
                    'Fit-time software, forward/BEM and execution provenance need matching.',
                    'Head/surface to NIfTI physical RAS bridge and registration provenance/QA remain unresolved.',
                    'Primary failures remain failures; whole-brain sensitivities must stay separate.'])
    return rows,report


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--phase',choices=['reuse-audit','plan','reconstruct'],default='reuse-audit')
    p.add_argument('--project-root',type=Path,default=ROOT)
    p.add_argument('--dataset',type=Path,default=Path.home()/'Documents'/'DATASETS SOURCE LOCALIZATION'/'Localize-MI')
    p.add_argument('--selection-root',type=Path,default=EXP/'outputs/04_single_reference_parameters')
    p.add_argument('--source-root',type=Path,default=EXP/'source/02_parameter_grids')
    p.add_argument('--preparation-root',type=Path,default=EXP/'outputs/01_prepare_experiment')
    p.add_argument('--code-source-root',type=Path,default=EXP/'source/03_solver_code')
    p.add_argument('--script03',type=Path,default=EXP/'scripts/03_run_localization.py')
    p.add_argument('--script02',type=Path,default=EXP/'scripts/02_select_loso_parameters.py')
    p.add_argument('--script04',type=Path,default=EXP/'scripts/04_select_single_reference_parameters.py')
    p.add_argument('--bem-root',type=Path,default=ROOT/'outputs/12_additional_methods_all_runs/bem')
    p.add_argument('--output-root',type=Path,help='Default output basename follows phase; existing plan/audit folders are refused')
    p.add_argument('--method',choices=METHODS,action='append')
    p.add_argument('--reference-subject',action='append')
    p.add_argument('--test-subject',action='append')
    p.add_argument('--run',action='append')
    p.add_argument('--limit',type=int,default=0)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--quiet',action='store_true')
    p.add_argument('--canonical-root',type=Path,default=ROOT/'cross_subject_no_warp/outputs/05_single_reference_final')
    a=p.parse_args()
    if a.output_root is None:
        basename={'reuse-audit':'05_single_reference_reuse_audit','plan':'05_single_reference_plan',
                  'reconstruct':'05_single_reference_reconstruction'}[a.phase]
        a.output_root=EXP/'outputs'/basename
    for name,value in vars(a).items():
        if isinstance(value,Path):setattr(a,name,value.expanduser().resolve())
    if a.limit<0:raise ValueError('--limit must be >= 0')
    if a.phase!='reconstruct' and a.resume:
        raise ValueError('--resume applies only to reconstruction')
    if a.output_root.exists() or a.output_root.is_symlink():
        if a.phase!='reconstruct' or not a.resume:
            raise FileExistsError('Output already exists; use a NEW --output-root')
        if not (a.output_root/'06_frozen_run_manifest.json').is_file():
            raise ValueError('Resume requires an existing frozen reconstruction manifest')
    for protected in [a.canonical_root,a.selection_root,a.preparation_root,a.source_root,
                      a.code_source_root,a.dataset,ROOT/'cross_subject_no_warp',ROOT/'cross_subject',
                      EXP/'source',EXP/'scripts',a.bem_root,a.project_root/'src']:
        if a.output_root==protected or protected in a.output_root.parents or a.output_root in protected.parents:
            raise ValueError(f'Output overlaps protected input tree: {protected}')
    a.project_root=a.project_root.resolve();a.dataset=a.dataset.expanduser().resolve()
    a.code_source_root=a.code_source_root.resolve()
    a.montage_template='reference'
    solver=load_module(a.script03.resolve())
    selected,schedule,doc,state=verify_selection(a,solver)
    solver.snapshot_code(a.project_root,a.code_source_root)
    configs,errors=resolve_configurations(selected,solver,a.source_root,doc)
    full=build_jobs(selected,schedule)
    requested=[j for j in full if (not a.method or j['method'] in a.method)
        and (not a.reference_subject or j['reference_subject'] in a.reference_subject)
        and (not a.test_subject or j['test_subject'] in a.test_subject)
        and (not a.run or j['run'] in a.run)]
    if not requested:raise ValueError('Filters matched no jobs')
    print(f'Full plan: {len(full)} fits; requested: {len(requested)} fits; configs: {len(configs)}/84.')
    if not a.resume:a.output_root.mkdir(parents=True,exist_ok=False)
    with solver.lock_output(a.output_root):
        out=a.output_root
        frozen_path=out/'06_frozen_run_manifest.json'
        versions={}
        for package in ('mne','numpy','scipy','pandas','sesameeg','invertmeeg'):
            try:versions[package]=importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:versions[package]='not_installed'
        fixed=dict(script05_sha256=solver.digest(__file__),script03_sha256=solver.digest(a.script03),
            selected_csv_sha256=solver.digest(a.selection_root/'01_single_reference_selected_parameters.csv'),
            schedule_sha256=solver.digest(a.selection_root/'09_reference_test_schedule.csv'),
            source_manifest_sha256=solver.digest(a.source_root/'source_manifest.json'),
            dataset=str(a.dataset),bem_root=str(a.bem_root.resolve()),
            code_source_root=str(a.code_source_root),montage_template='reference',
            versions=versions,hemisphere_peak_policy='known stimulation hemisphere',
            code_manifest_sha256=solver.digest(a.code_source_root/'code_manifest.json'),
            coordinate_audit_sha256=solver.digest(a.preparation_root/'05_coordinate_audit.csv'),
            registration_modes=['affine','syn'],anatomical_mapping_executed=False)
        if frozen_path.exists():
            old=json.loads(frozen_path.read_text())
            if old!=fixed:raise ValueError('Run manifest mismatch; use a NEW output-root for changed inputs/script')
        solver.atomic_csv(out/'02_reconstruction_plan.csv',pd.DataFrame(full))
        solver.atomic_csv(out/'03_configuration_preflight.csv',pd.DataFrame(errors,columns=['method','test_subject','error']))
        solver.atomic_json(out/'05_resolved_configurations.json',{'|'.join(k):v for k,v in configs.items()})
        if errors:raise SystemExit(f'Parameter preflight failed for {len(errors)} configs; NO reconstruction attempted')
        if a.phase=='reuse-audit':
            solver.check_audit(a,a.source_root,doc)
            rows,report=audit_reuse(a,solver,full,requested,configs)
            solver.atomic_csv(out/'11_reuse_candidate_audit.csv',pd.DataFrame(rows))
            solver.atomic_json(out/'12_reuse_audit_manifest.json',report)
            print('Reuse audit written; candidates are not imported or independently validated.')
            if report['record_mismatches'] or report['incomplete_failure_metadata']:raise SystemExit(2)
            return
        if a.phase=='plan':
            print('Plan/preflight passed; no EEG loaded, no solver run.');return
        solver.check_audit(a,a.source_root,doc)
        solver.snapshot_code(a.project_root,a.code_source_root)
        engine=solver.Engine(a)
        if not frozen_path.exists():solver.atomic_json(frozen_path,fixed)
        status_path=out/'01_execution_status.csv'
        old_rows=pd.read_csv(status_path,keep_default_na=False).to_dict('records') if status_path.exists() else []
        if old_rows and not a.resume:raise ValueError('Existing checkpoint: use --resume')
        recorded={tuple(r[k] for k in KEYS):r for r in old_rows}
        if len(recorded)!=len(old_rows):raise ValueError('Duplicate checkpoint keys')
        allowed={tuple(j[k] for k in KEYS) for j in full}
        if not set(recorded).issubset(allowed):raise ValueError('Checkpoint contains jobs outside plan')
        n=0
        for job in requested:
            key=tuple(job[k] for k in KEYS)
            if key in recorded:continue
            config=configs[(job['method'],job['reference_subject'])]
            if config['configuration_key']!=job['configuration_key']:raise ValueError('Configuration key changed')
            row=dict(**job,status='FAIL',failure_reason='',configuration_parameters=json.dumps(config['parameters'],sort_keys=True))
            start=time.perf_counter()
            try:
                row.update(engine.solve(job,config['parameters']))
                row.update(status='PASS',failure_reason='')
            except Exception as exc:
                row['failure_reason']=f'{type(exc).__name__}: {exc}'
                with (out/'04_failure_tracebacks.log').open('a') as f:
                    f.write('\n'+json.dumps(job)+'\n'+traceback.format_exc())
            row['runtime_seconds']=round(time.perf_counter()-start,6)
            recorded[key]=row;n+=1
            table=pd.DataFrame(list(recorded.values()))
            solver.atomic_csv(status_path,table)
            channel_audits(table,solver,out)
            print(f'{len(recorded)}/{len(full)} recorded | {job["method"]} {job["reference_subject"]} -> {job["test_subject"]} {job["run"]}: {row["status"]} {row["failure_reason"]}',flush=True)
            if a.limit and n>=a.limit:break
        result=pd.DataFrame(list(recorded.values()))
        if len(result):
            summary=result.groupby(['method','reference_subject','status']).size().reset_index(name='count')
            solver.atomic_csv(out/'07_execution_summary.csv',summary)
            print(summary.to_string(index=False))
        (out/'README.txt').write_text(__doc__)
        print(f'Saved {len(recorded)}/{len(full)} status records in {out}')
        print('PASS means solver completed, NOT validated cross-subject localization error.')

if __name__=='__main__':main()

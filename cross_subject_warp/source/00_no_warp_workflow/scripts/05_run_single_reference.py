#!/usr/bin/env python3
"""Script 05 — Single-reference EEG source reconstruction, no warping.

PURPOSE
  Use the 84 reference-only configurations selected by Script 04 (7 references
  x 12 inverse methods). Transfer each reference's parameters and forward
  model to all six other test participants' EEG, without affine/SyN/warping.
  The test participant supplies EEG and bad-channel flags; the reference
  supplies the forward model and nominal 32/64/128 montage geometry.
  Uses the already-audited, PATCHED Script 03 solver (including all_good fix).
  The actual montage count is selected by Script 04's parameters. The original
  reference montage is not replaced with a fixed sub-01 montage.

INPUTS (project-root-relative defaults)
  outputs/04_single_reference_parameters/{01_single_reference_selected_parameters.csv,
    07_selection_manifest.json,09_reference_test_schedule.csv}
  source/02_parameter_grids/source_manifest.json (frozen historical grids)
  outputs/01_prepare_experiment/{07_experiment_manifest.json,05_coordinate_audit.csv}
  source/03_solver_code (immutable copies of previous original solver code)
  scripts/03_run_localization.py (PATCHED version; imported, not modified)
  Localize-MI dataset, forward models and BEM solution where needed.

SELECTION AND EVALUATION
  Script 04 selects the best configuration using only reference-subject
  historical runs (strict all-pass). No test subject performance is used.
  61 runs x 6 other references x 12 methods = 4,392 CROSS fits, no within.
  Known stimulation hemisphere is used for peak selection exactly as in
  Script 03. No localization error is computed (reserved for Script 06).
  Different head models have different coordinate frames: never interpret
  raw coordinate distances across subjects as anatomical localization error.
  Historical grid search on the same cohort remains a retrospective study.

EXAMPLE TERMINAL INPUTS (from project root, activate .venv first)
  python cross_subject_no_warp/scripts/05_run_single_reference.py --phase plan
  python cross_subject_no_warp/scripts/05_run_single_reference.py \\
    --phase reconstruct --method MNE --reference-subject sub-01 \\
    --test-subject sub-03 --run run-01 --limit 1 --quiet
  python cross_subject_no_warp/scripts/05_run_single_reference.py \\
    --phase reconstruct --method MNE --resume --quiet
  python cross_subject_no_warp/scripts/05_run_single_reference.py \\
    --phase reconstruct --resume --quiet

OUTPUTS (outputs/05_single_reference_reconstruction)
  01_execution_status.csv         Each attempted job with coordinates/failure
  02_reconstruction_plan.csv      All 4392 planned combinations
  03_configuration_preflight.csv  Parameter-preflight errors (must be empty)
  04_failure_tracebacks.log       Exception traces when needed
  05_resolved_configurations.json Exact parameters for 84 method/reference pairs
  06_frozen_run_manifest.json     Checksums, source paths, solver versions
  07_execution_summary.csv        Method/reference PASS/FAIL summary
  09_channel_selection_audit.csv  One row per attempted reconstruction,
                                  nominal/actual/bad/replaced channel lists
  10_channel_selection_summary.csv Compact per-attempt channel QA
  README.txt                     Data provenance, interpretation

SAFETY
  Single writer per folder. Existing checkpoint requires --resume. Completed
  PASS/FAIL are not retried automatically. Do not delete earlier results.
  Manifest rejects changed scripts, input selections and solver snapshots.
  Runs from a NEW output root if you change any experimental setting.
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
    for field,path in [('selected_csv_sha256',sel),('source_manifest_sha256',args.source_root/'source_manifest.json')]:
        if solver.digest(path)!=state[field]:raise ValueError(f'Script 04 input checksum mismatch: {path}')
    if solver.digest(args.script02)!=state['script02_sha256'] or solver.digest(args.script04)!=state['script04_sha256']:
        raise ValueError('Selection scripts differ from frozen Script 04 manifest')
    if state.get('intended_reconstruction_montage_template')!='reference subject':
        raise ValueError('Script 04 montage policy mismatch')
    source_doc=json.loads((args.source_root/'source_manifest.json').read_text())
    for item in source_doc['files']:
        if solver.digest(args.source_root/item['copy'])!=item['sha256']:
            raise ValueError(f'Frozen source grid modified: {item["copy"]}')
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
                condition='single_reference_no_warp',model_subject=p.reference_subject,
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

def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--phase',choices=['plan','reconstruct'],default='plan')
    p.add_argument('--project-root',type=Path,default=Path.cwd())
    p.add_argument('--dataset',type=Path,default=Path.home()/'Documents'/'DATASETS SOURCE LOCALIZATION'/'Localize-MI')
    p.add_argument('--selection-root',type=Path,default=Path('cross_subject_no_warp/outputs/04_single_reference_parameters'))
    p.add_argument('--source-root',type=Path,default=Path('cross_subject_no_warp/source/02_parameter_grids'))
    p.add_argument('--preparation-root',type=Path,default=Path('cross_subject_no_warp/outputs/01_prepare_experiment'))
    p.add_argument('--code-source-root',type=Path,default=Path('cross_subject_no_warp/source/03_solver_code'))
    p.add_argument('--script03',type=Path,default=Path('cross_subject_no_warp/scripts/03_run_localization.py'))
    p.add_argument('--script02',type=Path,default=Path('cross_subject_no_warp/scripts/02_select_loso_parameters.py'))
    p.add_argument('--script04',type=Path,default=Path('cross_subject_no_warp/scripts/04_select_single_reference_parameters.py'))
    p.add_argument('--bem-root',type=Path,default=Path('outputs/12_additional_methods_all_runs/bem'))
    p.add_argument('--output-root',type=Path,default=Path('cross_subject_no_warp/outputs/05_single_reference_reconstruction'))
    p.add_argument('--method',choices=METHODS,action='append')
    p.add_argument('--reference-subject',action='append')
    p.add_argument('--test-subject',action='append')
    p.add_argument('--run',action='append')
    p.add_argument('--limit',type=int,default=0)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--quiet',action='store_true')
    a=p.parse_args()
    if a.limit<0:raise ValueError('--limit must be >= 0')
    a.project_root=a.project_root.resolve();a.dataset=a.dataset.expanduser().resolve()
    a.code_source_root=a.code_source_root.resolve()
    a.montage_template='reference'
    solver=load_module(a.script03.resolve())
    selected,schedule,doc,state=verify_selection(a,solver)
    configs,errors=resolve_configurations(selected,solver,a.source_root,doc)
    full=build_jobs(selected,schedule)
    requested=[j for j in full if (not a.method or j['method'] in a.method)
        and (not a.reference_subject or j['reference_subject'] in a.reference_subject)
        and (not a.test_subject or j['test_subject'] in a.test_subject)
        and (not a.run or j['run'] in a.run)]
    if not requested:raise ValueError('Filters matched no jobs')
    print(f'Full plan: {len(full)} fits; requested: {len(requested)} fits; configs: {len(configs)}/84.')
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
            versions=versions,hemisphere_peak_policy='known stimulation hemisphere')
        if frozen_path.exists():
            old=json.loads(frozen_path.read_text())
            if old!=fixed:raise ValueError('Run manifest mismatch; use a NEW output-root for changed inputs/script')
        solver.atomic_csv(out/'02_reconstruction_plan.csv',pd.DataFrame(full))
        solver.atomic_csv(out/'03_configuration_preflight.csv',pd.DataFrame(errors,columns=['method','test_subject','error']))
        solver.atomic_json(out/'05_resolved_configurations.json',{'|'.join(k):v for k,v in configs.items()})
        if errors:raise SystemExit(f'Parameter preflight failed for {len(errors)} configs; NO reconstruction attempted')
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

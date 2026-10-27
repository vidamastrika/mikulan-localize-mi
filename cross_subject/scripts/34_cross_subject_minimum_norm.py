#!/usr/bin/env python3
"""Script 34: four fixed minimum-norm methods, within & cross subject, evaluation and sensitivity.

Uses Script 09's inverse and montage helper functions, Script 32's validated ANTs
mapping functions and Script 33's cluster-aware sensitivity summaries.

Run from repository root with .venv + localizemi-registration ANTs available.

  python cross_subject/scripts/34_cross_subject_minimum_norm.py --phase plan
  python cross_subject/scripts/34_cross_subject_minimum_norm.py --phase reconstruct --limit 10 --resume
  python cross_subject/scripts/34_cross_subject_minimum_norm.py --phase reconstruct --resume
  micromamba run -n localizemi-registration .venv/bin/python \\
    cross_subject/scripts/34_cross_subject_minimum_norm.py --phase all --resume

Output folder: cross_subject/outputs/34_cross_subject_minimum_norm/

IMPORTANT scientific qualifiers:
* Reference model registration dependent; native T1w RAS remains provisional.
* Hemisphere selection follows Script 09's known stimulation hemisphere and is
  therefore an evaluation-informed hemisphere restriction, not blind localization.
* Frozen hyperparameters were tuned on these same seven subjects, not held-out CV.
* Script 33 CIs resample seven test-subject clusters; exploratory only.
* Reconstruction runtime includes preprocessing, covariance and inverse fitting;
  differs from Script 09's apply-inverse-only timing definition.

  Methods:
    MNE, dSPM, sLORETA, eLORETA

Experiment:
    366 cross-subject combinations × 4 methods = 1,464
    61 subject-specific runs × 4 methods = 244
    Total = 1,708 reconstructions

Run all commands from the project root:
    cd ~/projects/mikulan-localize-mi
    source .venv/bin/activate


1. CHECK EXPERIMENT PLAN (no reconstruction)
--------------------------------------------------
python cross_subject/scripts/34_cross_subject_minimum_norm.py \
    --phase plan


2. SMALL PILOT (10 new reconstructions)
--------------------------------------------------
python cross_subject/scripts/34_cross_subject_minimum_norm.py \
    --phase reconstruct \
    --limit 10 \
    --resume \
    --quiet


3. RUN ALL REMAINING RECONSTRUCTIONS
--------------------------------------------------
python cross_subject/scripts/34_cross_subject_minimum_norm.py \
    --phase reconstruct \
    --resume \
    --quiet


4. RUN ONLY ONE METHOD (optional)
--------------------------------------------------
python cross_subject/scripts/34_cross_subject_minimum_norm.py \
    --phase reconstruct \
    --method MNE \
    --resume \
    --quiet

Replace MNE with dSPM, sLORETA, or eLORETA.


5. EVALUATE LOCALIZATION ERRORS
--------------------------------------------------
micromamba run -n localizemi-registration \
    .venv/bin/python \
    cross_subject/scripts/34_cross_subject_minimum_norm.py \
    --phase evaluate


6. ANALYZE REFERENCE-SUBJECT SENSITIVITY
--------------------------------------------------
python cross_subject/scripts/34_cross_subject_minimum_norm.py \
    --phase analyze


7. RUN ALL PHASES AUTOMATICALLY (alternative)
--------------------------------------------------
micromamba run -n localizemi-registration \
    .venv/bin/python \
    cross_subject/scripts/34_cross_subject_minimum_norm.py \
    --phase all \
    --resume \
    --quiet


8. REPEAT SENSITIVITY ANALYSIS WITH 5000 BOOTSTRAPS
--------------------------------------------------
python cross_subject/scripts/34_cross_subject_minimum_norm.py \
    --phase analyze \
    --bootstrap 5000


OUTPUT DIRECTORY:
    cross_subject/outputs/34_cross_subject_minimum_norm/

IMPORTANT:
    --resume : reuse completed reconstructions
    --limit  : limit the number of NEW reconstructions
    --quiet  : reduce terminal output

    Existing results and runtime records are preserved.
    Affine and SyN registrations from Script 31 are reused.
    The evaluation phase requires the corresponding
    reconstruction results to be complete.
"""
from __future__ import annotations
import argparse, csv, hashlib, importlib.util, json, math, os, subprocess, sys, time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import mne
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT/'src'))
from localize_mi import load_run
from localize_mi.inverse import (apply_average_reference, estimate_noise_covariance,
    create_inverse_operator, create_target_evoked, apply_inverse_method)
from localize_mi.metrics import load_stimulation_info, source_vertex_coordinate

METHODS=('MNE','dSPM','sLORETA','eLORETA')
# Frozen selections verified against the user-supplied Script 17 selection CSV.
FROZEN={
  'MNE':{'montage':'32','loose':0.6,'depth':0.2,'snr':1.0,'lambda2':1.0},
  'dSPM':{'montage':'128','loose':0.9,'depth':0.6,'snr':1.0,'lambda2':1.0},
  'sLORETA':{'montage':'128','loose':0.5,'depth':0.6,'snr':1.0,'lambda2':1.0},
  'eLORETA':{'montage':'128','loose':0.1,'depth':0.1,'snr':1.0,'lambda2':1.0},
}
FIELDS=['method','test_subject','reference_subject','run','is_within','status','failure_reason',
        'montage','loose','depth','snr','lambda2','channels','replacement_count',
        'n_epochs','stimulation_pair','hemisphere','peak_vertex','peak_time_s',
        'reference_head_x_mm','reference_head_y_mm','reference_head_z_mm',
        'runtime_seconds','covariance_method_selected']
EVAL_FIELDS=['method','test_subject','reference_subject','run','status','failure_reason',
  'within_subject_error_mm','cross_affine_error_mm','cross_syn_error_mm',
  'affine_minus_within_mm','syn_minus_within_mm',
  'ground_truth_x_mm','ground_truth_y_mm','ground_truth_z_mm',
  'reference_head_x_mm','reference_head_y_mm','reference_head_z_mm',
  'affine_test_x_mm','affine_test_y_mm','affine_test_z_mm',
  'syn_test_x_mm','syn_test_y_mm','syn_test_z_mm',
  'affine_roundtrip_mm','syn_roundtrip_mm','runtime_seconds','montage','loose','depth',
  'snr','lambda2','channels','source_status','coordinate_assumption']


def module(path,name):
    if not path.exists(): raise FileNotFoundError(f'Required project script missing: {path}')
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec); sys.modules[name]=mod
    spec.loader.exec_module(mod)
    return mod


def args_parse():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--phase',choices=['plan','reconstruct','evaluate','analyze','all'],default='plan')
    p.add_argument('--resume',action='store_true',help='Reuse completed rows; reject changed manifest')
    p.add_argument('--limit',type=int,default=0,help='Maximum NEW reconstructions this call (pilot)')
    p.add_argument('--method',choices=METHODS,action='append',help='Subset of four methods')
    p.add_argument('--dataset',type=Path,default=Path.home()/'Documents'/'DATASETS SOURCE LOCALIZATION'/'Localize-MI')
    p.add_argument('--selected-configs',type=Path,default=None,help='Optional Script 17 CSV; verified against embedded frozen values')
    p.add_argument('--schedule',type=Path,default=Path('cross_subject/outputs/30_all_reference_sesame_batch/experiment_schedule_all_references.csv'))
    p.add_argument('--registration-status',type=Path,default=Path('cross_subject/outputs/31_all_reference_registration/registration_status.csv'))
    p.add_argument('--output-root',type=Path,default=Path('cross_subject/outputs/34_cross_subject_minimum_norm'))
    p.add_argument('--script09',type=Path,default=Path('scripts/09_full_parameter_grid.py'))
    p.add_argument('--script32',type=Path,default=Path('cross_subject/scripts/32_all_reference_localization_evaluation.py'))
    p.add_argument('--script33',type=Path,default=Path('cross_subject/scripts/33_reference_subject_sensitivity.py'))
    p.add_argument('--bootstrap',type=int,default=3000)
    p.add_argument('--quiet',action='store_true')
    return p.parse_args()


def config_table(path,methods):
    if path is None:return {m:dict(FROZEN[m]) for m in methods}
    d=pd.read_csv(path)
    d=d[d.method.isin(methods)]
    if len(d)!=len(methods) or d.method.duplicated().any():
        raise ValueError('Selected configurations CSV must have exactly one row for each chosen method')
    configs={}
    for _,r in d.iterrows():
        method=str(r.method)
        cfg={'montage':str(r.montage),'loose':float(r.loose),'depth':float(r.depth),
             'snr':float(r.snr),'lambda2':float(r.lambda2)}
        if not np.isclose(cfg['lambda2'],1/cfg['snr']**2):raise ValueError(f'{method}: lambda2/SNR mismatch')
        if cfg['montage'] not in ('32','64','128','all_good'):raise ValueError('Invalid montage '+cfg['montage'])
        if not 0<=cfg['loose']<=1 or not 0<=cfg['depth']<=1:raise ValueError('Invalid loose/depth')
        for k,v in cfg.items():
            expected=FROZEN[method][k]
            if k=='montage':
                if str(v)!=str(expected):raise ValueError(f'{method} {k}: selected CSV differs from frozen definition')
            elif not np.isclose(v,expected):raise ValueError(f'{method} {k}: selected CSV differs from frozen definition')
        configs[method]=cfg
    return configs


def schedule_jobs(path):
    d=pd.read_csv(path)
    needed={'target','donor','run'}
    if not needed.issubset(d): raise ValueError(f'Schedule lacks {needed-set(d)}')
    d=d[['target','donor','run']].drop_duplicates().sort_values(['target','run','donor'])
    if len(d)!=366 or d.target.nunique()!=7:raise ValueError('Expected full 366 cross-subject schedule')
    if (d.target==d.donor).any():raise ValueError('Cross schedule contains self pair')
    groups=d.groupby(['target','run']).donor.nunique()
    if len(groups)!=61 or not groups.eq(6).all():raise ValueError('Expected 61 EEG runs x 6 references')
    cross=[(r.target,r.donor,r.run) for r in d.itertuples(index=False)]
    own=[(t,t,run) for t,run in groups.index]
    return own+cross


def atomic_csv(path,rows,cols):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore'); w.writeheader(); w.writerows(rows)
    os.replace(tmp,path)


def existing(path):
    if not path.exists():return []
    return pd.read_csv(path,keep_default_na=False).to_dict('records')


def get_forward(dataset,ref,cache):
    if ref not in cache:
        path=dataset/'derivatives'/'sourcemodelling'/ref/'fwd'/f'{ref}_fwd.fif'
        cache[ref]=mne.read_forward_solution(str(path),verbose=False)
    return cache[ref]


def transplant_geometry(epochs,forward):
    names=list(forward['info']['ch_names'])
    if set(epochs.ch_names)!=set(names):
        raise ValueError('Test EEG sensor names do not equal reference forward names')
    ch_pos={ch['ch_name']:np.asarray(ch['loc'][:3],float)
            for ch in forward['info']['chs']}
    if any(not np.isfinite(x).all() for x in ch_pos.values()):raise ValueError('Invalid reference sensor positions')
    montage=mne.channels.make_dig_montage(ch_pos=ch_pos,coord_frame='head')
    cloned=epochs.copy()
    cloned.set_montage(montage,on_missing='raise',verbose=False)
    for ch in cloned.info['chs']:
        if not np.allclose(ch['loc'][:3],ch_pos[ch['ch_name']],atol=1e-6):
            raise ValueError('Reference sensor coordinate assignment failed')
    return cloned


def selected_channels(script09,forward,good,nominal,montage,template_info):
    if montage=='all_good':return list(good),0
    names,xyz=template_info
    if set(forward['info']['ch_names'])!=set(names):raise ValueError('Reference model channel names mismatch template')
    selected,repl=script09.replace_bad_nominal(nominal[montage],names,xyz,good)
    return selected,len(repl)


def fit(a,job,configs,script09,forward_cache,nominal,template_info):
    method,test,ref,run=job
    cfg=configs[method];t0=time.perf_counter()
    out={k:'' for k in FIELDS}
    out.update(method=method,test_subject=test,reference_subject=ref,run=run,
               is_within=(test==ref),status='FAIL',montage=cfg['montage'],
               loose=cfg['loose'],depth=cfg['depth'],snr=cfg['snr'],lambda2=cfg['lambda2'])
    try:
        dataset=a.dataset.expanduser()
        loaded=load_run(dataset,test,task='seegstim',run=run)
        forward=get_forward(dataset,ref,forward_cache)
        # Source evaluation uses only test stimulation hemisphere in peak selection;
        # stimulation coordinate itself is NEVER passed to inverse solver.
        stim=load_stimulation_info(loaded.metadata['Description'],dataset/'derivatives'/'epochs'/test/'ieeg'/f'{test}_task-seegstim_space-surface_electrodes.tsv')
        out.update(stimulation_pair=stim.pair,hemisphere=stim.hemisphere,n_epochs=len(loaded.epochs))
        good=[n for n in loaded.epochs.ch_names if n not in loaded.epochs.info['bads']]
        actual,nrepl=selected_channels(script09,forward,good,nominal,cfg['montage'],template_info)
        out.update(channels=len(actual),replacement_count=nrepl)
        eeg=transplant_geometry(loaded.epochs,forward)
        referenced=apply_average_reference(eeg)
        # Match Script 09: average reference FIRST, then select montage.
        chosen=referenced if cfg['montage']=='all_good' else referenced.copy().pick(actual)
        cov=estimate_noise_covariance(chosen,-.250,-.050,method='auto',verbose=not a.quiet)
        evoked=create_target_evoked(chosen,-.002,.002)
        inverse=create_inverse_operator(chosen,forward,cov,loose=cfg['loose'],depth=cfg['depth'],verbose=not a.quiet)
        stc,_,_=apply_inverse_method(evoked,inverse,method,snr=cfg['snr'],verbose=not a.quiet)
        vertex,peak_time=stc.get_peak(hemi=stim.hemisphere,mode='abs')
        hemi_idx=0 if stim.hemisphere=='lh' else 1
        # The forward source vertices are used as head coordinates as in Script 32's
        # SESAME source-point calculation; verify forward coordinate frames in pilot.
        head=np.asarray(forward['src'][hemi_idx]['rr'][int(vertex)],float)*1000
        if not np.isfinite(head).all():raise ValueError('Invalid peak coordinates')
        out.update(status='PASS',failure_reason='',peak_vertex=int(vertex),peak_time_s=float(peak_time),
                   reference_head_x_mm=head[0],reference_head_y_mm=head[1],reference_head_z_mm=head[2],
                   covariance_method_selected=str(cov.get('method','auto')))
    except Exception as exc:
        out['failure_reason']=f'{type(exc).__name__}: {exc}'
    out['runtime_seconds']=round(time.perf_counter()-t0,6)
    return out


def evaluate(a,configs,s32,source_rows,jobs):
    out=[]; reg_cache={};exe=s32.exe_location(a)
    reg=pd.read_csv(a.registration_status)
    for r in reg.itertuples(index=False):
        if str(r.affine_status) not in ('PASS','SKIPPED_COMPLETE') or str(r.syn_status) not in ('PASS','SKIPPED_COMPLETE'):
            raise ValueError(f'Incomplete registration: {r.reference_subject}->{r.test_subject}')
    if len(reg)!=42:raise ValueError('Expected 42 registration pairs')
    keys={(str(r['method']),str(r['test_subject']),str(r['reference_subject']),str(r['run'])):r for r in source_rows}
    if len(keys)!=len(source_rows):raise ValueError('Duplicate source rows')
    expected={(m,t,ref,run) for m in configs for t,ref,run in jobs}
    if set(keys)!=expected:raise ValueError(f'Incomplete reconstruction grid: {len(keys)} vs {len(expected)}; finish reconstruction first')
    for method in configs:
      for test,ref,run in jobs:
        if test==ref:continue
        k=(method,test,ref,run); src=keys[k]; own=keys[(method,test,test,run)]
        row={name:'' for name in EVAL_FIELDS}
        row.update(method=method,test_subject=test,reference_subject=ref,run=run,status='FAIL',
                   source_status=src['status'],runtime_seconds=src['runtime_seconds'],
                   montage=src['montage'],loose=src['loose'],depth=src['depth'],
                   snr=src['snr'],lambda2=src['lambda2'],channels=src['channels'],
                   coordinate_assumption='Provisional T1w NIfTI RAS + head/surface mapping')
        try:
            if src['status']!='PASS' or own['status']!='PASS':raise ValueError('Source or matched within reconstruction not PASS')
            if str(src['stimulation_pair'])!=str(own['stimulation_pair']):raise ValueError('Stimulation pair mismatch')
            # From source head positions to subject-specific surface mm.
            source=np.array([float(src[f'reference_head_{i}_mm']) for i in 'xyz'])
            ownhead=np.array([float(own[f'reference_head_{i}_mm']) for i in 'xyz'])
            ref_h=s32.h5(a.dataset,ref);test_h=s32.h5(a.dataset,test)
            own_native=s32.transform(test_h,ownhead)
            # Ground truth is loaded only for evaluation, never fitting.
            truth=s32.contact_midpoint(a.dataset,test,src['stimulation_pair'])
            own_err=float(np.linalg.norm(own_native-truth))
            # Replicate Script 32 point mapping exactly, including RAS/LPS.
            reference_surface=s32.transform(ref_h,source)
            ref_t1w=s32.transform(np.linalg.inv(ref_h),reference_surface)
            pair=s32.pair_check(exe,a,ref,test,reg_cache)
            aff=s32.point_map(exe,ref_t1w,pair['terms']['aff_r2t'])[0]
            syn=s32.point_map(exe,ref_t1w,pair['terms']['syn_r2t'])[0]
            back_a=s32.point_map(exe,aff,pair['terms']['aff_t2r'])[0]
            back_s=s32.point_map(exe,syn,pair['terms']['syn_t2r'])[0]
            ra=float(np.linalg.norm(back_a-ref_t1w));rs=float(np.linalg.norm(back_s-ref_t1w))
            if max(ra,rs)>a.max_roundtrip_mm:raise ValueError(f'Roundtrip too large: {ra:.4f} {rs:.4f} mm')
            an=s32.transform(test_h,aff);sn=s32.transform(test_h,syn)
            ea=float(np.linalg.norm(an-truth));es=float(np.linalg.norm(sn-truth))
            row.update(status='PASS_PROVISIONAL',failure_reason='',
                       within_subject_error_mm=own_err,cross_affine_error_mm=ea,cross_syn_error_mm=es,
                       affine_minus_within_mm=ea-own_err,syn_minus_within_mm=es-own_err,
                       affine_roundtrip_mm=ra,syn_roundtrip_mm=rs)
            for prefix,xyz in [('ground_truth',truth),('reference_head',source),('affine_test',an),('syn_test',sn)]:
                for coord,val in zip('xyz',xyz):row[f'{prefix}_{coord}_mm']=float(val)
        except Exception as exc:
            row['failure_reason']=f'{type(exc).__name__}: {exc}'
        out.append(row)
        if len(out)%30==0:
            atomic_csv(a.output_root/'02_localization_errors.csv',out,EVAL_FIELDS)
            print(f'Evaluated {len(out)} / {len(configs)*366}',flush=True)
    atomic_csv(a.output_root/'02_localization_errors.csv',out,EVAL_FIELDS)
    summary={'created_utc':datetime.now(timezone.utc).isoformat(),
             'status':'PROVISIONAL_REGISTRATION_DEPENDENT',
             'expected':len(configs)*366,'counts':dict(Counter(x['status'] for x in out)),
             'method_counts':{m:dict(Counter(x['status'] for x in out if x['method']==m)) for m in configs},
             'limitations':['Registration-dependent and RAS frame provisional',
             'Known hemisphere restriction like Script 09','Frozen parameters tuned on same subjects']}
    (a.output_root/'02_localization_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    if any(r['status']!='PASS_PROVISIONAL' for r in out):raise RuntimeError('Evaluation failures saved to CSV: inspect failure_reason')
    return out


def analyze(a,configs):
    inp=a.output_root/'02_localization_errors.csv'
    if not inp.exists():raise FileNotFoundError('Run --phase evaluate first')
    df=pd.read_csv(inp)
    if len(df)!=366*len(configs):raise ValueError('Incomplete evaluation CSV')
    summary=[]
    for m in configs:
        subset=df[df.method==m].copy()
        if len(subset)!=366:raise ValueError('Missing method '+m)
        folder=a.output_root/'sensitivity'/m
        cmd=[sys.executable,str(a.script33),'--input',str(a.output_root/f'_temporary_{m}_sensitivity_input.csv'),
             '--output-root',str(folder),'--bootstrap',str(a.bootstrap)]
        subset.to_csv(cmd[cmd.index('--input')+1],index=False)
        try:
            subprocess.run(cmd,check=True)
        finally:
            Path(cmd[cmd.index('--input')+1]).unlink(missing_ok=True)
        row={'method':m,'n':len(subset),'pass':int(subset.status.eq('PASS_PROVISIONAL').sum())}
        for field in ['within_subject_error_mm','cross_affine_error_mm','cross_syn_error_mm','runtime_seconds']:
            vals=pd.to_numeric(subset.loc[subset.status=='PASS_PROVISIONAL',field],errors='coerce')
            row[field+'_median']=float(vals.median());row[field+'_mean']=float(vals.mean())
        summary.append(row)
        for filename in ['per_eeg_run_reference_sensitivity.csv','per_test_subject_sensitivity.csv',
                         'per_reference_subject_summary.csv','per_reference_test_pair_summary.csv',
                         'subject_cluster_bootstrap.csv']:
            frame=pd.read_csv(folder/filename)
            frame.insert(0,'method',m)
            target=a.output_root/filename
            if m==list(configs)[0]:frame.to_csv(target,index=False)
            else:frame.to_csv(target,index=False,mode='a',header=False)
    pd.DataFrame(summary).to_csv(a.output_root/'08_method_comparison.csv',index=False)
    return summary


def main():
    a=args_parse();a.dataset=a.dataset.expanduser()
    methods=tuple(m for m in METHODS if not a.method or m in a.method)
    if a.limit<0:raise ValueError('--limit must be nonnegative')
    if a.phase in ('evaluate','analyze','all') and a.limit:raise ValueError('--limit only allowed with reconstruct')
    configs=config_table(a.selected_configs,methods)
    jobs=schedule_jobs(a.schedule)
    manifest={'version':1,'methods':configs,'jobs':len(jobs),'schedule_sha256':hashlib.sha256(a.schedule.read_bytes()).hexdigest(),
              'selected_configs_sha256':hashlib.sha256(a.selected_configs.read_bytes()).hexdigest() if a.selected_configs else 'embedded_verified_user_csv',
              'dataset':str(a.dataset),'implementation':'Script09 helpers; Script32 spatial; Script33 sensitivity'}
    print('Methods:',list(configs),'Cross fits:',len(configs)*366,'Within fits:',len(configs)*61,
          'Total:',len(configs)*len(jobs))
    if a.phase=='plan':
        print(json.dumps(manifest,indent=2));return
    a.output_root.mkdir(parents=True,exist_ok=True)
    man=a.output_root/'frozen_configuration.json'
    if man.exists():
        if json.loads(man.read_text())!=manifest:raise ValueError('Manifest differs from prior run; use a new output folder')
        if a.phase in ('reconstruct','all') and not a.resume:raise ValueError('Output already exists. Add --resume')
    else:man.write_text(json.dumps(manifest,indent=2)+'\n')
    status_path=a.output_root/'01_execution_status.csv'
    alljobs=[(m,t,ref,run) for m in methods for t,ref,run in jobs]
    if a.phase in ('reconstruct','all'):
        script09=module(a.script09,'localizemi34_script09')
        # Fixed nominal montage derived from ref sub-01 forward, exactly Script 09.
        forward_cache={}
        template=get_forward(a.dataset,'sub-01',forward_cache)
        names,xyz,nominal=script09.fixed_spatial_subsets(template)
        past=existing(status_path)
        keyed={(r['method'],r['test_subject'],r['reference_subject'],r['run']):r for r in past}
        if len(keyed)!=len(past):raise ValueError('Duplicate prior results')
        if set(keyed)-set(alljobs):raise ValueError('Existing rows not in frozen job list')
        new=0
        for job in alljobs:
            old=keyed.get(job)
            if old is not None:continue  # preserve FAIL as recorded, never quietly overwrite
            row=fit(a,job,configs,script09,forward_cache,nominal,(names,xyz))
            keyed[job]=row;new+=1
            atomic_csv(status_path,[keyed[k] for k in alljobs if k in keyed],FIELDS)
            print(f'{len(keyed)}/{len(alljobs)} {job} {row["status"]} {row["runtime_seconds"]:.2f}s '+row['failure_reason'][:150],flush=True)
            if a.limit and new>=a.limit:break
        print('Reconstruction status:',len(keyed),'/',len(alljobs),'rows; new:',new)
        if a.phase=='reconstruct':return
    if a.phase in ('evaluate','all'):
        s32=module(a.script32,'localizemi34_script32')
        # s32 expects these file root names and QA settings.
        a.affine_root=Path('cross_subject/outputs/09_anatomical_registration')
        a.syn_root=Path('cross_subject/outputs/19_syn_registration')
        a.ants_bin=None;a.image_qa_samples=200;a.max_image_rmse=.015;a.max_roundtrip_mm=.1
        source_rows=existing(status_path)
        evaluate(a,configs,s32,source_rows,jobs)
    if a.phase in ('analyze','all'):
        analyze(a,configs)
        print('Finished. Outputs:',a.output_root)

if __name__=='__main__':main()

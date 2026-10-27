#!/usr/bin/env python3
"""Script 36: irMxNE / Gamma-MAP / Champagne cross-subject comparison.

Reuse Script 23 solver implementations and Script 34's registration evaluation.
1) Load selected configuration definitions from Script 26 (no reselection).
2) Reconstruct 61 within-subject per method in separate Script 36 outputs.
3) Run --phase audit-within (61 results matched to Script 23).
4) Only after successful reproduction, reconstruct 366 cross-subject per method.
5) Evaluate affine and SyN, analyze reference sensitivity.

IMPORTANT: Defaults to the final Script 26 selection-definitions CSV.
No automated reselection by alpha-group or failure rates. Validate against Script 23.

No modifications to Script 34, 35, or SESAME outputs.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import mne
import numpy as np
import pandas as pd

PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT/'src'))
from localize_mi import load_run
from localize_mi.inverse import (apply_average_reference, estimate_noise_covariance,
                                  create_target_evoked, create_inverse_operator)
from localize_mi.metrics import load_stimulation_info

METHODS=('irMxNE','Gamma-MAP','Champagne')
# TENTATIVE placeholders used only to show expected config keys.
# They are NEVER used for fitting without --selected-configs and are replaced by its rows.
FROZEN={
 'irMxNE':{'montage':'128','irmxne_alpha':30.0,'irmxne_loose':0.1,
           'irmxne_depth':1.0,'irmxne_iterations':10,'irmxne_random_state':0},
 'Gamma-MAP':{'montage':'64','gamma_alpha':0.4,'gamma_loose':0.5,
              'gamma_depth':0.5,'gamma_xyz_same_gamma':True,
              'gamma_maxit':2000,'gamma_tol':1e-4,'gamma_update_mode':1},
 'Champagne':{'montage':'32','champagne_alpha':'auto','champagne_prune':True,
              'champagne_pruning_thresh':0.001,
              'champagne_convergence_criterion':1e-8,'champagne_max_iter':2000},
}
EXTRA=['irmxne_alpha','irmxne_loose','irmxne_depth','irmxne_iterations',
       'irmxne_random_state','gamma_alpha','gamma_loose','gamma_depth',
       'gamma_xyz_same_gamma','gamma_maxit','gamma_tol','gamma_update_mode',
       'champagne_alpha','champagne_prune','champagne_pruning_thresh',
       'champagne_convergence_criterion','champagne_max_iter',
       'champagne_noise_covariance_supplied','champagne_make_operator_signature',
       'solver_notes']


def parser():
 p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
 p.add_argument('--phase',choices=['plan','parameter-audit','reconstruct','audit-within','evaluate','analyze','config-check','all'],default='plan')
 p.add_argument('--method',choices=METHODS,action='append')
 p.add_argument('--resume',action='store_true')
 p.add_argument('--limit',type=int,default=0,help='Maximum NEW fits this run')
 p.add_argument('--within-only',action='store_true',help='Only 61 within-subject fits per selected method')
 p.add_argument('--cross-only',action='store_true',help='Only 366 cross-subject fits per selected method')
 p.add_argument('--quiet',action='store_true')
 p.add_argument('--bootstrap',type=int,default=3000)
 p.add_argument('--dataset',type=Path,default=Path.home()/'Documents'/'DATASETS SOURCE LOCALIZATION'/'Localize-MI')
 p.add_argument('--schedule',type=Path,default=Path('cross_subject/outputs/30_all_reference_sesame_batch/experiment_schedule_all_references.csv'))
 p.add_argument('--output-root',type=Path,default=Path('cross_subject/outputs/36_cross_subject_irmxne_gamma_champagne'))
 p.add_argument('--script09',type=Path,default=Path('scripts/09_full_parameter_grid.py'))
 p.add_argument('--script23',type=Path,default=Path('scripts/23_full_next_method_grid.py'))
 p.add_argument('--script32',type=Path,default=Path('cross_subject/scripts/32_all_reference_localization_evaluation.py'))
 p.add_argument('--script33',type=Path,default=Path('cross_subject/scripts/33_reference_subject_sensitivity.py'))
 p.add_argument('--script34',type=Path,default=Path('cross_subject/scripts/34_cross_subject_minimum_norm.py'))
 p.add_argument('--registration-status',type=Path,default=Path('cross_subject/outputs/31_all_reference_registration/registration_status.csv'))
 p.add_argument('--selected-configs',type=Path,default=Path('outputs/26_twelve_methods_comparison/complete_all_pass/tables/01_selected_configuration_definitions.csv'),help='Script 26 final configuration definitions CSV')
 p.add_argument('--grid-root',type=Path,default=Path('outputs/23_full_next_method_grid'))
 p.add_argument('--existing-root',type=Path,action='append',default=[],help='Existing Script 36 output folder(s), repeat this flag for multiple roots')
 p.add_argument('--comparison-output',type=Path,default=Path('cross_subject/outputs/36_configuration_check'),help='Separate read-only comparison report output folder')

 return p.parse_args()


def load_selected_configs(path, methods):
 """Load EXACT Script 26 selected definitions; no re-selection or guessed values.

 Script 26 outputs configuration_parameters as a serialized JSON object.
 Old flat CSV/JSON inputs are also accepted for backward compatibility.
 """
 if path is None or not path.is_file():
  raise FileNotFoundError(f'Script 26 final selections missing: {path}. Supply --selected-configs PATH')
 if path.suffix.lower()=='.json':
  doc=json.loads(path.read_text())
  obj=doc.get('methods',doc)
  df=pd.DataFrame([dict(method=k,**v) for k,v in obj.items() if isinstance(v,dict)])
 else:
  df=pd.read_csv(path)
 if 'method' not in df:raise ValueError(f'{path}: no method column')
 expected={
  'irMxNE':('montage','irmxne_alpha','irmxne_loose','irmxne_depth'),
  'Gamma-MAP':('montage','gamma_alpha','gamma_loose','gamma_depth','gamma_xyz_same_gamma'),
  'Champagne':('montage','champagne_pruning_thresh'),
 }
 selected={}
 for method in methods:
  g=df[df.method.astype(str).str.casefold()==method.casefold()]
  if len(g)!=1:
   raise ValueError(f'{path}: expected exactly one final selection for {method}, got {len(g)}')
  r=g.iloc[0].to_dict()
  if 'selected_pass' in r and 'selected_fail' in r:
   if int(r['selected_pass']) != 61 or int(r['selected_fail']) != 0:
    raise ValueError(f'{method} not 61/61 ALL-PASS: {r["selected_pass"]}/{r["selected_fail"]}')
  if pd.notna(r.get('configuration_parameters')):
   raw=r['configuration_parameters']
   obj=json.loads(raw) if isinstance(raw,str) else raw
   if not isinstance(obj,dict):raise ValueError(f'{method}: configuration_parameters is not a JSON object')
   r={**r,**obj}
  cfg=dict(FROZEN[method])
  for key in expected[method]:
   if key not in r or pd.isna(r[key]):
    raise ValueError(f'{method}: Script 26 selection lacks {key}; fields={list(r)}')
  for key,default in cfg.items():
   if key not in r or pd.isna(r[key]):continue
   value=r[key]
   if isinstance(default,bool):
    if str(value).lower() not in ('true','false'):
     raise ValueError(f'{method}.{key} invalid bool: {value}')
    cfg[key]=str(value).lower()=='true'
   elif isinstance(default,str):cfg[key]=str(value)
   elif isinstance(default,int):cfg[key]=int(value)
   else:cfg[key]=float(value)
  cfg['montage']=str(cfg['montage']).removesuffix('.0')
  if cfg['montage'] not in ('32','64','128','all_good'):
   raise ValueError(f'{method}: unsupported montage {cfg["montage"]}')
  selected[method]=cfg
 return selected


def scalar_peak(stc,forward,hemisphere):
 if not hasattr(stc,'get_peak'):
  raise ValueError('Method returned no source time course')
 if not np.isfinite(stc.data).all() or stc.data.size==0 or not np.any(np.abs(stc.data)>0):
  raise ValueError('Empty, nonfinite or all-zero source estimate')
 vertex,tm=stc.get_peak(hemi=hemisphere,mode='abs')
 hi=0 if hemisphere=='lh' else 1
 head=np.asarray(forward['src'][hi]['rr'][int(vertex)],float)
 if head.shape!=(3,) or not np.isfinite(head).all():raise ValueError('Invalid cortical vertex')
 return head,int(vertex),float(tm)


def fit(a,job,cfg,s09,s23,s34,cache,nominal,template_info):
 method,test,ref,run=job
 t0=time.perf_counter()
 row={col:'' for col in s34.FIELDS+EXTRA}
 row.update(method=method,test_subject=test,reference_subject=ref,run=run,
            is_within=(test==ref),status='FAIL',montage=cfg['montage'],
            loose=cfg.get('irmxne_loose',cfg.get('gamma_loose','')),
            depth=cfg.get('irmxne_depth',cfg.get('gamma_depth','')),
            snr='',lambda2='')
 row.update(cfg)
 try:
  loaded=load_run(a.dataset,test,task='seegstim',run=run)
  forward=s34.get_forward(a.dataset,ref,cache)
  stim=load_stimulation_info(loaded.metadata['Description'],a.dataset/'derivatives'/'epochs'/test/'ieeg'/f'{test}_task-seegstim_space-surface_electrodes.tsv')
  row.update(stimulation_pair=stim.pair,hemisphere=stim.hemisphere,n_epochs=len(loaded.epochs))
  good=[c for c in loaded.epochs.ch_names if c not in loaded.epochs.info['bads']]
  actual,nrepl=s34.selected_channels(s09,forward,good,nominal,cfg['montage'],template_info)
  row.update(channels=len(actual),replacement_count=nrepl)
  eeg=s34.transplant_geometry(loaded.epochs,forward)
  # Script 23: physically keep good EEG channels BEFORE average reference.
  eeg.pick('eeg',exclude='bads')
  eeg=apply_average_reference(eeg)
  chosen=eeg.copy().pick(actual)
  if chosen.info['bads']:raise ValueError('Bad channels remain')
  cov=estimate_noise_covariance(chosen,-.250,-.050,method='auto',verbose=not a.quiet)
  evoked=create_target_evoked(chosen,-.002,.002)
  row['covariance_method_selected']=str(cov.get('method','auto'))
  if method=='irMxNE':
   if not hasattr(s23,'pilot') or not hasattr(s23.pilot,'run_irmxne'):
    raise RuntimeError('Script 23 missing pilot.run_irmxne; verify local Script 23 dependencies')
   stc,_=s23.pilot.run_irmxne(evoked,forward,cov,cfg['irmxne_alpha'],
       cfg['irmxne_loose'],cfg['irmxne_depth'],cfg['irmxne_iterations'],a.quiet)
  elif method=='Gamma-MAP':
   stc,_=s23.run_gamma(evoked,forward,cov,cfg,a.quiet)
  else:
   stc,signature,cov_supplied=s23.run_champagne(
       evoked,forward,cov,cfg['champagne_pruning_thresh'],a.quiet)
   row['champagne_make_operator_signature']=signature
   row['champagne_noise_covariance_supplied']=cov_supplied
   # Script 23 permits solver APIs without explicit noise_cov; preserve and audit metadata.
  head,peak_vertex,peak_time=scalar_peak(stc,forward,stim.hemisphere)
  if head.shape!=(3,) or not np.isfinite(head).all():raise ValueError('Invalid source coordinate')
  if np.linalg.norm(head)>1:raise ValueError('Expected source head coordinate in metres')
  row.update(status='PASS',failure_reason='',peak_vertex=peak_vertex,
      peak_time_s=peak_time,reference_head_x_mm=float(head[0]*1000),
      reference_head_y_mm=float(head[1]*1000),reference_head_z_mm=float(head[2]*1000))
 except Exception as exc:
  row['failure_reason']=f'{type(exc).__name__}: {exc}'
 row['runtime_seconds']=round(time.perf_counter()-t0,6)
 return row


def parameter_audit(a, methods):
 """Read-only parameter-group failure-rate audit using Script 23 grids."""
 report=[]
 for method in methods:
  slug={'irMxNE':'irmxne','Gamma-MAP':'gamma_map','Champagne':'champagne'}[method]
  path=a.grid_root/slug/'grid_results.csv'
  if not path.is_file():
   print(f'{method}: grid missing at {path}; locate original results first')
   continue
  df=pd.read_csv(path,low_memory=False)
  df=df[df.method==method].copy()
  field={'irMxNE':'irmxne_alpha','Gamma-MAP':'gamma_alpha',
         'Champagne':'champagne_pruning_thresh'}[method]
  if field not in df:raise ValueError(f'{path}: missing {field}')
  for value,g in df.groupby(field,dropna=False):
   n_pass=int(g.status.eq('PASS').sum())
   n_fail=int((~g.status.eq('PASS')).sum())
   report.append({'method':method,'parameter':field,'value':value,
                  'pass':n_pass,'fail':n_fail,'group_failure_free':n_fail==0,
                  'selected_group':np.isclose(float(value),float(FROZEN[method][field]))})
 tab=pd.DataFrame(report)
 if len(tab):
  print(tab.to_string(index=False))
  audit=a.output_root/'audit';audit.mkdir(parents=True,exist_ok=True)
  tab.to_csv(audit/'00_parameter_group_reliability.csv',index=False)
 print('Check that all extensions and failed rows are present in these grids.')
 print('Audit does not automatically change frozen configuration.')

def audit_within(a, methods):
 """Reproduce Script 26 *selected* 61-run baseline, sourced through Script 25.

 Script 23's single method/grid_results.csv is NOT authoritative: the
 selected settings can originate from extension experiments gathered by
 Script 25. Use the exact configuration_key from Script 26 definitions to
 select rows in Script 25's canonical complete_all_pass dataset.

 Compares MRI/surface-frame estimated coordinates to original selected_x/y/z,
 and recomputes localization errors against the original ground truth when
 those columns exist. No reconstruction files are modified.
 """
 import h5py
 original=Path('outputs/25_next_method_audit/07_complete_all_pass.csv')
 definitions=a.selected_configs
 status=a.output_root/'01_execution_status.csv'
 for path in (original,definitions,status):
  if not path.is_file():raise FileNotFoundError(path)
 if 'complete_all_pass' not in str(a.selected_configs):
  raise ValueError('Audit expects --selected-configs from Script 26 complete_all_pass')
 oldall=pd.read_csv(original,low_memory=False)
 defs=pd.read_csv(definitions,low_memory=False)
 d=pd.read_csv(status,low_memory=False)
 records=[]
 for method in methods:
  choice=defs[defs.method.eq(method)]
  if len(choice)!=1:raise ValueError(f'{method}: expected one Script 26 definition, got {len(choice)}')
  if int(choice.iloc[0]['selected_pass'])!=61 or int(choice.iloc[0]['selected_fail'])!=0:
   raise ValueError(f'{method}: selected configuration not 61/61 PASS in Script 26')
  key=str(choice.iloc[0]['configuration_key'])
  # Script 25 retains original solver IDs (gamma_map/irmxne/champagne);
  # Script 26 displays Gamma-MAP/irMxNE/Champagne.
  # Configuration_key is the canonical identity; method spelling is normalized.
  def canonical(name):
   return str(name).lower().replace('-', '').replace('_', '').replace(' ', '')
  old=oldall[
   (oldall.method.map(canonical)==canonical(method)) &
   (oldall.configuration_key.astype(str).str.strip()==key.strip()) &
   (oldall.status.astype(str).str.upper()=='PASS')
  ].copy()
  mine=d[(d.method==method)&(d.is_within.astype(str).str.lower()=='true')].copy()
  if len(old)!=61 or len(mine)!=61 or not mine.status.eq('PASS').all():
   available=oldall.loc[oldall.method.map(canonical)==canonical(method),'configuration_key'].astype(str)
   exact_key_count=int((oldall.configuration_key.astype(str).str.strip()==key.strip()).sum())
   raise ValueError(
    f'{method}: Script25 matched={len(old)}, Script36 within={len(mine)}; '
    f'key matches regardless of method={exact_key_count}; '
    f'available config keys for method={available.nunique()}. '
    'If Script25 matched=0, verify Script25 and Script26 were generated '
    'from the same audited input and cohort.'
   )
  joined=old.merge(mine,left_on=['subject','run'],right_on=['test_subject','run'],validate='one_to_one',suffixes=('_old','_new'))
  if len(joined)!=61:raise ValueError(f'{method}: matched {len(joined)} instead of 61')
  for _,r in joined.iterrows():
   subject=str(r['subject']);run=str(r['run'])
   transform=a.dataset/'derivatives'/'sourcemodelling'/subject/'xfm'/f'{subject}_from-head_to-surface.h5'
   with h5py.File(transform) as f:matrix=np.asarray(f['trans'][()],float)
   head=np.array([r[f'reference_head_{x}_mm'] for x in 'xyz'],float)/1000
   xyz=(matrix@np.r_[head,1])[:3]*1000
   out={'method':method,'subject':subject,'run':run,'configuration_key':key}
   if all(f'selected_{x}_mm' in joined for x in 'xyz'):
    prev=np.array([r[f'selected_{x}_mm'] for x in 'xyz'],float)
    out['position_diff_mm']=float(np.linalg.norm(xyz-prev))
   else:out['position_diff_mm']=np.nan
   if all(f'ground_truth_{x}_mm' in joined for x in 'xyz'):
    truth=np.array([r[f'ground_truth_{x}_mm'] for x in 'xyz'],float)
    out['new_error_mm']=float(np.linalg.norm(xyz-truth))
    error_col=next((c for c in ['localization_error_mm','localization_distance_mm','selected_localization_error_mm'] if c in joined),None)
    if error_col is not None:
     out['old_error_mm']=float(r[error_col])
     out['error_diff_mm']=out['new_error_mm']-out['old_error_mm']
   oldvertex='peak_vertex_old' if 'peak_vertex_old' in joined else 'peak_vertex'
   newvertex='peak_vertex_new' if 'peak_vertex_new' in joined else None
   if oldvertex in joined and newvertex:
    out['vertex_match']=bool(pd.notna(r[oldvertex]) and pd.notna(r[newvertex]) and int(r[oldvertex])==int(r[newvertex]))
   t='selected_time_ms' if 'selected_time_ms' in joined else None
   if t and 'peak_time_s' in joined:
    out['time_diff_ms']=float(r['peak_time_s']*1000-r[t])
   records.append(out)
 tab=pd.DataFrame(records)
 audit=a.output_root/'audit';audit.mkdir(parents=True,exist_ok=True)
 dest=audit/'01_within_reproduction.csv';tab.to_csv(dest,index=False)
 for method,g in tab.groupby('method'):
  print(f'{method}: matched={len(g)}/61')
  if g.position_diff_mm.notna().any():
   print(f'  positions <=0.05mm: {(g.position_diff_mm<=.05).sum()}/61; max={g.position_diff_mm.max():.6f}mm')
  else:print('  Original peak coordinates missing in Script 25 canonical file; position audit unavailable')
  if 'error_diff_mm' in g:
   print(f'  errors <=0.05mm: {(g.error_diff_mm.abs()<=.05).sum()}/61; max delta={g.error_diff_mm.abs().max():.6f}mm')
   print(f'  previous median={g.old_error_mm.median():.6f}mm; new median={g.new_error_mm.median():.6f}mm')
  if 'vertex_match' in g:print(f'  vertex matching: {g.vertex_match.sum()}/61')
  if 'time_diff_ms' in g:print(f'  peak time <=0.001ms: {(g.time_diff_ms.abs()<=.001).sum()}/61')
 print('Audit saved:',dest)
 if not tab.position_diff_mm.notna().all():
  raise RuntimeError('Original coordinates missing: audit is INCOMPLETE, not PASS; inspect audit CSV')
 if not (tab.position_diff_mm<=.05).all():
  raise RuntimeError('Reproduction positions differ; inspect audit CSV before running cross-subject')
 if 'error_diff_mm' in tab and not (tab.error_diff_mm.abs()<=.05).all():
  raise RuntimeError('Reproduction errors differ; inspect audit CSV before running cross-subject')
 if 'time_diff_ms' in tab and not (tab.time_diff_ms.abs()<=.001).all():
  raise RuntimeError('Reproduction peak times differ; inspect audit CSV')
 print('WITHIN-SUBJECT REPRODUCTION AUDIT PASSED for available metrics')


def config_check(a, selected):
 """Compare CURRENT Script 26 all-pass selection with previous Script 36 folders.

 No reconstructions, past CSVs, or frozen manifests are edited.
 Different settings => never mix old rows with the new configuration.
 """
 roots=a.existing_root or [
  Path('cross_subject/outputs/36_cross_subject_irmxne_gammamap_champagne'),
  Path('cross_subject/outputs/36_complete_all_pass'),
  Path('cross_subject/outputs/36_cross_subject_irmxne_gamma_champagne'),
 ]
 defs=pd.read_csv(a.selected_configs)
 rows=[]
 for method in METHODS:
  dr=defs[defs.method.astype(str).str.casefold()==method.casefold()]
  if len(dr)!=1:raise ValueError(f'Expected one Script26 row for {method}')
  n_pass=int(dr.iloc[0]['selected_pass']);n_fail=int(dr.iloc[0]['selected_fail'])
  if n_pass!=61 or n_fail!=0:
   raise ValueError(f'{method} selection is not 61/61 all-pass: {n_pass} PASS, {n_fail} FAIL')
  new=selected[method]
  print(f'\n{method} -- NEW Script26 all-pass: {json.dumps(new,sort_keys=True)}')
  found=False
  for root in roots:
   mp=root/'frozen_configuration.json'
   status=root/'01_execution_status.csv'
   if not mp.exists() and not status.exists():continue
   if not mp.is_file():
    print(f'  {root}: STATUS WITHOUT FROZEN MANIFEST; cannot safely reuse')
    rows.append(dict(method=method,root=str(root),same_config=False,completed=0,within=0,cross=0,pass_count=0,fail_count=0,action='INSPECT: missing frozen manifest'))
    continue
   document=json.loads(mp.read_text())
   prior=document.get('methods',{}).get(method)
   if not isinstance(prior,dict):continue
   found=True
   # Compare just the effective method inputs, NOT entire CSV file hash or other methods.
   old={k:prior.get(k) for k in new}
   same=all((str(old[k]).lower()==str(new[k]).lower()) if isinstance(new[k],(str,bool))
            else np.isclose(float(old[k]),float(new[k]),rtol=0,atol=1e-12)
            for k in new if old.get(k) is not None) and all(k in prior for k in new)
   if status.is_file():
    frame=pd.read_csv(status,low_memory=False)
    frame=frame.loc[frame.method.astype(str).eq(method)].copy()
    if frame.duplicated(['method','test_subject','reference_subject','run']).any():
     raise ValueError(f'{root} has duplicate {method} results')
    mask=frame.is_within.astype(str).str.lower().eq('true') if 'is_within' in frame else pd.Series(False,index=frame.index)
    inside=int(mask.sum());outside=int((~mask).sum());passed=int(frame.status.eq('PASS').sum());failed=int(frame.status.eq('FAIL').sum())
   else:inside=outside=passed=failed=0
   count=inside+outside
   action=('REUSE existing fits; run only uncompleted cases in this original root' if same else
           f'RERUN {count} completed fits with NEW config in a NEW output root (and complete remaining scheduled jobs)')
   rows.append(dict(method=method,root=str(root),same_config=same,completed=count,within=inside,cross=outside,pass_count=passed,fail_count=failed,action=action))
   print(f'  {root}: same_settings={same}, within={inside}, cross={outside}, PASS={passed}, FAIL={failed}')
   if not same:
    print('    Changed keys:',{k:(old.get(k),new.get(k)) for k in new if str(old.get(k)).lower()!=str(new.get(k)).lower()})
  if not found:print('  No prior matching method manifests found; schedule 61 within + 366 cross fits')
 dest=a.comparison_output;dest.mkdir(parents=True,exist_ok=True)
 report=pd.DataFrame(rows,columns=['method','root','same_config','completed','within','cross','pass_count','fail_count','action'])
 report.to_csv(dest/'01_method_config_reuse_plan.csv',index=False)
 print('\nComparison report:',dest/'01_method_config_reuse_plan.csv')
 print('IMPORTANT: A configuration change requires a fresh method-specific output root.')
 print('Unchanged configuration + different whole-CSV hash may still be rejected by the old manifest check; do not bypass it or manually overwrite old results.')
 print('Existing FAIL runs are never retried by --resume; investigate failures separately.')


def main():
 a=parser();a.dataset=a.dataset.expanduser()
 if a.limit<0:raise ValueError('--limit must be non-negative')
 if a.within_only and a.cross_only:raise ValueError('Choose only one of --within-only/--cross-only')
 if a.phase not in ('reconstruct','plan') and (a.within_only or a.cross_only or a.limit):
  raise ValueError('Within/cross/limit filters are for reconstruction only')
 methods=tuple(m for m in METHODS if not a.method or m in a.method)
 if a.phase != 'parameter-audit':
  selections=load_selected_configs(a.selected_configs,METHODS)
  FROZEN.update(selections)
 from importlib.util import spec_from_file_location,module_from_spec
 def import_script(path,name):
  if not path.is_file():raise FileNotFoundError(f'Missing dependency: {path}')
  spec=spec_from_file_location(name,path);mod=module_from_spec(spec)
  sys.modules[name]=mod;spec.loader.exec_module(mod);return mod
 s34=import_script(a.script34,'localizemi34_for_36')
 jobs=s34.schedule_jobs(a.schedule)
 configs={m:dict(FROZEN[m]) for m in methods}
 all_configs={m:dict(FROZEN[m]) for m in METHODS}
 sha=hashlib.sha256(a.selected_configs.read_bytes()).hexdigest() if a.selected_configs else 'NOT_SUPPLIED'
 manifest={'version':1,'methods':all_configs,'n_jobs':len(jobs),
           'schedule_sha256':hashlib.sha256(a.schedule.read_bytes()).hexdigest(),
           'selected_config_sha256':sha,'dataset':str(a.dataset),
           'implementation':'Script23 solvers; Script34 evaluation; within reproduction gate'}
 print(f'{len(methods)} methods; {len(methods)*61} within, {len(methods)*366} cross, {len(methods)*len(jobs)} fits')
 if a.phase=='config-check':return config_check(a,selections)
 if a.phase=='plan':
  print(json.dumps(manifest,indent=2))
  print('Final Script 25/26 fixed settings loaded; no reselection performed.')
  return
 if a.phase=='parameter-audit':return parameter_audit(a,methods)
 if a.phase=='audit-within':return audit_within(a,methods)

 a.output_root.mkdir(parents=True,exist_ok=True)
 mp=a.output_root/'frozen_configuration.json'
 if mp.exists():
  previous=json.loads(mp.read_text())
  if previous!=manifest:
   # An updated Script 26 file may change OTHER methods while the selected
   # method remains identical.  Permit method-filtered continuation only.
   # Never replace the historical manifest, or silently mix changed settings.
   if not a.method or a.phase=='all':
    raise ValueError('Frozen manifest differs. Specify --method for unchanged settings, or use a NEW output root.')
   for k in ('schedule_sha256','dataset','implementation','n_jobs'):
    if previous.get(k)!=manifest.get(k):
     raise ValueError(f'Frozen manifest {k} differs: use a NEW output root')
   for method in methods:
    if previous.get('methods',{}).get(method)!=manifest['methods'][method]:
     raise ValueError(f'{method} configuration CHANGED. Use a NEW output root; do not --resume here.')
   print('Existing selected-method settings match; other methods/CSV hash changed.')
   print('Continuing ONLY the requested methods; existing frozen manifest remains unchanged.')
  if a.phase in ('reconstruct','all') and not a.resume:raise ValueError('Output already exists. Add --resume')
 else:
  if a.phase not in ('reconstruct','plan','all'):
   raise ValueError('No frozen manifest exists for evaluation/analysis. Run reconstruction first.')
  mp.write_text(json.dumps(manifest,indent=2)+'\n')
 status=a.output_root/'01_execution_status.csv'
 all_jobs=[(m,t,r,run) for m in METHODS for t,r,run in jobs]
 run_jobs=[(m,t,r,run) for m in methods for t,r,run in jobs
          if (not a.within_only or t==r) and (not a.cross_only or t!=r)]
 columns=s34.FIELDS+EXTRA
 if a.phase in ('reconstruct','all'):
  s09=import_script(a.script09,'localizemi09_for_36')
  s23=import_script(a.script23,'localizemi23_for_36')
  cache={};template=s34.get_forward(a.dataset,'sub-01',cache)
  names,xyz,nominal=s09.fixed_spatial_subsets(template)
  past=s34.existing(status)
  seen={(str(x['method']),str(x['test_subject']),str(x['reference_subject']),str(x['run'])):x for x in past}
  if len(seen)!=len(past):raise ValueError('Duplicate prior reconstruction rows')
  if set(seen)-set(all_jobs):raise ValueError('Unexpected job in status')
  new=0
  for job in run_jobs:
   if job in seen:continue  # Never silently overwrite FAIL or PASS
   r=fit(a,job,configs[job[0]],s09,s23,s34,cache,nominal,(names,xyz))
   seen[job]=r;new+=1
   s34.atomic_csv(status,[seen[j] for j in all_jobs if j in seen],columns)
   print(f'{len(seen)}/{len(all_jobs)} {job} {r["status"]} {r["runtime_seconds"]:.2f}s {str(r["failure_reason"])[:160]}',flush=True)
   if a.limit and new>=a.limit:break
  print('Saved:',status,'rows:',len(seen),'new:',new,'PASS:',sum(x['status']=='PASS' for x in seen.values()))
  if a.phase=='reconstruct':return
 if a.phase in ('evaluate','all'):
  s32=import_script(a.script32,'localizemi32_for_36')
  a.affine_root=Path('cross_subject/outputs/09_anatomical_registration')
  a.syn_root=Path('cross_subject/outputs/19_syn_registration')
  a.ants_bin=None;a.image_qa_samples=200;a.max_image_rmse=.015;a.max_roundtrip_mm=.1
  prev=s34.EVAL_FIELDS
  s34.EVAL_FIELDS=prev+EXTRA
  try:s34.evaluate(a,configs,s32,[r for r in s34.existing(status) if r['method'] in methods],jobs)
  finally:s34.EVAL_FIELDS=prev
 if a.phase in ('analyze','all'):
  s34.analyze(a,configs)
  print('Completed:',a.output_root)

if __name__=='__main__':main()

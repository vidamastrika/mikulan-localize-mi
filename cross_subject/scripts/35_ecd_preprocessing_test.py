#!/usr/bin/env python3
"""Script 35 — Fixed cross-subject Continuous-ECD, RAP-MUSIC, LCMV, MxNE.

Run from the repository root. Requires Script 34, Script 32, Script 33 and
Script 09 at their established paths. Uses all 42 existing registrations.

    cd ~/projects/mikulan-localize-mi
    source .venv/bin/activate

    # Inspect all 4 x (61 within + 366 cross) jobs, without fitting
    python cross_subject/scripts/35_cross_subject_ecd_music_lcmv_mxne.py --phase plan

    # ONE-TIME transition from older alpha-50 results to alpha 35:
    # python cross_subject/scripts/35_cross_subject_ecd_music_lcmv_mxne.py --reset-mxne-alpha50

    # Pilot one method at a time (strongly recommended)
    python cross_subject/scripts/35_cross_subject_ecd_music_lcmv_mxne.py --phase reconstruct --method Continuous-ECD --limit 3 --resume --quiet
    python cross_subject/scripts/35_cross_subject_ecd_music_lcmv_mxne.py --phase reconstruct --method RAP-MUSIC --limit 3 --resume --quiet
    python cross_subject/scripts/35_cross_subject_ecd_music_lcmv_mxne.py --phase reconstruct --method LCMV --limit 3 --resume --quiet
    python cross_subject/scripts/35_cross_subject_ecd_music_lcmv_mxne.py --phase reconstruct --method MxNE --limit 3 --resume --quiet

    # Complete the fits; --resume preserves original per-job runtimes
    python cross_subject/scripts/35_cross_subject_ecd_music_lcmv_mxne.py --phase reconstruct --resume --quiet

    # Registration/error evaluation requires ANTs
    micromamba run -n localizemi-registration .venv/bin/python \\
        cross_subject/scripts/35_cross_subject_ecd_music_lcmv_mxne.py --phase evaluate

    # Script 33-equivalent complete sensitivity tables and clustered bootstrap
    python cross_subject/scripts/35_cross_subject_ecd_music_lcmv_mxne.py --phase analyze

    # Optional all steps in one process (only after pilots)
    micromamba run -n localizemi-registration .venv/bin/python \\
        cross_subject/scripts/35_cross_subject_ecd_music_lcmv_mxne.py --phase all --resume --quiet

Files: cross_subject/outputs/35_cross_subject_ecd_music_lcmv_mxne/
  01_execution_status.csv                  1708 source fits, runtime per fit
  02_localization_errors.csv               1464 cross-subject evaluations
  08_method_comparison.csv                four-method summary
  sensitivity/<method>/*                  Script 33 reports for each method

Frozen parameters come from Script 17/16 selected-configuration CSV:
  Continuous-ECD: montage=128, min_dist=15 mm
  RAP-MUSIC:      montage=128, n_dipoles=1
  LCMV:           montage=32, reg=0, +/-10 ms data covariance,
                  method=diagonal_fixed, weight_norm=unit-noise-gain-invariant
  MxNE:           montage=128, alpha=35, loose=0.7, depth=1.0
                  (restricted failure-free alpha search)

IMPORTANT:
 - Pilot against original within-subject results BEFORE running all 1708 jobs.
 - Fitting uses only test EEG + reference forward/anatomy. The evaluation
   uses test MRI and stimulation truth only afterwards.
 - Known hemisphere is used when selecting cortical STC peaks; consistent
   with Script 09 but an evaluation-informed constraint.
 - Registration/RAS assumptions remain provisional.
 - Original published method implementations may differ in SSP, BEM, rank,
   dipole scoring, and MxNE solver parameters. If a within-subject reproduction
   audit fails, do NOT treat the batch as methodologically matched.
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

METHODS=('Continuous-ECD','RAP-MUSIC','LCMV','MxNE')
FROZEN={
 'Continuous-ECD':{'montage':'128','ecd_min_dist_mm':15.0},
 'RAP-MUSIC':{'montage':'128','rap_music_n_dipoles':1},
 'LCMV':{'montage':'32','lcmv_reg':0.0,'lcmv_window_half_ms':10.0,
         'lcmv_data_covariance_method':'diagonal_fixed',
         'lcmv_weight_norm':'unit-noise-gain-invariant'},
 'MxNE':{'montage':'128','mxne_alpha':35.0,'mxne_loose':0.7,'mxne_depth':1.0},
}
# Extra fields beyond Script34 are retained in the execution log.
EXTRA=['ecd_min_dist_mm','rap_music_n_dipoles','lcmv_reg','lcmv_window_half_ms',
       'lcmv_data_covariance_method','lcmv_weight_norm','mxne_alpha',
       'mxne_loose','mxne_depth','bem_solution_path','solver_notes']


def parser():
 p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
 p.add_argument('--phase',choices=['plan','reconstruct','evaluate','analyze','all'],default='plan')
 p.add_argument('--method',choices=METHODS,action='append')
 p.add_argument('--resume',action='store_true')
 p.add_argument('--reset-mxne-alpha50',action='store_true',
                help='One-time in-place migration: DELETE alpha-50 MxNE rows and stale analyses; preserve other three methods')
 p.add_argument('--limit',type=int,default=0,help='Limit NEW fitting jobs, not cached jobs')
 p.add_argument('--quiet',action='store_true')
 p.add_argument('--bootstrap',type=int,default=3000)
 p.add_argument('--dataset',type=Path,default=Path.home()/'Documents'/'DATASETS SOURCE LOCALIZATION'/'Localize-MI')
 p.add_argument('--schedule',type=Path,default=Path('cross_subject/outputs/30_all_reference_sesame_batch/experiment_schedule_all_references.csv'))
 p.add_argument('--output-root',type=Path,default=Path('cross_subject/outputs/35_cross_subject_ecd_music_lcmv_mxne'))
 p.add_argument('--script09',type=Path,default=Path('scripts/09_full_parameter_grid.py'))
 p.add_argument('--script32',type=Path,default=Path('cross_subject/scripts/32_all_reference_localization_evaluation.py'))
 p.add_argument('--script33',type=Path,default=Path('cross_subject/scripts/33_reference_subject_sensitivity.py'))
 p.add_argument('--script34',type=Path,default=Path('cross_subject/scripts/34_cross_subject_minimum_norm.py'))
 p.add_argument('--registration-status',type=Path,default=Path('cross_subject/outputs/31_all_reference_registration/registration_status.csv'))
 p.add_argument('--selected-configs',type=Path,default=None,help='Optional Script 17 CSV for strict config audit')
 p.add_argument('--bem-root',type=Path,default=Path('outputs/12_additional_methods_all_runs/bem'),
                help='Root containing per-reference *-bem-sol.fif for ECD')
 return p.parse_args()


def check_configs(csv_path,methods):
 if csv_path is None:return
 d=pd.read_csv(csv_path)
 for method in methods:
  g=d[d.method==method]
  if len(g)!=1:raise ValueError(f'Expected one selected config for {method}; found {len(g)}')
  r=g.iloc[0]
  for k,v in FROZEN[method].items():
   if k=='montage':
    if str(r[k])!=str(v):raise ValueError(f'{method}: {k} mismatch: {r[k]} != {v}')
   elif k=='lcmv_data_covariance_method' or k=='lcmv_weight_norm':
    if str(r[k])!=v:raise ValueError(f'{method}: {k} mismatch')
   elif not np.isclose(float(r[k]),float(v)):
    raise ValueError(f'{method}: {k} mismatch: {r[k]} != {v}')


def bem_for_reference(a,ref):
 roots=[a.dataset/'derivatives'/'sourcemodelling'/ref/'anat',
        a.dataset/'derivatives'/'sourcemodelling'/ref,
        a.dataset/'derivatives'/'epochs'/ref/'anat']
 if a.bem_root:roots.insert(0,a.bem_root)
 matches=[]
 for root in roots:
  if root.is_dir():
   matches.extend(root.rglob('*-bem-sol.fif'))
   matches.extend(root.rglob('*bem_solution.fif'))
  # skip unrelated subjects when searching broad override root
 matches=[x for x in dict.fromkeys(matches) if ref in str(x)]
 if len(matches)==0:
  raise FileNotFoundError(f'{ref}: no BEM solution FIF found. Provide --bem-root, or verify old Continuous-ECD BEM inputs.')
 if len(matches)>1:
  raise ValueError(f'{ref}: ambiguous BEM solutions: {[str(p) for p in matches]}')
 return matches[0]


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


def fit(a,job,cfg,s09,s34,cache,nominal,template_info):
 method,test,ref,run=job
 t0=time.perf_counter()
 row={col:'' for col in s34.FIELDS+EXTRA}
 row.update(method=method,test_subject=test,reference_subject=ref,run=run,
            is_within=test==ref,status='FAIL',montage=cfg['montage'],
            loose=cfg.get('mxne_loose',''),depth=cfg.get('mxne_depth',''),
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
  eeg = s34.transplant_geometry(loaded.epochs, forward)

  # Keep only good EEG channels before average referencing
  eeg.pick("eeg", exclude="bads")

  eeg = apply_average_reference(eeg)

  chosen = eeg.copy().pick(actual)
  if chosen.info['bads']:raise ValueError('Bad channels survive selection')
  cov=estimate_noise_covariance(chosen,-0.250,-0.050,method='auto',verbose=not a.quiet)
  evoked=create_target_evoked(chosen,-.002,.002)
  row['covariance_method_selected']=str(cov.get('method','auto'))
  if method=='Continuous-ECD':
   bempath=bem_for_reference(a,ref)
   row['bem_solution_path']=str(bempath)
   # MNE fits dipole positions in HEAD coordinates (metres).
   dip,_residual=mne.fit_dipole(evoked,cov,str(bempath),trans=forward.get('mri_head_t'),
                    min_dist=cfg['ecd_min_dist_mm'],verbose=not a.quiet)
   if len(dip.pos)==0 or not np.isfinite(dip.pos).all():raise ValueError('ECD returned no finite dipoles')
   idx=int(np.argmax(dip.gof))
   head=np.asarray(dip.pos[idx],float)
   peak_vertex=-1;peak_time=float(dip.times[idx])
   row['solver_notes']=f'ECD max GOF={float(dip.gof[idx]):.6f}%; dipole index={idx}'
  elif method=='RAP-MUSIC':
   dip=mne.beamformer.rap_music(evoked,forward,cov,n_dipoles=int(cfg['rap_music_n_dipoles']),verbose=not a.quiet)
   if isinstance(dip,(tuple,list)):
    candidates=list(dip)
    if not candidates:raise ValueError('RAP-MUSIC returned zero dipoles')
    dip=candidates[0]
   if len(dip.pos)==0 or not np.isfinite(dip.pos).all():raise ValueError('RAP-MUSIC returned no finite dipoles')
   idx=int(np.argmax(dip.gof)) if len(dip.gof) else 0
   head=np.asarray(dip.pos[idx],float)
   peak_vertex=-1;peak_time=float(dip.times[idx])
   row['solver_notes']=f'RAP-MUSIC dipole index={idx}'
  elif method=='LCMV':
   half=cfg['lcmv_window_half_ms']/1000.
   data_cov=mne.compute_covariance(chosen,tmin=-half,tmax=half,
                 method=cfg['lcmv_data_covariance_method'],verbose=not a.quiet)
   filters = mne.beamformer.make_lcmv(
    evoked.info,
    forward,
    data_cov,
    reg=cfg["lcmv_reg"],
    noise_cov=cov,
    pick_ori="max-power",
    weight_norm=cfg["lcmv_weight_norm"],
    rank=None,
    reduce_rank=True,
    verbose=not a.quiet,
)
   stc=mne.beamformer.apply_lcmv(evoked,filters,verbose=not a.quiet)
   head,peak_vertex,peak_time=scalar_peak(stc,forward,stim.hemisphere)
  else:
   stc=mne.inverse_sparse.mixed_norm(evoked,forward,cov,
                     alpha=cfg['mxne_alpha'],loose=cfg['mxne_loose'],
                     depth=cfg['mxne_depth'],verbose=not a.quiet)
   head,peak_vertex,peak_time=scalar_peak(stc,forward,stim.hemisphere)
  if head.shape!=(3,) or not np.isfinite(head).all():raise ValueError('Invalid head coordinate')
  if np.linalg.norm(head)>1:raise ValueError('Unexpected source coordinate scale; expected metres')
  row.update(status='PASS',failure_reason='',peak_vertex=peak_vertex,
             peak_time_s=peak_time,reference_head_x_mm=float(head[0]*1000),
             reference_head_y_mm=float(head[1]*1000),reference_head_z_mm=float(head[2]*1000))
 except Exception as ex:
  row['failure_reason']=f'{type(ex).__name__}: {ex}'
 row['runtime_seconds']=round(time.perf_counter()-t0,6)
 return row


def reset_previous_mxne(a, manifest):
 """Explicitly discard only OLD alpha-50 MxNE results; leave other methods intact.

 User-requested destructive migration. Not invoked during normal --resume.
 """
 root=a.output_root
 mp=root/'frozen_configuration.json'
 status=root/'01_execution_status.csv'
 if not mp.is_file():
  raise FileNotFoundError(f'Cannot migrate: missing {mp}')
 old=json.loads(mp.read_text())
 old_cfg=old.get('methods',{}).get('MxNE',{})
 new_cfg=manifest['methods']['MxNE']
 if old_cfg==new_cfg:
  print('MxNE manifest already uses alpha=35; nothing to reset.')
  return
 if old_cfg.get('mxne_alpha')!=50.0 or old_cfg.get('mxne_loose')!=0.8 or old_cfg.get('mxne_depth')!=1.0:
  raise ValueError('Refusing migration: existing MxNE configuration is not expected alpha=50/loose=0.8/depth=1.0')
 if not status.is_file():
  raise FileNotFoundError(f'Cannot migrate: missing {status}')
 rows=pd.read_csv(status,keep_default_na=False)
 if 'method' not in rows.columns:raise ValueError('Status file missing method column')
 removed=int((rows['method']=='MxNE').sum())
 if removed>427:raise ValueError(f'Refusing to remove unexpected {removed} MxNE rows')
 other=rows[rows['method']!='MxNE'].copy()
 # Do not touch the successful Continuous-ECD / RAP-MUSIC / LCMV data.
 tmp=status.with_name(status.name+'.tmp')
 other.to_csv(tmp,index=False)
 os.replace(tmp,status)
 tmp=mp.with_name(mp.name+'.tmp')
 tmp.write_text(json.dumps(manifest,indent=2)+'\n')
 os.replace(tmp,mp)
 # Old derived reports must not be mistaken for valid alpha-35 analyses.
 for name in ('02_localization_errors.csv','08_method_comparison.csv',
              'localization_summary.json','sensitivity_summary.json'):
  f=root/name
  if f.is_file():f.unlink()
 sensitivity=root/'sensitivity'
 if sensitivity.is_dir():
  import shutil
  shutil.rmtree(sensitivity)
 print(f'RESET COMPLETE: deleted {removed} alpha-50 MxNE rows; retained {len(other)} other-method rows.')
 print('Removed stale evaluation/sensitivity reports. Rerun reconstruct, evaluate, analyze.')


def main():
 a=parser();a.dataset=a.dataset.expanduser()
 if a.limit<0:raise ValueError('--limit cannot be negative')
 if a.phase in ('evaluate','analyze','all') and a.limit:
  raise ValueError('--limit only supported during reconstruct')
 methods=tuple(m for m in METHODS if not a.method or m in a.method)
 check_configs(a.selected_configs,methods)
 # Load Script34 logic for its existing validated evaluation/analyze tables.
 from importlib.util import spec_from_file_location,module_from_spec
 def import_script(path,name):
  if not path.is_file():raise FileNotFoundError(f'Missing dependency: {path}')
  spec=spec_from_file_location(name,path);mod=module_from_spec(spec)
  sys.modules[name]=mod;spec.loader.exec_module(mod);return mod
 s34=import_script(a.script34,'localizemi_script34_for_35')
 jobs=s34.schedule_jobs(a.schedule)
 configs={m:dict(FROZEN[m]) for m in methods}
 full_configs={m:dict(FROZEN[m]) for m in METHODS}
 selected_sha=hashlib.sha256(a.selected_configs.read_bytes()).hexdigest() if a.selected_configs else 'frozen_from_user_selected_configurations'
 manifest={'version':1,'methods':full_configs,'n_jobs':len(jobs),
           'schedule_sha256':hashlib.sha256(a.schedule.read_bytes()).hexdigest(),
           'selected_config_sha256':selected_sha,
           'dataset':str(a.dataset),'implementation':'MNE methods; Script34 evaluations; Script33 sensitivity'}
 print(f'{len(methods)} methods, {len(methods)*61} within, {len(methods)*366} cross, total {len(methods)*len(jobs)} fits')
 if a.reset_mxne_alpha50:
  reset_previous_mxne(a,manifest)
  return
 if a.phase=='plan':print(json.dumps(manifest,indent=2));return
 a.output_root.mkdir(parents=True,exist_ok=True)
 mp=a.output_root/'frozen_configuration.json'
 if mp.exists():
  if json.loads(mp.read_text())!=manifest:
   raise ValueError('Frozen manifest mismatch. Use original method args or a NEW output folder')
  if a.phase in ('reconstruct','all') and not a.resume:
   raise ValueError('Output already exists. Add --resume')
 else:mp.write_text(json.dumps(manifest,indent=2)+'\n')
 status=a.output_root/'01_execution_status.csv'
 all_jobs=[(m,t,r,run) for m in METHODS for t,r,run in jobs]
 run_jobs=[(m,t,r,run) for m in methods for t,r,run in jobs]
 columns=s34.FIELDS+EXTRA
 if a.phase in ('reconstruct','all'):
  s09=import_script(a.script09,'localizemi_script09_for_35')
  cache={};template=s34.get_forward(a.dataset,'sub-01',cache)
  names,xyz,nominal=s09.fixed_spatial_subsets(template)
  past=s34.existing(status)
  seen={(str(x['method']),str(x['test_subject']),str(x['reference_subject']),str(x['run'])):x for x in past}
  if len(seen)!=len(past):raise ValueError('Duplicate prior reconstruction rows')
  if set(seen)-set(all_jobs):raise ValueError('Status file has jobs not in selected method plan')
  new=0
  for job in run_jobs:
   if job in seen and str(seen[job].get('status'))=='PASS':continue
   method=job[0]
   r=fit(a,job,configs[method],s09,s34,cache,nominal,(names,xyz))
   seen[job]=r;new+=1
   s34.atomic_csv(status,[seen[j] for j in all_jobs if j in seen],columns)
   print(f'{len(seen)}/{len(all_jobs)} {job} {r["status"]} {r["runtime_seconds"]:.2f}s {str(r["failure_reason"])[:160]}',flush=True)
   if a.limit and new>=a.limit:break
  print('Saved:',status,'rows:',len(seen),'new:',new,'PASS:',sum(x['status']=='PASS' for x in seen.values()))
  if a.phase=='reconstruct':return
 if a.phase in ('evaluate','all'):
  s32=import_script(a.script32,'localizemi_script32_for_35')
  a.affine_root=Path('cross_subject/outputs/09_anatomical_registration')
  a.syn_root=Path('cross_subject/outputs/19_syn_registration')
  a.ants_bin=None;a.image_qa_samples=200;a.max_image_rmse=.015;a.max_roundtrip_mm=.1
  # Same Script34 evaluation with identical coordinate/registration semantics.
  original=s34.EVAL_FIELDS
  s34.EVAL_FIELDS=original+['ecd_min_dist_mm','rap_music_n_dipoles','lcmv_reg',
                           'lcmv_data_covariance_method','lcmv_weight_norm',
                           'mxne_alpha','mxne_loose','mxne_depth']
  try:s34.evaluate(a,configs,s32,[r for r in s34.existing(status) if r['method'] in methods],jobs)
  finally:s34.EVAL_FIELDS=original
 if a.phase in ('analyze','all'):
  s34.analyze(a,configs)
  print('Completed: ',a.output_root)

if __name__=='__main__':main()

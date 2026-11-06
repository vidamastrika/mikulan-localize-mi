#!/usr/bin/env python3
"""Script 03 — LOSO within-subject and single-reference localization, no warping.

Run from the project root with .venv active:
  python cross_subject_no_warp/scripts/03_run_localization.py --phase plan \
      --montage-template reference --output-root cross_subject_no_warp/outputs/03_reference_based
  python cross_subject_no_warp/scripts/03_run_localization.py --phase reconstruct \
      --montage-template reference --output-root cross_subject_no_warp/outputs/03_reference_based \
      --method MNE --test-subject sub-01 --run run-01 --limit 2 --quiet
  python cross_subject_no_warp/scripts/03_run_localization.py --phase reconstruct \
      --montage-template reference --output-root cross_subject_no_warp/outputs/03_reference_based \
      --resume --quiet
  # OPTIONAL sensitivity: fixed sub-01, separate outputs and test sub-01 only:
  python cross_subject_no_warp/scripts/03_run_localization.py --phase reconstruct \
      --montage-template sub-01 --output-root cross_subject_no_warp/outputs/03_fixed_sub01_sensitivity \
      --test-subject sub-01 --resume --quiet
Inputs: complete Script02 selection + its immutable source grids, and Script01
coordinate audit. Outputs: cross_subject_no_warp/outputs/03_run_localization_v2/.
Existing MNE output from previous script is not reused or overwritten.
Local project code is copied to source/03_solver_code/ on first reconstruction.
No affine, SyN, scaling, source warping, or localization-error computation.

Both conditions use the same fold parameters and same selected EEG channels.
Average referencing precedes the montage pick, as in Script09, for ALL methods.
A channel/EEG fingerprint is saved per reconstruction for paired verification.
Nominal 32/64/128 subsets and replacements are generated with Script09's
same geometric algorithm, by default from each fold's reference subject.
Optional --montage-template sub-01 reproduces the legacy nominal montage.
For an unbiased policy comparison test sub-01 in a SEPARATE output folder.
Parameter selection remains from frozen Script02 (earlier within-subject
fixed-sub-01 grids); parameters are transferred, NOT re-optimized for montage.
Good/bad flags are always taken from TEST EEG. Both solver conditions get
identical selected EEG channels and epochs, and distinct forward models.
The original parameter-search space includes data from these participants;
this is retrospective LOSO selection, not prospective independent validation.
Full channel-selection provenance saved to 09_channel_selection_audit.csv;
10_channel_selection_summary.csv summarizes status, channel counts/replacements.
The audit rows are built from the recorded reconstruction status, and
cover only attempted fits, not unexecuted planned cases.
Known stimulation hemisphere restricts cortical peak selection (as in old grids).
The stimulation coordinate is never passed to an inverse solver. This is an
informed-hemisphere experiment, not an unrestricted whole-brain benchmark.

Resume preserves PASS and FAIL rows; failures are not automatically retried.
Use a separate output folder for diagnostic reruns. One writer per output root.
"""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time
import traceback
import numpy as np
import pandas as pd

METHODS=['MNE','dSPM','sLORETA','eLORETA','Continuous-ECD','RAP-MUSIC','LCMV','MxNE','irMxNE','Gamma-MAP','Champagne','SESAME']
KEYS=['method','test_subject','run','condition']
OLD=METHODS[:8]
ALIASES={re.sub('[^a-z0-9]','',m.lower()):m for m in METHODS}
ALIASES.update({'ecd':'Continuous-ECD','cecd':'Continuous-ECD'})
FIELDS={
 'Continuous-ECD':{'ecd_min_dist_mm':['ecd_min_dist_mm','min_dist']},
 'RAP-MUSIC':{'rap_music_n_dipoles':['rap_music_n_dipoles','n_dipoles']},
 'LCMV':{'lcmv_reg':['lcmv_reg'],'lcmv_window_half_ms':['lcmv_window_half_ms','lcmv_covariance_half_window_ms','lcmv_window_ms'],
         'lcmv_data_covariance_method':['lcmv_data_covariance_method','data_covariance_method','lcmv_covariance_method'],
         'lcmv_pick_ori':['lcmv_pick_ori'],'lcmv_weight_norm':['lcmv_weight_norm']},
 'MxNE':{'mxne_alpha':['mxne_alpha','alpha'],'mxne_loose':['mxne_loose','loose'],'mxne_depth':['mxne_depth','depth']},
}
REQUIRED={m:['montage','loose','depth','snr','lambda2'] for m in METHODS[:4]}
REQUIRED.update({m:['montage']+list(fields) for m,fields in FIELDS.items()})
REQUIRED.update({
 'irMxNE':['montage','irmxne_alpha','irmxne_loose','irmxne_depth','irmxne_iterations','irmxne_random_state'],
 'Gamma-MAP':['montage','gamma_alpha','gamma_loose','gamma_depth','gamma_xyz_same_gamma','gamma_update_mode','gamma_maxit','gamma_tol'],
 'Champagne':['montage','champagne_alpha','champagne_prune','champagne_pruning_thresh','champagne_convergence_criterion','champagne_max_iter'],
 'SESAME':['montage','sesame_fs','sesame_n_parts','sesame_max_n_dips','sesame_lam','sesame_hyper_q','sesame_random_seed'],
})


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def atomic_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)


def atomic_csv(path,frame):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');frame.to_csv(tmp,index=False);os.replace(tmp,path)


def norm(v):
    if pd.isna(v) or str(v).strip() in ('','<MISSING>'):return '<MISSING>'
    s=str(v).strip()
    if s.lower() in ('true','false'):return s.lower()
    try:return format(float(s),'.12g')
    except ValueError:return s


def boolean(v):
    if str(v).lower() in ('true','1'):return True
    if str(v).lower() in ('false','0'):return False
    raise ValueError(f'Invalid boolean {v}')


def load_inputs(args):
    selection_file=args.selection_root/'01_loso_selected_parameters.csv'
    state=json.loads((args.selection_root/'07_selection_manifest.json').read_text())
    if state.get('status')!='COMPLETE' or state.get('actual_selections')!=84:raise ValueError('Script02 must finish all 84 selections first')
    if digest(selection_file)!=state['selected_parameters_sha256']:raise ValueError('Script02 selected CSV changed')
    source=args.grid_source_root or Path(state['source_root'])
    manifest=source/'source_manifest.json'
    if digest(manifest)!=state['source_manifest_sha256']:raise ValueError('Script02 source manifest changed')
    doc=json.loads(manifest.read_text())
    for item in doc['files']:
        if digest(source/item['copy'])!=item['sha256']:raise ValueError(f'Grid snapshot changed: {item["copy"]}')
    selected=pd.read_csv(selection_file,dtype=str)
    if len(selected)!=84 or selected.duplicated(['method','test_subject']).any():raise ValueError('Expected 84 unique method/fold selections')
    cases=pd.read_csv(source/doc['preparation']['03_single_reference_cases.csv'],dtype=str)
    folds=pd.read_csv(source/doc['preparation']['01_loso_folds.csv'],dtype=str)
    subjects=set(folds.test_subject)
    expected={(m,s) for m in METHODS for s in subjects}
    if len(subjects)!=7 or set(zip(selected.method,selected.test_subject))!=expected:raise ValueError('Selection method/fold coverage mismatch')
    if len(cases)!=61 or cases.duplicated(['test_subject','run']).any():raise ValueError('Bad 61-run case plan')
    for row in selected.itertuples():
        if set(row.training_subjects.split(';'))!=subjects-{row.test_subject}:raise ValueError('Training/test membership violation')
        f=folds[folds.test_subject.eq(row.test_subject)].iloc[0]
        if row.reference_subject!=f.reference_subject or row.fold_id!=f.fold_id:raise ValueError('Selection/reference mismatch')
    return selected,cases,source,doc,state


def enrich_configs(selected,source,doc):
    """Resolve actual controlled settings from matching TRAINING grid rows only.
    No parameters are chosen again, and no held-out outcomes are read here.
    """
    configs={};errors=[];cache={}
    for row in selected.itertuples():
        method=row.method;cfg=json.loads(row.configuration_parameters)
        try:
            needed=FIELDS.get(method,{})
            if needed:
                parts=[]
                for rel in doc['methods'][method]:
                    if rel not in cache:cache[rel]=pd.read_csv(source/rel,low_memory=False)
                    df=cache[rel]
                    method_mask=df.method.map(lambda v:ALIASES.get(re.sub('[^a-z0-9]','',str(v).lower()))).eq(method)
                    g=df.loc[method_mask & df.subject.isin(row.training_subjects.split(';'))].copy()
                    # All fields stored by selection must match, not just the ID.
                    for key,value in cfg.items():
                        if key in g:g=g.loc[g[key].map(norm).eq(norm(value))]
                    if len(g):parts.append(g)
                if not parts:raise ValueError('No matching training rows for configuration')
                raw=pd.concat(parts,ignore_index=True)
                for dest,aliases in needed.items():
                    vals=set()
                    for name in aliases:
                        if name in cfg:vals.add(norm(cfg[name]))
                        if name in raw:vals.update(raw[name].dropna().map(norm))
                    vals.discard('<MISSING>')
                    if len(vals)!=1:raise ValueError(f'{dest}: expected one recorded value, got {sorted(vals)}')
                    cfg[dest]=vals.pop()
                for optional in ['mxne_iterations','lcmv_reduce_rank']:
                    if optional in raw and raw[optional].notna().any():
                        vals=set(raw[optional].dropna().map(norm))
                        if len(vals)!=1:raise ValueError(f'Ambiguous {optional}')
                        cfg[optional]=vals.pop()
            for field in REQUIRED[method]:
                if field not in cfg or norm(cfg[field])=='<MISSING>':raise ValueError(f'Missing setting: {field}')
            if cfg['montage'] not in ['all_good','32','64','128']:raise ValueError('Unsupported montage')
            if method in METHODS[:4] and not np.isclose(float(cfg['lambda2']),1/float(cfg['snr'])**2):raise ValueError('lambda2/SNR mismatch')
            if method=='RAP-MUSIC' and int(float(cfg['rap_music_n_dipoles']))!=1:raise ValueError('Multi-dipole peak selection needs explicit implementation; expected n_dipoles=1')
            for key in ['gamma_xyz_same_gamma','champagne_prune','sesame_hyper_q']:
                if key in cfg:boolean(cfg[key])
            # Reproduce previously fixed time windows, never silently change them.
            for key,value in [('covariance_tmin_s',-.250),('covariance_tmax_s',-.050),('target_tmin_s',-.002),('target_tmax_s',.002)]:
                if key in cfg and not np.isclose(float(cfg[key]),value):raise ValueError(f'Unexpected processing window {key}={cfg[key]}')
                cfg[key]=value
            configs[(method,row.test_subject)]=dict(configuration_key=row.configuration_key,parameters=cfg)
        except Exception as exc:errors.append(dict(method=method,test_subject=row.test_subject,error=str(exc)))
    return configs,errors


def make_jobs(selected,cases):
    jobs=[]
    for m in METHODS:
        for case in cases.sort_values(['test_subject','run']).to_dict('records'):
            selection=selected[selected.method.eq(m)&selected.test_subject.eq(case['test_subject'])].iloc[0]
            if selection.reference_subject!=case['reference_subject']:raise ValueError('Case reference mismatch')
            for condition in ['within','single_reference_no_warp']:
                jobs.append(dict(**case,method=m,condition=condition,model_subject=case['test_subject'] if condition=='within' else case['reference_subject'],configuration_key=selection.configuration_key))
    return jobs


def snapshot_code(project,root):
    manifest=root/'code_manifest.json'
    if manifest.exists():
        doc=json.loads(manifest.read_text())
        for item in doc['files']:
            if digest(root/item['path'])!=item['sha256']:raise ValueError(f'Code snapshot changed: {item["path"]}')
        return doc
    if root.exists() and any(root.iterdir()):raise ValueError('Code source directory is nonempty without manifest')
    needed=['scripts/09_full_parameter_grid.py','scripts/19_next_method_pilot.py','src/localize_mi/__init__.py']
    for name in needed:
        if not (project/name).is_file():raise FileNotFoundError(f'Missing original solver source: {project/name}')
    files=list((project/'scripts').rglob('*.py'))
    files.extend(p for p in (project/'src').rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc')
    root.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.solver03_',dir=root.parent) as td:
        stage=Path(td)/'snapshot';stage.mkdir();doc={'version':1,'files':[]}
        for p in sorted(set(files)):
            if '__pycache__' in p.parts:continue
            rel=p.relative_to(project);dest=stage/rel;dest.parent.mkdir(parents=True,exist_ok=True)
            before=digest(p);shutil.copyfile(p,dest)
            if digest(dest)!=before or digest(p)!=before:raise RuntimeError('Source changed while copying')
            doc['files'].append(dict(path=str(rel),sha256=before))
        atomic_json(stage/'code_manifest.json',doc)
        if root.exists():root.rmdir()
        stage.rename(root)
    return doc


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod


def check_audit(args,source,doc):
    manifest=json.loads((args.preparation_root/'07_experiment_manifest.json').read_text())
    if not manifest.get('rigid_coordinate_audit_passed'):raise ValueError('Run Script01 without --plan-only and resolve the coordinate audit first')
    if Path(manifest['dataset']).resolve()!=args.dataset.resolve():raise ValueError('Dataset differs from Script01 audit')
    for name in ['01_loso_folds.csv','03_single_reference_cases.csv']:
        if digest(args.preparation_root/name)!=digest(source/doc['preparation'][name]):raise ValueError('Preparation plan differs from frozen Script02 plan')
    audit=pd.read_csv(args.preparation_root/'05_coordinate_audit.csv',dtype=str)
    if len(audit)!=7 or not audit.status.eq('PASS_NUMERICAL').all():raise ValueError('Incomplete coordinate audit')
    for row in audit.itertuples():
        for pathfield,hashfield in [('forward_path','forward_sha256'),('head_to_surface_path','h5_sha256'),('contacts_path','contacts_sha256')]:
            if digest(Path(getattr(row,pathfield)))!=getattr(row,hashfield):raise ValueError(f'Audited input changed: {row.subject}/{pathfield}')
    return audit


def peak_head(stc,fwd,hemisphere,mne):
    if hemisphere not in ('lh','rh'):raise ValueError('Missing known hemisphere for cortical peak selection')
    if not stc.data.size or not np.isfinite(stc.data).all() or not np.any(np.abs(stc.data)>0):raise ValueError('Empty/nonfinite/all-zero source estimate')
    print("STC vertices:", [len(v) for v in stc.vertices])
    print("STC shape:", stc.data.shape)
    print("Requested hemisphere:", hemisphere)
    vertex,tm=stc.get_peak(hemi=hemisphere,mode='abs')
    source=fwd['src'][0 if hemisphere=='lh' else 1]
    xyz=np.asarray(source['rr'][int(vertex)],float)
    from mne.io.constants import FIFF
    frame=int(source['coord_frame'])
    if frame==int(FIFF.FIFFV_COORD_MRI):
        t=fwd['mri_head_t']
        if int(t['from'])==int(FIFF.FIFFV_COORD_HEAD):t=mne.transforms.invert_transform(t)
        if int(t['from'])!=int(FIFF.FIFFV_COORD_MRI) or int(t['to'])!=int(FIFF.FIFFV_COORD_HEAD):raise ValueError('Invalid MRI-to-head transform')
        xyz=mne.transforms.apply_trans(t,xyz)
    elif frame!=int(FIFF.FIFFV_COORD_HEAD):raise ValueError('Unknown source coordinate frame')
    return xyz,int(vertex),float(tm)


def find_bem(root,subject):
    matches=sorted({p.resolve() for p in root.rglob('*bem*sol*.fif') if subject in p.parts or subject in p.name})
    if len(matches)!=1:raise ValueError(f'{subject}: expected one BEM solution under {root}, found {len(matches)}; set --bem-root')
    return matches[0]


class Engine:
    def __init__(self,args):
        self.args=args;self.cache={};self.nominal={};self.pilot=None
        sys.path.insert(0,str(args.code_source_root/'src'))
        import mne
        self.mne=mne
        from localize_mi import load_run
        from localize_mi import inverse
        from localize_mi.metrics import load_stimulation_info
        self.load_run=load_run;self.inverse=inverse;self.load_stim=load_stimulation_info
        self.grid09=module(args.code_source_root/'scripts/09_full_parameter_grid.py','no_warp_grid09')
    def pilot19(self):
        if self.pilot is None:self.pilot=module(self.args.code_source_root/'scripts/19_next_method_pilot.py','no_warp_pilot19')
        return self.pilot
    def forward(self,subject):
        if subject not in self.cache:
            path=self.args.dataset/'derivatives/sourcemodelling'/subject/'fwd'/f'{subject}_fwd.fif'
            self.cache[subject]=self.mne.read_forward_solution(str(path),verbose=False)
        return self.cache[subject]
    def prepare(self,job,cfg):
        mne=self.mne;test=job['test_subject'];ref=job['reference_subject'];model=job['model_subject']
        # Legacy loader may also open test anatomy; its forward is never passed
        # to the cross solver. Only the explicitly chosen model below is used.
        loaded=self.load_run(self.args.dataset,test,task='seegstim',run=job['run'])
        forward=self.forward(model)
        contacts=self.args.dataset/'derivatives/epochs'/test/'ieeg'/f'{test}_task-seegstim_space-surface_electrodes.tsv'
        stim=self.load_stim(loaded.metadata['Description'],contacts)
        template = ref if self.args.montage_template == 'reference' else self.args.montage_template
        if template not in self.nominal:self.nominal[template]=self.grid09.fixed_spatial_subsets(self.forward(template))
        names,xyz,nominal=self.nominal[template]
        eeg=loaded.epochs.copy()
        good=[c for c in eeg.ch_names if c not in eeg.info['bads']]
        montage=str(cfg['montage'])
        if montage=='all_good':actual=good;replacement=[]
        else:actual,replacement=self.grid09.replace_bad_nominal(nominal[montage],names,xyz,good)
        positions={ch['ch_name']:np.asarray(ch['loc'][:3],float) for ch in forward['info']['chs']}
        if set(eeg.ch_names)!=set(positions):raise ValueError('EEG/reference forward channel names differ')
        from mne.io.constants import FIFF
        if any(int(ch['coord_frame'])!=int(FIFF.FIFFV_COORD_HEAD) for ch in forward['info']['chs']):raise ValueError('Forward sensors not in head coordinates')
        if any(not np.isfinite(v).all() for v in positions.values()):raise ValueError('Invalid model electrode coordinates')
        eeg.set_montage(mne.channels.make_dig_montage(ch_pos=positions,coord_frame='head'),on_missing='raise',verbose=False)
        # Exact Script09 ordering: reference all loaded epochs BEFORE selecting
        # any subset; do not pick 'good EEG' first for sparse/other methods.
        eeg=self.inverse.apply_average_reference(eeg)
        # Preserve Script09's full good-channel average-reference computation.
        # Unlike legacy Script09, physically remove bad channels before handing
        # the all_good data to inverse methods. This prevents a metadata-level
        # 'Bad channel survived selection' failure for LCMV/MxNE/SESAME.
        # For reduced montages, continue to use Script09's exact selection order.
        chosen=eeg.copy().pick(actual)
        if chosen.info['bads']:raise ValueError('Bad channel survived selection')
        # Hash actual values, not just channel names or epoch counts. The two
        # model conditions must have identical EEG samples and times.
        raw_data=np.ascontiguousarray(chosen.get_data(copy=True),dtype=np.float64)
        eeg_sha=hashlib.sha256(raw_data.tobytes()).hexdigest()
        times_sha=hashlib.sha256(np.ascontiguousarray(chosen.times).tobytes()).hexdigest()
        cov=self.inverse.estimate_noise_covariance(chosen,-.250,-.050,method='auto',verbose=not self.args.quiet)
        evoked=self.inverse.create_target_evoked(chosen,-.002,.002)
        # Pass hemisphere/pair only to reconstruction logic; never a truth point.
        meta=dict(stimulation_pair=str(stim.pair),hemisphere=str(stim.hemisphere),
                  channels=len(actual),actual_channels_json=json.dumps(actual),
                  nominal_channels_json=json.dumps(good if montage=='all_good' else nominal[montage]),
                  replacement_details_json=json.dumps(replacement,sort_keys=True),
                  original_bad_channels_json=json.dumps(list(loaded.epochs.info['bads'])),
                  reference_channel_scope='all loaded channels; MNE average-ref excludes marked bads',
                  eeg_data_sha256=eeg_sha,epoch_times_sha256=times_sha,
                  sampling_frequency_hz=float(chosen.info['sfreq']),
                  highpass_hz=float(chosen.info['highpass']),lowpass_hz=float(chosen.info['lowpass']),
                  baseline_json=json.dumps(chosen.baseline),
                  montage_template_subject=template,n_epochs=len(chosen),
                  covariance_method_selected=str(cov.get('method','auto')))
        return forward,chosen,cov,evoked,meta
    def sesame(self,evoked,forward,c):
        prepare=self.pilot19().require_sesame();peak=float(np.max(np.abs(evoked.data)))
        if not np.isfinite(peak) or peak<=0:raise ValueError('Invalid EEG target peak')
        noise=float(c['sesame_fs'])*peak
        np.random.seed(int(float(c['sesame_random_seed'])))
        obj=prepare(forward,evoked,n_parts=int(float(c['sesame_n_parts'])),noise_std=noise,
            top_min=float(evoked.times[0]),top_max=float(evoked.times[-1]),dip_mom_std=None,
            hyper_q=boolean(c['sesame_hyper_q']),noise_cov=None,subsample=None,
            max_n_dips=int(float(c['sesame_max_n_dips'])),lam=float(c['sesame_lam']),verbose=not self.args.quiet)
        import sesameeg.sesame as sm
        original=sm.Particle
        def compatible(n_verts_value,lam_value,dip_mom_std=None,prior_locs=None,fixed_ori=None,hyper_q=None):
            return original(n_verts_value,lam_value,dip_mom_std=dip_mom_std,
                prior_locs=obj.prior_locs if prior_locs is None else prior_locs,
                fixed_ori=obj.fixed_ori if fixed_ori is None else fixed_ori,
                hyper_q=obj.hyper_q if hyper_q is None else hyper_q)
        sm.Particle=compatible
        try:obj.apply_sesame()
        finally:sm.Particle=original
        if not obj.est_n_dips or int(obj.est_n_dips[-1])<1:raise ValueError('SESAME selected zero sources')
        return obj.compute_stc(),dict(sesame_noise_std=noise,sesame_target_peak=peak,sesame_estimated_n_dips=int(obj.est_n_dips[-1]))
    def champagne(self,evoked,forward,cov,c):
        mne=self.mne
        if not hasattr(mne,'make_projector'):
            from mne._fiff.proj import make_projector
            def wrapper(*args,**kwargs):kwargs.pop('verbose',None);return make_projector(*args,**kwargs)
            mne.make_projector=wrapper
        fwd=mne.pick_channels_forward(forward,include=list(evoked.ch_names),exclude=[],ordered=True,copy=True,verbose=False)
        solver=self.pilot19().require_champagne_solver()('champagne')
        fn=solver.make_inverse_operator;signature=inspect.signature(fn);params=signature.parameters
        alpha=c['champagne_alpha'];alpha=alpha if alpha=='auto' else float(alpha)
        requested=dict(alpha=alpha,prune=boolean(c['champagne_prune']),pruning_thresh=float(c['champagne_pruning_thresh']),
                       convergence_criterion=float(c['champagne_convergence_criterion']),max_iter=int(float(c['champagne_max_iter'])))
        accepts_kwargs=any(p.kind==inspect.Parameter.VAR_KEYWORD for p in params.values())
        unsupported=[k for k in requested if k not in params and not accepts_kwargs]
        if unsupported:raise ValueError(f'Champagne cannot accept selected settings {unsupported}: {signature}')
        if 'noise_cov' in params:requested['noise_cov']=cov
        if 'mne_obj' in params:requested['mne_obj']=evoked
        elif 'evoked' in params:requested['evoked']=evoked
        fn(fwd,**requested);stc=solver.apply_inverse_operator(evoked)
        if isinstance(stc,(list,tuple)):
            if len(stc)!=1:raise ValueError('Multiple Champagne STCs')
            stc=stc[0]
        return stc,dict(champagne_signature=str(signature),champagne_noise_covariance_supplied='noise_cov' in params)
    def solve(self,job,c):
        mne=self.mne;m=job['method'];quiet=self.args.quiet
        fwd,epochs,cov,evoked,meta=self.prepare(job,c)
        stc=None
        if m in METHODS[:4]:
            inv=self.inverse.create_inverse_operator(epochs,fwd,cov,loose=float(c['loose']),depth=float(c['depth']),verbose=not quiet)
            stc,_,_=self.inverse.apply_inverse_method(evoked,inv,m,snr=float(c['snr']),verbose=not quiet)
        elif m in ['MxNE','irMxNE']:
            prefix='irmxne' if m=='irMxNE' else 'mxne'
            stc,_=mne.inverse_sparse.mixed_norm(evoked,fwd,cov,alpha=float(c[prefix+'_alpha']),loose=float(c[prefix+'_loose']),depth=float(c[prefix+'_depth']),
                n_mxne_iter=int(float(c.get(prefix+'_iterations',1))),random_state=int(float(c.get(prefix+'_random_state',0))),
                return_residual=True,return_as_dipoles=False,rank=None,verbose=not quiet)
        elif m=='Gamma-MAP':
            stc,_=mne.inverse_sparse.gamma_map(evoked,fwd,cov,alpha=float(c['gamma_alpha']),loose=float(c['gamma_loose']),depth=float(c['gamma_depth']),
                xyz_same_gamma=boolean(c['gamma_xyz_same_gamma']),maxit=int(float(c['gamma_maxit'])),tol=float(c['gamma_tol']),
                update_mode=int(float(c['gamma_update_mode'])),return_residual=True,return_as_dipoles=False,rank=None,verbose=not quiet)
        elif m=='LCMV':
            half=float(c['lcmv_window_half_ms'])/1000
            dc=mne.compute_covariance(epochs,tmin=-half,tmax=half,method=c['lcmv_data_covariance_method'],verbose=not quiet)
            ori=c['lcmv_pick_ori'];ori=None if str(ori).lower() in ['none','null'] else ori
            wn=c['lcmv_weight_norm'];wn=None if str(wn).lower() in ['none','null'] else wn
            filt=mne.beamformer.make_lcmv(evoked.info,fwd,dc,reg=float(c['lcmv_reg']),noise_cov=cov,pick_ori=ori,weight_norm=wn,
                reduce_rank=boolean(c.get('lcmv_reduce_rank','false')),verbose=not quiet)
            stc=mne.beamformer.apply_lcmv(evoked,filt,verbose=not quiet)
        elif m=='SESAME':stc,extra=self.sesame(evoked,fwd,c);meta.update(extra)
        elif m=='Champagne':stc,extra=self.champagne(evoked,fwd,cov,c);meta.update(extra)
        elif m in ['Continuous-ECD','RAP-MUSIC']:
            if m=='Continuous-ECD':
                bem=find_bem(self.args.bem_root,job['model_subject']);meta['bem_path']=str(bem);meta['bem_sha256']=digest(bem)
                dip,_=mne.fit_dipole(evoked,cov,str(bem),trans=fwd['mri_head_t'],min_dist=float(c['ecd_min_dist_mm']),verbose=not quiet)
            else:
                dips=mne.beamformer.rap_music(evoked,fwd,cov,n_dipoles=1,verbose=not quiet)
                if not isinstance(dips,(list,tuple)) or len(dips)!=1:raise ValueError('Unexpected RAP-MUSIC dipole result')
                dip=dips[0]
            if not len(dip.pos) or not np.isfinite(dip.pos).all() or not np.isfinite(dip.gof).all():raise ValueError('Invalid dipole estimate')
            ix=int(np.argmax(dip.gof));head=np.asarray(dip.pos[ix],float);vertex=-1;tm=float(dip.times[ix])
            meta.update(selection_criterion='maximum_GOF',peak_gof_percent=float(dip.gof[ix]),hemisphere_constraint_applied=False)
        else:raise ValueError(f'Unsupported method {m}')
        if stc is not None:
            if getattr(stc.data,'ndim',2)!=2:raise ValueError('Expected scalar cortical source estimate')
            head,vertex,tm=peak_head(stc,fwd,meta['hemisphere'],mne)
            meta.update(selection_criterion='maximum_posterior_probability' if m=='SESAME' else 'maximum_absolute_amplitude',hemisphere_constraint_applied=True)
        if head.shape!=(3,) or not np.isfinite(head).all() or np.linalg.norm(head)>1:raise ValueError('Invalid head coordinate in metres')
        return dict(**meta,estimated_head_x_mm=float(head[0]*1000),estimated_head_y_mm=float(head[1]*1000),estimated_head_z_mm=float(head[2]*1000),peak_vertex=vertex,peak_time_s=tm,coordinate_frame='model_head',coordinate_units='mm')


@contextlib.contextmanager
def lock_output(root):
    import fcntl
    root.mkdir(parents=True,exist_ok=True)
    with (root/'.writer.lock').open('w') as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another Script03 process is writing this output directory')
        try:yield
        finally:fcntl.flock(f,fcntl.LOCK_UN)


def run_jobs(engine,jobs,configs,status_path,resume=False,limit=0):
    past=pd.read_csv(status_path,keep_default_na=False).to_dict('records') if status_path.exists() else []
    if past and not resume:raise ValueError('Results already exist; add --resume')
    keyed={tuple(r[k] for k in KEYS):r for r in past}
    if len(keyed)!=len(past):raise ValueError('Duplicate checkpoint rows')
    n=0
    for job in jobs:
        key=tuple(job[k] for k in KEYS)
        if key in keyed:continue
        config=configs[(job['method'],job['test_subject'])];start=time.perf_counter()
        row=dict(**job,status='FAIL',failure_reason='',configuration_parameters=json.dumps(config['parameters'],sort_keys=True))
        try:row.update(engine.solve(job,config['parameters']));row.update(status='PASS',failure_reason='')
        except Exception as exc:
            row['failure_reason']=f'{type(exc).__name__}: {exc}'
            with (status_path.parent/'04_failure_tracebacks.log').open('a') as f:
                f.write('\n'+json.dumps(job)+'\n'+traceback.format_exc())
        row['runtime_seconds']=round(time.perf_counter()-start,6);keyed[key]=row;n+=1
        atomic_csv(status_path,pd.DataFrame(list(keyed.values())))
        print(f'{len(keyed)} recorded | {job["method"]} {job["test_subject"]} {job["run"]} {job["condition"]}: {row["status"]} {row["failure_reason"]}',flush=True)
        if limit and n>=limit:break
    return pd.DataFrame(list(keyed.values()))


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--phase',choices=['plan','reconstruct'],default='plan')
    p.add_argument('--project-root',type=Path,default=Path.cwd())
    p.add_argument('--dataset',type=Path,default=Path.home()/'Documents'/'DATASETS SOURCE LOCALIZATION'/'Localize-MI')
    p.add_argument('--selection-root',type=Path,default=Path('cross_subject_no_warp/outputs/02_select_loso_parameters'))
    p.add_argument('--preparation-root',type=Path,default=Path('cross_subject_no_warp/outputs/01_prepare_experiment'))
    p.add_argument('--grid-source-root',type=Path,help='Override only if the unchanged source snapshot was moved')
    p.add_argument('--code-source-root',type=Path,default=Path('cross_subject_no_warp/source/03_solver_code'))
    p.add_argument('--output-root',type=Path,default=Path('cross_subject_no_warp/outputs/03_reference_based'))
    p.add_argument('--bem-root',type=Path,default=Path('outputs/12_additional_methods_all_runs/bem'))
    p.add_argument('--montage-template',default='reference',
        choices=['sub-01','reference'],
        help='reference (default) derives nominal montage from reference head; sub-01 reproduces historical fixed template')
    p.add_argument('--method',choices=METHODS,action='append')
    p.add_argument('--test-subject',action='append')
    p.add_argument('--run',action='append')
    p.add_argument('--condition',choices=['within','single_reference_no_warp'],action='append')
    p.add_argument('--limit',type=int,default=0,help='Maximum new fits; 0 means all filtered jobs')
    p.add_argument('--retry-from',type=Path,
        help='Old 01_execution_status.csv: run only previously FAIL jobs in a NEW output directory')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--quiet',action='store_true')
    args=p.parse_args();args.dataset=args.dataset.expanduser().resolve();args.code_source_root=args.code_source_root.resolve()
    if args.limit<0:raise ValueError('--limit cannot be negative')
    selected,cases,source,doc,state=load_inputs(args)
    alljobs=make_jobs(selected,cases)
    jobs=[j for j in alljobs if (not args.method or j['method'] in args.method) and (not args.test_subject or j['test_subject'] in args.test_subject) and (not args.run or j['run'] in args.run) and (not args.condition or j['condition'] in args.condition)]
    if args.retry_from:
        old_path=args.retry_from.expanduser().resolve()
        if not old_path.is_file():raise FileNotFoundError(old_path)
        if args.output_root.resolve()==old_path.parent.resolve():
            raise ValueError('Retry output must be separate from the original results')
        previous=pd.read_csv(old_path,dtype=str,keep_default_na=False)
        required=set(KEYS+['status','failure_reason'])
        if not required.issubset(previous):raise ValueError(f'Retry CSV missing columns: {required-set(previous)}')
        if previous.duplicated(KEYS).any():raise ValueError('Duplicate jobs in retry CSV')
        old_keys=set(map(tuple,previous[KEYS].values))
        plan_keys={tuple(j[k] for k in KEYS) for j in alljobs}
        if not old_keys<=plan_keys:raise ValueError('Retry CSV contains unknown jobs')
        failed={tuple(r[k] for k in KEYS) for _,r in previous[previous.status.eq('FAIL')].iterrows()}
        jobs=[j for j in jobs if tuple(j[k] for k in KEYS) in failed]
        print(f'Retry source: {old_path} | failed jobs: {len(failed)} | requested: {len(jobs)}')
    if not jobs:raise ValueError('No jobs match the requested filters')
    # Resolve all configurations once; no fitting begins if any setting is ambiguous.
    configs,errors=enrich_configs(selected,source,doc)
    print(f'Full plan: {len(alljobs)} fits; requested: {len(jobs)} fits.')
    with lock_output(args.output_root):
        existing_manifest=args.output_root/'06_frozen_run_manifest.json'
        if existing_manifest.exists():
            prior=json.loads(existing_manifest.read_text())
            if (prior['selection_sha256']!=digest(args.selection_root/'01_loso_selected_parameters.csv')
                    or prior['script_sha256']!=digest(__file__)):
                raise ValueError('Existing run uses different selection or script bytes; use a new output directory')
        atomic_csv(args.output_root/'02_reconstruction_plan.csv',pd.DataFrame(alljobs))
        if args.retry_from:
            atomic_json(args.output_root/'11_retry_source.json',{'source_csv':str(args.retry_from.resolve()),
                'source_sha256':digest(args.retry_from.resolve()),'n_requested_jobs':len(jobs),
                'policy':'only rows with previous status FAIL; never overwrite original'})
        atomic_csv(args.output_root/'03_configuration_preflight.csv',pd.DataFrame(errors,columns=['method','test_subject','error']))
        atomic_json(args.output_root/'05_resolved_configurations.json',{'|'.join(k):v for k,v in configs.items()})
        if errors:raise SystemExit('Parameter preflight failed. Inspect 03_configuration_preflight.csv; no solvers ran.')
        if args.phase=='plan':
            print('Configuration preflight passed. Plan only: no solvers or coordinate audit rerun.');return
        audit=check_audit(args,source,doc)
        snapshot_code(args.project_root.resolve(),args.code_source_root)
        engine=Engine(args)
        versions={}
        for dist in ['mne','numpy','scipy','pandas','sesameeg','invertmeeg']:
            try:versions[dist]=importlib.metadata.version(dist)
            except importlib.metadata.PackageNotFoundError:versions[dist]='not_installed'
        frozen=dict(version=1,dataset=str(args.dataset),selection_sha256=digest(args.selection_root/'01_loso_selected_parameters.csv'),
                    code_manifest_sha256=digest(args.code_source_root/'code_manifest.json'),script_sha256=digest(__file__),
                    coordinate_audit_sha256=digest(args.preparation_root/'05_coordinate_audit.csv'),versions=versions,
                    resolved_configs={'|'.join(k):v for k,v in configs.items()},nominal_montage=args.montage_template,
                    retry_source_sha256=digest(args.retry_from.resolve()) if args.retry_from else None,
                    cortical_peak='known stimulation hemisphere',bem_root=str(args.bem_root.resolve()),brain_registration=False,warping=False)
        manifest=args.output_root/'06_frozen_run_manifest.json'
        if manifest.exists() and json.loads(manifest.read_text())!=frozen:raise ValueError('Run settings/code/input files changed; use a new output directory')
        if not manifest.exists():atomic_json(manifest,frozen)
        status=args.output_root/'01_execution_status.csv'
        if status.exists():
            old=pd.read_csv(status,keep_default_na=False)
            allowed={tuple(j[k] for k in KEYS) for j in alljobs}
            if not set(map(tuple,old[KEYS].values))<=allowed:raise ValueError('Checkpoint contains jobs outside the plan')
        result=run_jobs(engine,jobs,configs,status,args.resume,args.limit)
        # Same test/run/method must feed identical numerical epochs to both
        # model conditions. This check is possible as soon as both finish.
        if len(result) and 'eeg_data_sha256' in result:
            pairs=result[result.status.eq('PASS')].groupby(['method','test_subject','run'])
            audit=[]
            for key,group in pairs:
                if set(group.condition)=={'within','single_reference_no_warp'}:
                    fields=['eeg_data_sha256','epoch_times_sha256','actual_channels_json',
                            'sampling_frequency_hz','n_epochs','baseline_json']
                    ok=all(group[f].nunique(dropna=False)==1 for f in fields)
                    audit.append(dict(zip(['method','test_subject','run'],key),
                                      identical_inputs=ok,fields_checked=';'.join(fields)))
                    if not ok:raise ValueError(f'Within/cross input mismatch: {key}')
            if audit:atomic_csv(args.output_root/'08_paired_input_audit.csv',pd.DataFrame(audit))
        if len(result):
            # Audit all attempted reconstruction records, including failures.
            # Avoid collapsing unique channel sets into summary statistics only.
            channel_columns=['method','test_subject','run','reference_subject',
                'model_subject','condition','status','failure_reason',
                'configuration_parameters','montage_template_subject',
                'channels','actual_channels_json','nominal_channels_json',
                'original_bad_channels_json','replacement_details_json',
                'reference_channel_scope','eeg_data_sha256','epoch_times_sha256',
                'sampling_frequency_hz','highpass_hz','lowpass_hz','baseline_json',
                'n_epochs','covariance_method_selected']
            audit=result.reindex(columns=channel_columns).copy()
            atomic_csv(args.output_root/'09_channel_selection_audit.csv',audit)
            summaries=[]
            for _, r in audit.iterrows():
                def read_list(key):
                    v=r.get(key)
                    if not isinstance(v,str) or not v.strip():return []
                    try:
                        x=json.loads(v)
                        return x if isinstance(x,list) else []
                    except (json.JSONDecodeError,TypeError):return []
                final=read_list('actual_channels_json')
                nominal=read_list('nominal_channels_json')
                bad=read_list('original_bad_channels_json')
                replacement=read_list('replacement_details_json')
                summaries.append(dict(method=r['method'],test_subject=r['test_subject'],
                    run=r['run'],reference_subject=r['reference_subject'],
                    model_subject=r['model_subject'],condition=r['condition'],
                    status=r['status'],montage_template_subject=r.get('montage_template_subject'),
                    n_nominal=len(nominal),n_selected=len(final),n_original_bad=len(bad),
                    n_replacements=len(replacement),
                    channel_names_unique=len(final)==len(set(final)),
                    selected_include_bad=bool(set(final)&set(bad)),
                    final_channels_joined=';'.join(final)))
            atomic_csv(args.output_root/'10_channel_selection_summary.csv',pd.DataFrame(summaries))
            result.groupby(['method','condition','status']).size().rename('count').to_csv(args.output_root/'07_execution_summary.csv')
            print(result.groupby(['method','condition','status']).size().to_string())
        print('Saved:',args.output_root)
        print('PASS means solver completed, not reproduction validation. Inspect pilot before scaling up; Script04 will evaluate errors.')

if __name__=='__main__':main()

#!/usr/bin/env python3
# SCRIPT 02 — Select training-only LOSO parameters; no inverse or registration.
# Run from the project root with the project's .venv activated.
#
# First selection (uses Script 01's default output):
# python cross_subject_warp/scripts/02_select_loso_parameters.py
#
# Use the reviewed preparation audit:
# python cross_subject_warp/scripts/02_select_loso_parameters.py \
#   --preparation-root cross_subject_warp/outputs/01_prepare_experiment_review
#
# Fresh repeat without overwriting any previous selection:
# python cross_subject_warp/scripts/02_select_loso_parameters.py \
#   --preparation-root cross_subject_warp/outputs/01_prepare_experiment_review \
#   --output-root cross_subject_warp/outputs/02_select_loso_parameters_review
#
# Every output directory must be new. Frozen inputs are verified, never changed.
# Exit 0 requires all 84 method/fold winners. No fallback to global winners.
"""Strict retrospective training-only LOSO selection using frozen historical grids.

The historical selection functions below are preserved unchanged. Only six
training subjects contribute to repair, reliability screening and scoring.
Affine and SyN share selections and subsequent native inverse fits. Coordinate
bridge readiness is recorded separately; it does not affect grid selection.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path
import numpy as np
import pandas as pd

METHODS=['MNE','dSPM','sLORETA','eLORETA','Continuous-ECD','RAP-MUSIC','LCMV','MxNE','irMxNE','Gamma-MAP','Champagne','SESAME']
ORIGINAL=METHODS[:4]
NEW=METHODS[8:]
ALIASES={re.sub('[^a-z0-9]','',m.lower()):m for m in METHODS}
ALIASES.update({'ecd':'Continuous-ECD','cecd':'Continuous-ECD'})
PARAMS={m:['method_origin','montage','loose','depth','snr','lambda2'] for m in ORIGINAL}
PARAMS.update({
 'Continuous-ECD':['montage','configuration_id','ecd_min_dist_mm'],
 'RAP-MUSIC':['montage','configuration_id','rap_music_n_dipoles'],
 'LCMV':['montage','configuration_id','lcmv_reg','lcmv_window_half_ms','lcmv_data_covariance_method','lcmv_pick_ori','lcmv_weight_norm','lcmv_reduce_rank'],
 'MxNE':['montage','configuration_id','mxne_alpha','mxne_loose','mxne_depth','mxne_iterations','alpha','loose','depth'],
 'irMxNE':['montage','irmxne_alpha','irmxne_loose','irmxne_depth','irmxne_iterations','irmxne_random_state'],
 'Gamma-MAP':['montage','gamma_alpha','gamma_loose','gamma_depth','gamma_xyz_same_gamma','gamma_update_mode','gamma_maxit','gamma_tol'],
 'Champagne':['montage','champagne_alpha','champagne_prune','champagne_pruning_thresh','champagne_convergence_criterion','champagne_max_iter'],
 'SESAME':['montage','sesame_fs','sesame_n_parts','sesame_max_n_dips','sesame_lam','sesame_hyper_q','sesame_random_seed'],
})
REQUIRED={m:['montage','loose','depth','snr','lambda2'] for m in ORIGINAL}
REQUIRED.update({m:PARAMS[m] for m in NEW})
REQUIRED.update({m:['montage','configuration_id'] for m in METHODS[4:8]})
PRIMARY={'Continuous-ECD':['ecd_min_dist_mm'],'MxNE':['mxne_alpha','alpha'],
         'irMxNE':['irmxne_alpha'],'Gamma-MAP':['gamma_alpha'],
         'Champagne':['champagne_pruning_thresh'],'SESAME':['sesame_fs']}
ALLOWED={
 'irMxNE':{'23_full_next_method_grid','23_irmxne_alpha15_extension','23_irmxne_alpha20_extension'},
 'Gamma-MAP':{'23_full_next_method_grid','23_gamma_alpha_extension','23_gamma_alpha_low_extension','23_gamma_xyz_true'},
 'Champagne':{'23_full_next_method_grid','23_champagne_pruning_extension'},
 'SESAME':{'23_full_next_method_grid','23_sesame_fs_extension'},
}
ROOT=Path(__file__).resolve().parents[2]
EXP=ROOT/'cross_subject_warp'
ERRORS=['localization_distance_mm','localization_error_mm','selected_localization_error_mm','distance_mm','error_mm']


def canonical(value):
    k=re.sub('[^a-z0-9]','',str(value).lower())
    if k not in ALIASES:raise ValueError(f'Unknown method {value}')
    return ALIASES[k]


def norm(value):
    if pd.isna(value) or str(value).strip()=='':return '<MISSING>'
    s=str(value).strip()
    if s.lower() in ['true','false']:return s.lower()
    if re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?',s):return format(float(s),'.12g')
    return s


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def kv(values):
    result={}
    for item in values:
        method,sep,value=item.partition('=')
        if not sep:raise ValueError('Expected METHOD=PATH or METHOD=COLUMN')
        result.setdefault(canonical(method),[]).append(value)
    return result


def snapshot(args):
    """Read only: require the pinned snapshot, never create or replace inputs."""
    root=args.source_root.expanduser().resolve()
    manifest=root/'source_manifest.json'
    doc=json.loads(manifest.read_text())
    pinned=json.loads((EXP/'source/provenance_manifest.json').read_text())
    for item in pinned['files']:
        if item['path'].startswith('source/02_parameter_grids/'):
            path=root/Path(item['path']).relative_to('source/02_parameter_grids')
            if not path.is_file() or digest(path)!=item['sha256']:
                raise ValueError(f'Frozen provenance mismatch: {path}')
    for item in doc['files']:
        path=root/item['copy']
        if not path.is_file() or digest(path)!=item['sha256']:
            raise ValueError(f'Snapshot missing or modified: {path}')
    print('Using verified frozen source snapshot:',root,flush=True)
    return root,doc


def verify_preparation(preparation,root,doc):
    """Match fresh plans to frozen plans without altering historical manifests."""
    records=[]
    for name,keys in [('01_loso_folds.csv',['fold_id']),
                      ('03_single_reference_cases.csv',['test_subject','run'])]:
        actual=pd.read_csv(preparation/name,dtype=str)
        frozen=pd.read_csv(root/doc['preparation'][name],dtype=str)
        if set(actual.columns)!=set(frozen.columns):
            raise ValueError(f'Preparation columns differ from frozen plan: {name}')
        actual=actual[frozen.columns].sort_values(keys).reset_index(drop=True)
        frozen=frozen.sort_values(keys).reset_index(drop=True)
        pd.testing.assert_frame_equal(actual,frozen)
        records.append(dict(path=str(preparation/name),sha256=digest(preparation/name)))
    manifest=preparation/'07_experiment_manifest.json'
    current=json.loads(manifest.read_text())
    if (current.get('n_folds'),current.get('n_runs'),current.get('n_methods'))!=(7,61,12):
        raise ValueError('Unexpected preparation manifest dimensions')
    if current.get('frozen_sources_match') is not True:
        raise ValueError('Preparation does not establish frozen-source integrity')
    records.append(dict(path=str(manifest),sha256=digest(manifest)))
    return dict(files=records,readiness=current['readiness'],
                missing_resources=current.get('missing_resources',[]),
                unresolved_assumptions=current.get('unresolved_assumptions',[]),
                plan_matches_frozen=True)


def load_plan(root,doc):
    folds=pd.read_csv(root/doc['preparation']['01_loso_folds.csv'],dtype=str)
    cases=pd.read_csv(root/doc['preparation']['03_single_reference_cases.csv'],dtype=str)
    subjects={f'sub-{i:02d}' for i in range(1,8)}
    if len(folds)!=7 or set(folds.test_subject)!=subjects or folds.fold_id.duplicated().any():raise ValueError('Invalid seven-fold plan')
    if len(cases)!=61 or cases.duplicated(['test_subject','run']).any():raise ValueError('Invalid 61-run plan')
    for r in folds.itertuples():
        if set(r.training_subjects.split(';'))!=subjects-{r.test_subject}:raise ValueError('Training membership includes test or omits training subject')
        if r.reference_subject==r.test_subject or r.reference_subject not in subjects:raise ValueError('Invalid reference')
        c=cases[cases.test_subject.eq(r.test_subject)]
        if not len(c) or not c.reference_subject.eq(r.reference_subject).all() or not c.fold_id.eq(r.fold_id).all():raise ValueError('Fold/case mismatch')
    return folds,cases


def load_method(root,doc,method,cache,error_override=None):
    frames=[]
    for rel in doc['methods'][method]:
        if rel not in cache:cache[rel]=pd.read_csv(root/rel,low_memory=False)
        d=cache[rel]
        if 'method' not in d:raise ValueError(f'{rel}: missing method column')
        labels=d.method.map(canonical)
        d=d.loc[labels.eq(method)].copy()
        if d.empty:raise ValueError(f'{rel}: no {method} rows')
        d['_snapshot_file']=rel;frames.append(d)
    d=pd.concat(frames,ignore_index=True,sort=False)
    if method in NEW and 'source_experiment' in d:
        d=d[d.source_experiment.isin(ALLOWED[method])].copy()
    for c in ['subject','run','status','montage']:
        if c not in d or d[c].isna().any():raise ValueError(f'{method}: missing {c}')
        d[c]=d[c].astype(str).str.strip()
    d['status']=d.status.str.upper()
    if error_override:
        if error_override not in d:raise ValueError(f'No error column {error_override}')
        ec=error_override
    else:
        cols=[c for c in ERRORS if c in d]
        if len(cols)!=1:raise ValueError(f'{method}: choose --error-column "{method}=COLUMN" from {cols}')
        ec=cols[0]
    # Values are converted only after filtering the training subjects below.
    d['_raw_error']=d[ec]
    return d,ec


def prepare_training(data,training,method):
    d=data.loc[data.subject.isin(training)].copy()  # CRITICAL: precedes every outcome-dependent operation
    if d.empty:raise ValueError(f'{method}: no training rows')
    missing=[c for c in REQUIRED[method] if c not in d]
    if missing:raise ValueError(f'{method}: absent configuration fields {missing}')
    fields=[c for c in PARAMS[method] if c in d and (c in REQUIRED[method] or d[c].notna().any())]
    # Carry recorded fixed processing windows as part of the configuration.
    for c in ['covariance_tmin_s','covariance_tmax_s','target_tmin_s','target_tmax_s']:
        if c in d and d[c].notna().any():fields.append(c)
    for c in fields:d[c]=d[c].map(norm).replace('<MISSING>',np.nan)
    if 'configuration_id' in d:
        d['configuration_id']=d.configuration_id.map(norm)
        # Equivalent numeric ID spellings (0.50 vs 0.5) must share repair groups.
        d['_repair_id']=d.configuration_id.map(lambda value:re.sub(r'\d+(?:\.\d+)?',lambda match:format(float(match.group()),'.12g'),str(value)))
        keys=['_snapshot_file','_repair_id','montage']
        if 'source_experiment' in d:keys.insert(1,'source_experiment')
        if 'source_file' in d:keys.insert(1,'source_file')
        for c in fields:
            if c in keys:continue
            grouped=d.groupby(keys,dropna=False)[c]
            mask=d[c].isna() & grouped.transform('nunique').eq(1)
            d.loc[mask,c]=grouped.transform('first').loc[mask]
    primary=next((c for c in PRIMARY.get(method,[]) if c in fields),None)
    # Some historical Gamma-MAP FAIL-only IDs have no successful rows in that
    # source. Recover ONLY the explicitly encoded primary value, not guessed
    # solver defaults. Their unknown remaining settings stay marked missing.
    patterns={'Gamma-MAP':r'^gamma_a([0-9.eE+-]+)_',
              'irMxNE':r'^irmxne_a([0-9.eE+-]+)_',
              'SESAME':r'^sesame_fs_([0-9p.eE+-]+)',
              'Champagne':r'^champagne_thresh_([0-9.eE+-]+)'}
    if primary and method in patterns and 'configuration_id' in d:
        mask=d[primary].isna()
        for idx in d.index[mask]:
            match=re.search(patterns[method],str(d.at[idx,'configuration_id']))
            if match:d.at[idx,primary]=norm(float(match.group(1).replace('p','.')))
    d['_missing_settings']=d[fields].isna().any(axis=1)
    unresolved=d['_missing_settings']
    # A partial failed row may block its known primary value conservatively;
    # it may never become an eligible configuration or disappear from screening.
    unsafe=unresolved & (d.status.eq('PASS') | (d[primary].isna() if primary else True))
    if unsafe.any():
        sample=d.loc[unsafe,['subject','run']+(['configuration_id'] if 'configuration_id' in d else [])].head().to_dict('records')
        raise ValueError(f'{method}: cannot resolve training configuration: {sample}. Inspect raw grid; do not fill from held-out data.')
    d[fields]=d[fields].fillna('<MISSING>')
    d['_params']=d[fields].apply(lambda r:json.dumps(r.to_dict(),sort_keys=True),axis=1)
    d['_key']=d['_params'].map(lambda x:hashlib.sha256(x.encode()).hexdigest()[:20])
    d['_error']=pd.to_numeric(d['_raw_error'],errors='coerce')
    d['_valid']=d.status.eq('PASS') & np.isfinite(d['_error']) & d['_error'].ge(0) & ~d['_missing_settings']
    primary=next((c for c in PRIMARY.get(method,[]) if c in fields),None)
    if method in PRIMARY and primary is None:
        raise ValueError(f'{method}: primary parameter missing; expected {PRIMARY[method]}. Supply a grid with explicit settings.')
    d['_primary']=d[primary] if primary else ''
    return d,fields,primary


def select_fold(data,training,expected,method,reliability='primary-all-pass'):
    d,fields,primary=prepare_training(data,training,method)
    records=[];duplicates=[]
    for ck,g in d.groupby('_key',sort=True):
        params=g['_params'].iloc[0];pv=g['_primary'].iloc[0]
        conflict=False
        repeated=g.loc[g.duplicated(['subject','run'],keep=False)]
        for (sub,run),dup in repeated.groupby(['subject','run']):
            if len(dup)>1:
                signatures=dup[['status','_error']].drop_duplicates()
                bad=len(signatures)>1;conflict|=bad
                duplicates.append(dict(method=method,configuration_key=ck,subject=sub,run=run,rows=len(dup),conflict=bad))
        g=g.drop_duplicates(['subject','run'])
        keys=set(zip(g.subject,g.run));n_bad=int((~g['_valid']).sum())
        complete=keys==expected
        config_ok=complete and not conflict and n_bad==0
        score=pooled_median=pooled_mean=np.nan
        if config_ok:
            medians=g.groupby('subject')['_error'].median()
            if set(medians.index)!=set(training):raise AssertionError('Not all training subjects represented')
            score=float(medians.mean());pooled_median=float(g['_error'].median());pooled_mean=float(g['_error'].mean())
        records.append(dict(method=method,configuration_key=ck,configuration_parameters=params,
            primary_parameter=primary or '',primary_value=pv,expected_runs=len(expected),observed_runs=len(g),
            missing_runs=len(expected-keys),unexpected_runs=len(keys-expected),invalid_runs=n_bad,
            duplicate_conflict=conflict,unresolved_failed_settings=bool(g['_missing_settings'].any()),configuration_all_pass=config_ok,
            score_mean_subject_median_mm=score,pooled_median_mm=pooled_median,pooled_mean_mm=pooled_mean))
    audit=pd.DataFrame(records)
    if audit.empty:raise ValueError(f'{method}: no training configurations')
    screen=[];blocked=set()
    if primary:
        for pv,g in audit.groupby('primary_value'):
            n_bad=int((~g.configuration_all_pass).sum())
            if n_bad:blocked.add(pv)
            screen.append(dict(method=method,primary_parameter=primary,primary_value=pv,
                               observed_configurations=len(g),ineligible_configurations=n_bad,primary_all_pass=n_bad==0))
    audit['primary_value_allowed']=~audit.primary_value.isin(blocked) if primary else True
    audit['eligible']=audit.configuration_all_pass
    if reliability=='primary-all-pass':audit['eligible'] &= audit.primary_value_allowed
    # Even the relaxed policy cannot ignore an ambiguously identified failure.
    uncertain_values=set(audit.loc[audit.unresolved_failed_settings,'primary_value'])
    audit['eligible'] &= ~audit.primary_value.isin(uncertain_values)
    candidates=audit[audit.eligible].sort_values(['score_mean_subject_median_mm','pooled_median_mm','pooled_mean_mm','configuration_key'])
    winner=None if candidates.empty else candidates.iloc[0].to_dict()
    return winner,audit,screen,duplicates,fields


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--preparation-root',type=Path,default=EXP/'outputs/01_prepare_experiment')
    p.add_argument('--source-root',type=Path,default=EXP/'source/02_parameter_grids')
    p.add_argument('--output-root',type=Path,default=EXP/'outputs/02_select_loso_parameters')
    p.add_argument('--error-column',action='append',default=[],help='METHOD=COLUMN, only if error-column selection is ambiguous')
    p.add_argument('--reliability',choices=['primary-all-pass','configuration-all-pass'],default='primary-all-pass',help='Default preserves Script26 strict primary-value screening, within each training fold only')
    a=p.parse_args();out=a.output_root.expanduser().resolve()
    a.preparation_root=a.preparation_root.expanduser().resolve()
    if out.exists() or out.is_symlink():
        raise FileExistsError(f'Output already exists: {out}. Choose a new --output-root; overwrites are refused.')
    root,doc=snapshot(a);folds,cases=load_plan(root,doc);error_overrides=kv(a.error_column)
    preparation=verify_preparation(a.preparation_root,root,doc)
    out.mkdir(parents=True,exist_ok=False)
    # Interrupted or failed selection must never appear complete.
    state={'status':'IN_PROGRESS','source_root':str(root),'reliability':a.reliability,
           'preparation':preparation,'registration_fitting_executed':False,
           'inverse_reconstruction_executed':False,
           'historical_selection_script_sha256':digest(EXP/'source/00_no_warp_workflow/scripts/02_select_loso_parameters.py')}
    (out/'07_selection_manifest.json').write_text(json.dumps(state,indent=2))
    winners=[];coverage=[];audits=[];screens=[];duplicates=[];schema=[];cache={}
    for method in METHODS:
        print(f'Selecting {method} ...',flush=True)
        data,ec=load_method(root,doc,method,cache,error_overrides.get(method,[None])[0])
        for fold in folds.itertuples():
            training=fold.training_subjects.split(';')
            expected=set(map(tuple,cases.loc[cases.test_subject.isin(training),['test_subject','run']].values))
            winner,audit,screen,dups,fields=select_fold(data,training,expected,method,a.reliability)
            print(f'  {fold.test_subject}: {int(audit.eligible.sum())}/{len(audit)} eligible configurations',flush=True)
            context=dict(fold_id=fold.fold_id,test_subject=fold.test_subject,reference_subject=fold.reference_subject,training_subjects=fold.training_subjects)
            for c,v in context.items():audit[c]=v
            audits.append(audit);screens.extend({**r,**context} for r in screen);duplicates.extend({**r,**context} for r in dups)
            schema.append(dict(method=method,fold_id=fold.fold_id,parameter_fields=fields,error_column=ec))
            coverage.append(dict(**context,method=method,expected_training_runs=len(expected),candidate_configurations=len(audit),eligible_configurations=int(audit.eligible.sum()),selection_status='SELECTED' if winner else 'NO_ELIGIBLE_CONFIGURATION'))
            if winner:winners.append(dict(**context,**winner,n_training_subjects=6,selection_status='SELECTED'))
        # Source files can be large; keep shared-file cache only through its last method.
        used_later={r for m in METHODS[METHODS.index(method)+1:] for r in doc['methods'][m]}
        cache={r:df for r,df in cache.items() if r in used_later}
    selected=pd.DataFrame(winners)
    selected.to_csv(out/'01_loso_selected_parameters.csv',index=False)
    pd.DataFrame(coverage).to_csv(out/'02_fold_selection_coverage.csv',index=False)
    pd.concat(audits,ignore_index=True).to_csv(out/'03_training_configuration_audit.csv',index=False)
    pd.DataFrame(screens).to_csv(out/'04_training_primary_value_audit.csv',index=False)
    pd.DataFrame(duplicates,columns=['method','configuration_key','subject','run','rows','conflict','fold_id','test_subject','reference_subject','training_subjects']).to_csv(out/'05_training_duplicate_audit.csv',index=False)
    (out/'06_parameter_fields.json').write_text(json.dumps(schema,indent=2)+'\n')
    complete=len(winners)==84
    state.update(status='COMPLETE' if complete else 'INCOMPLETE',expected_selections=84,actual_selections=len(winners),
        score='mean of six training-participant medians',tie_breaks=['pooled median','pooled mean','configuration key'],
        source_manifest_sha256=digest(root/'source_manifest.json'),script_sha256=digest(__file__),
        selected_parameters_sha256=digest(out/'01_loso_selected_parameters.csv'),
        test_outcomes_used=False,selection_geometry='prior within-subject grid metric',
        peak_policy='inherited from prior grids; verify hemisphere restriction before reconstruction',
        parameter_use='same fold configuration for within and cross-reference localization; affine/SyN share inverse fits',
        anatomical_evaluation_ready=False,
        error_column_overrides=error_overrides,allowed_experiments={m:sorted(v) for m,v in ALLOWED.items()})
    (out/'07_selection_manifest.json').write_text(json.dumps(state,indent=2)+'\n')
    shutil.copyfile(root/'source_manifest.json',out/'08_source_manifest.json')
    (out/'README.txt').write_text('''Training-only LOSO parameter selection from frozen prior-grid copies.
This stage verifies and reuses the existing frozen raw grids, including held-out outcomes for provenance. Selection filters out the held-out rows before all parameter repair, reliability screening, duplicate handling, and scoring. No held-out error summary is produced here.
Default reliability=primary-all-pass follows Script26 strict primary-value checks on the SIX training subjects only, plus per-configuration all-pass coverage. Configuration-all-pass is a separate optional policy and must be reported as such.
Rank by mean of participant medians, then pooled median, pooled mean, deterministic key. Equal-participant weighting follows the earlier LOSO script, not Script26 pooled-median scoring.
All 84 method/fold selections must exist before Script03. No eligible configuration produces an explicit missing-selection record and nonzero exit; no fallback to globally selected parameters.
Exact duplicate status/error observations are collapsed; conflicting duplicates invalidate their training configuration. Missing settings are repaired only from unambiguous same-source/configuration/montage TRAINING rows. No values are borrowed from the held-out subject. Failed rows with unresolved non-primary settings remain explicit incomplete evidence and conservatively block their ID-encoded primary value under either reliability policy; successful rows with unresolved settings stop selection.
Gamma depth-test experiments remain outside the inherited experiment allowlist. MxNE uses extension inputs as in Script26. All input copies are preserved byte-for-byte; allowlist filtering is applied during analysis.
This is parameter transfer from prior within-subject errors. The reference head does not determine which training subjects select parameters: all six train subjects contribute, even though one reference head is used later.
Prior grids and their search ranges were developed using this dataset. This is retrospective LOSO re-selection, not an untouched prospective validation. Candidate search-space history and known-hemisphere peak restrictions must be reported.
The source snapshot is pinned and SHA-256 verified. Existing output directories are refused; use a fresh --output-root for another run. No original results are changed. Fresh warp plans are checked against frozen historical plans; the historical preparation manifest is preserved. Coordinate readiness remains separate from parameter selection.
''')
    print(f'Selected {len(winners)}/84 method/fold configurations. Saved: {out}')
    if not complete:raise SystemExit('Some folds have no eligible configuration. Inspect 02_fold_selection_coverage.csv; do not run reconstruction yet.')

if __name__=='__main__':main()

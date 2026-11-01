#!/usr/bin/env python3
"""Script 02: snapshot prior grids, then select training-only LOSO parameters.

COMMAND (project root, activated .venv):
    python cross_subject_no_warp/scripts/02_select_loso_parameters.py

FIRST RUN copies unfiltered original/Script15/MxNE-extension/Script24-master
CSVs and Script01 plans into source/02_parameter_grids/. Later runs verify and
reuse those copies, even if the old experiment changes. Source files are never
modified. To use different inputs, supply a new --source-root directory.

Selection: for each held-out subject, exclude its rows BEFORE repairing missing
settings, detecting duplicate conflicts, screening reliability or scoring.
Default reliability=primary-all-pass: exclude a primary value if any observed
training configuration using it is incomplete or has a non-PASS/invalid result;
each candidate must also pass every training run. No test outcomes are checked.
Score = mean of six participant medians; ties: pooled median, pooled mean, key.
This differs from Script26's pooled-median ranking, and follows Script39's LOSO
participant-balanced score. Candidate grids and prior peak-definition remain
inherited from earlier experiments; this is not a prospectively untouched grid.

Output: cross_subject_no_warp/outputs/02_select_loso_parameters/
No MNE, inverse solver, registration, or EEG loading is needed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import shutil
import tempfile
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


def source_paths(args):
    original=Path('outputs/grid_all_runs/grid_results.csv')
    root=Path('outputs/15_full_additional_method_grid')
    paths={m:[original] for m in ORIGINAL}
    for m,slug in [('Continuous-ECD','continuous_ecd'),('RAP-MUSIC','rap_music'),('LCMV','lcmv')]:
        paths[m]=[root/slug/'grid_results.csv']
    extensions=set(Path('outputs/15_mxne_alpha_extension').rglob('grid_results*.csv'))
    if root.exists():
        extensions.update(p for p in root.rglob('grid_results*.csv')
                          if 'mxne' in str(p.parent).lower() and p.parent.resolve()!=(root/'mxne').resolve())
    paths['MxNE']=sorted(extensions)
    master=Path('outputs/23_full_next_method_grid/23_all_next_methods_combined.csv')
    paths.update({m:[master] for m in NEW})
    for m,ps in kv(args.grid).items():paths[m]=[Path(p).expanduser() for p in ps]
    missing=[]
    for m,ps in paths.items():
        if not ps:missing.append(f'{m}: no extension CSV found; pass --grid MxNE=PATH')
        for p in ps:
            if not p.is_file():missing.append(f'{m}: {p}')
            if p.name in {'06_complete_any_status.csv','07_complete_all_pass.csv'}:
                raise ValueError('Use unfiltered raw/master grids, not globally filtered Script25 files')
    if missing:raise FileNotFoundError('Missing source inputs:\n'+'\n'.join(missing))
    return paths


def snapshot(args):
    root=args.source_root.resolve();manifest=root/'source_manifest.json'
    if manifest.exists():
        if args.grid:raise ValueError('Existing snapshot: input overrides ignored by design. Choose a NEW --source-root for different inputs.')
        doc=json.loads(manifest.read_text())
        for item in doc['files']:
            path=root/item['copy']
            if not path.is_file() or digest(path)!=item['sha256']:raise ValueError(f'Snapshot missing or modified: {path}')
        print('Using verified local source snapshot:',root,flush=True)
        return root,doc
    if root.exists() and any(root.iterdir()):raise ValueError('Source directory nonempty without manifest; choose a new directory')
    paths=source_paths(args)
    plans={name:args.preparation_root/name for name in ['01_loso_folds.csv','03_single_reference_cases.csv','07_experiment_manifest.json']}
    for path in plans.values():
        if not path.is_file():raise FileNotFoundError(f'Run Script01 first: {path}')
    root.parent.mkdir(parents=True,exist_ok=True)
    # Copy a complete verified snapshot transactionally before selecting anything.
    with tempfile.TemporaryDirectory(prefix='.snapshot02_',dir=root.parent) as temp:
        stage=Path(temp)/'snapshot';stage.mkdir();(stage/'inputs').mkdir();(stage/'preparation').mkdir()
        doc={'version':1,'files':[],'methods':{},'preparation':{}};mapped={}
        for m,ps in paths.items():
            doc['methods'][m]=[]
            for p in ps:
                absolute=p.resolve()
                if absolute not in mapped:
                    rel=Path('inputs')/f'{len(mapped)+1:02d}_{p.name}'
                    before=digest(absolute);shutil.copyfile(absolute,stage/rel)
                    if digest(stage/rel)!=before or digest(absolute)!=before:raise RuntimeError(f'Source changed while copying: {p}')
                    mapped[absolute]=str(rel)
                    doc['files'].append(dict(original=str(absolute),copy=str(rel),sha256=before,size_bytes=absolute.stat().st_size))
                doc['methods'][m].append(mapped[absolute])
        for name,p in plans.items():
            rel=Path('preparation')/name;before=digest(p);shutil.copyfile(p,stage/rel)
            if digest(stage/rel)!=before:raise RuntimeError(f'Preparation changed while copying: {p}')
            doc['files'].append(dict(original=str(p.resolve()),copy=str(rel),sha256=before,size_bytes=p.stat().st_size))
            doc['preparation'][name]=str(rel)
        (stage/'source_manifest.json').write_text(json.dumps(doc,indent=2)+'\n')
        if root.exists():root.rmdir()  # empty directory only
        stage.rename(root)
    print('Copied prior inputs into:',root,flush=True)
    return root,doc


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
    p.add_argument('--preparation-root',type=Path,default=Path('cross_subject_no_warp/outputs/01_prepare_experiment'))
    p.add_argument('--source-root',type=Path,default=Path('cross_subject_no_warp/source/02_parameter_grids'))
    p.add_argument('--output-root',type=Path,default=Path('cross_subject_no_warp/outputs/02_select_loso_parameters'))
    p.add_argument('--grid',action='append',default=[],help='Initial snapshot override METHOD=CSV; repeat for multiple CSVs')
    p.add_argument('--error-column',action='append',default=[],help='METHOD=COLUMN, only if error-column selection is ambiguous')
    p.add_argument('--reliability',choices=['primary-all-pass','configuration-all-pass'],default='primary-all-pass',help='Default preserves Script26 strict primary-value screening, within each training fold only')
    p.add_argument('--overwrite',action='store_true',help='Regenerate selection outputs, never replace the source snapshot')
    a=p.parse_args();out=a.output_root
    if out.exists() and any(out.iterdir()) and not a.overwrite:raise FileExistsError('Output directory nonempty; use --overwrite to recompute from the same snapshot')
    root,doc=snapshot(a);folds,cases=load_plan(root,doc);error_overrides=kv(a.error_column)
    out.mkdir(parents=True,exist_ok=True)
    # Invalidate readiness before any new work, including an interrupted overwrite.
    state={'status':'IN_PROGRESS','source_root':str(root),'reliability':a.reliability}
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
        parameter_use='same fold configuration for within and single-reference localization',
        error_column_overrides=error_overrides,allowed_experiments={m:sorted(v) for m,v in ALLOWED.items()})
    (out/'07_selection_manifest.json').write_text(json.dumps(state,indent=2)+'\n')
    shutil.copyfile(root/'source_manifest.json',out/'08_source_manifest.json')
    (out/'README.txt').write_text('''Training-only LOSO parameter selection from frozen prior-grid copies.
The first invocation copies the full raw grids, including held-out outcomes, for provenance. Selection filters out the held-out rows before all parameter repair, reliability screening, duplicate handling, and scoring. No held-out error summary is produced here.
Default reliability=primary-all-pass follows Script26 strict primary-value checks on the SIX training subjects only, plus per-configuration all-pass coverage. Configuration-all-pass is a separate optional policy and must be reported as such.
Rank by mean of participant medians, then pooled median, pooled mean, deterministic key. Equal-participant weighting follows the earlier LOSO script, not Script26 pooled-median scoring.
All 84 method/fold selections must exist before Script03. No eligible configuration produces an explicit missing-selection record and nonzero exit; no fallback to globally selected parameters.
Exact duplicate status/error observations are collapsed; conflicting duplicates invalidate their training configuration. Missing settings are repaired only from unambiguous same-source/configuration/montage TRAINING rows. No values are borrowed from the held-out subject. Failed rows with unresolved non-primary settings remain explicit incomplete evidence and conservatively block their ID-encoded primary value under either reliability policy; successful rows with unresolved settings stop selection.
Gamma depth-test experiments remain outside the inherited experiment allowlist. MxNE uses extension inputs as in Script26. All input copies are preserved byte-for-byte; allowlist filtering is applied during analysis.
This is parameter transfer from prior within-subject errors. The reference head does not determine which training subjects select parameters: all six train subjects contribute, even though one reference head is used later.
Prior grids and their search ranges were developed using this dataset. This is retrospective LOSO re-selection, not an untouched prospective validation. Candidate search-space history and known-hemisphere peak restrictions must be reported.
The source snapshot is immutable and SHA-256 verified. --overwrite replaces outputs only; choose a new --source-root for different source bytes. No original results are changed.
''')
    print(f'Selected {len(winners)}/84 method/fold configurations. Saved: {out}')
    if not complete:raise SystemExit('Some folds have no eligible configuration. Inspect 02_fold_selection_coverage.csv; do not run reconstruction yet.')

if __name__=='__main__':main()

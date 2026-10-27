#!/usr/bin/env python3
"""Script 39 — strict all-PASS parameter selection for 12 Localize-MI methods.

No inverse computations; all results are selected from EXISTING grid CSVs.

Strategies:
  single_reference: choose one configuration using only one reference subject;
                    later apply to every other subject (Script 40).
  loso:             choose one configuration using the SIX reference subjects,
                    excluding the designated test subject completely.
  global:           descriptive baseline selected using all seven subjects;
                    NOT an unbiased held-out benchmark.

Eligibility is strict: a configuration must have an unambiguous PASS row, with a
finite localization error, for EVERY expected run in the reference training set.
Any missing, FAIL, nonfinite, conflicting duplicate, or ambiguous configuration
row disqualifies that configuration in that fold. Missing *test* rows are NOT
used in selection; Script 40 will report them separately.

Aggregate selection score: average of each reference subject's median run error
(equal-subject weighting). Tie-break: pooled median, pooled mean, montage, config.
This policy does not use test-subject results or ground truth.

Known locations:
  Script 23 consolidated CSV (Script 24):
    outputs/23_full_next_method_grid/23_all_next_methods_combined.csv
  Script 25 all-PASS output (eligible complete-61 configs only):
    outputs/25_next_method_audit/07_complete_all_pass.csv
  Script 23 raw per-method CSV:
    outputs/23_full_next_method_grid/<method>/grid_results.csv

For four newer methods, use the Script-24 combined master (then independently
validate all-PASS per fold). The Script-25 eligible CSV is an OPTIONAL source,
but using only it will exclude configs that fail on an omitted test subject,
which is unnecessarily strict for LOSO. Thus Script 24 master is the default.
Script 24 may combine extensions; Script 25's curated allowlist is reproduced
here to exclude experimental gamma depth tests. Original four use outputs/grid_all_runs/grid_results.csv; Script-15 methods
use outputs/15_full_additional_method_grid/<slug>/grid_results.csv.
All paths are based on the Script-26 input declarations.

Examples:
  python cross_subject/39_loso_parameter_selection.py \\
      --grid MNE=outputs/09_full_parameter_grid/mne/grid_results.csv \\
      --grid dSPM=/actual/path/to/dspm_grid.csv \\
      --grid sLORETA=/actual/path/to/sloreta_grid.csv \\
      --grid eLORETA=/actual/path/to/eloreta_grid.csv \\
      --grid Continuous-ECD=/actual/path/to/ecd_grid.csv \\
      --grid RAP-MUSIC=/actual/path/to/music_grid.csv \\
      --grid LCMV=/actual/path/to/lcmv_grid.csv \\
      --grid MxNE=/actual/path/to/mxne_grid.csv

  # Audit SESAME only, starting from original raw Script-23 CSV:
  python cross_subject/39_loso_parameter_selection.py \\
      --grid SESAME=outputs/23_full_next_method_grid/sesame/grid_results.csv \\
      --methods SESAME

  # Audit four newer methods after Script 24 produces the master:
  python cross_subject/39_loso_parameter_selection.py \\
      --methods irMxNE Gamma-MAP Champagne SESAME

Outputs under cross_subject/outputs/39_loso_parameter_selection/.
Never overwrites an input CSV. No dependency on MNE or sesameeg.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
from pathlib import Path
import pandas as pd
import numpy as np

METHODS = ['MNE','dSPM','sLORETA','eLORETA','Continuous-ECD','RAP-MUSIC',
           'LCMV','MxNE','irMxNE','Gamma-MAP','Champagne','SESAME']
ALIASES = {re.sub('[^a-z0-9]','',m.lower()):m for m in METHODS}
ALIASES.update({'ecd':'Continuous-ECD','continuousdipole':'Continuous-ECD',
                'music':'RAP-MUSIC','gamma':'Gamma-MAP','wmne':'MNE'})
NEW_METHODS = {'irMxNE','Gamma-MAP','Champagne','SESAME'}
ALLOWED_SOURCE_EXPERIMENTS = {
    'irMxNE': {'23_full_next_method_grid','23_irmxne_alpha15_extension','23_irmxne_alpha20_extension'},
    'Gamma-MAP': {'23_full_next_method_grid','23_gamma_alpha_extension',
                  '23_gamma_alpha_low_extension','23_gamma_xyz_true'},
    'Champagne': {'23_full_next_method_grid','23_champagne_pruning_extension'},
    'SESAME': {'23_full_next_method_grid','23_sesame_fs_extension'},
}
PARAMS = {
    'MNE': ['method_origin','montage','loose','depth','snr','lambda2'],
    'dSPM': ['method_origin','montage','loose','depth','snr','lambda2'],
    'sLORETA': ['method_origin','montage','loose','depth','snr','lambda2'],
    'eLORETA': ['method_origin','montage','loose','depth','snr','lambda2'],
    'Continuous-ECD':['montage','configuration_id'],
    'RAP-MUSIC':['montage','configuration_id'],
    'LCMV':['montage','configuration_id'],
    'MxNE':['montage','configuration_id'],
    'irMxNE':['montage','irmxne_alpha','irmxne_loose','irmxne_depth','irmxne_iterations','irmxne_random_state'],
    'Gamma-MAP':['montage','gamma_alpha','gamma_loose','gamma_depth','gamma_xyz_same_gamma',
                 'gamma_update_mode','gamma_maxit','gamma_tol'],
    'Champagne':['montage','champagne_alpha','champagne_prune','champagne_pruning_thresh',
                 'champagne_convergence_criterion','champagne_max_iter','champagne_noise_covariance_supplied'],
    'SESAME':['montage','sesame_fs','sesame_n_parts','sesame_max_n_dips','sesame_lam',
              'sesame_hyper_q','sesame_random_seed','sesame_noise_covariance_supplied'],
}
# These columns are different from controlled parameters; never group on them.
METRIC_COLS = ['localization_distance_mm','localization_error_mm','selected_localization_error_mm',
               'distance_mm','error_mm']
# Generic columns whose names are usually controlled inputs. New/unknown columns
# require explicit --params METHOD=col1,col2 to ensure they're not omitted.


def canonical(name):
    k = re.sub('[^a-z0-9]','',str(name).lower())
    if k not in ALIASES: raise ValueError(f'Unknown method {name!r}')
    return ALIASES[k]

def normalize(value):
    if pd.isna(value): return '<MISSING>'
    s=str(value).strip()
    if s.lower() in ('true','false'): return s.lower()
    try:
        if re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?',s):
            return format(float(s),'.12g')
    except ValueError: pass
    return s

def parse_kv(values, sep='='):
    result={}
    for v in values:
        if sep not in v: raise ValueError(f'Expected METHOD{sep}VALUE: {v}')
        m,p=v.split(sep,1)
        m=canonical(m)
        result.setdefault(m,[]).append(p)
    return result

def pick_error_column(df, forced=None):
    if forced:
        if forced not in df: raise ValueError(f'Error column missing: {forced}')
        return forced
    available=[c for c in METRIC_COLS if c in df and df[c].notna().any()]
    if not available: raise ValueError(f'No localization error column among {METRIC_COLS}')
    if len(available)>1:
        raise ValueError(f'Multiple error columns {available}; specify --error-column METHOD=column')
    return available[0]

def pathload(path, assigned_method):
    frame=pd.read_csv(path,low_memory=False)
    if frame.empty: raise ValueError(f'Empty CSV: {path}')
    if 'method' in frame:
        frame['_canonical_method'] = frame['method'].map(lambda s: canonical(s) if pd.notna(s) else '<MISSING>')
        frame=frame.loc[frame._canonical_method == assigned_method].copy()
    else:
        frame['_canonical_method']=assigned_method
    if frame.empty: raise ValueError(f'No {assigned_method} rows in {path}')
    frame['_input_file']=str(path)
    return frame

def run(args):
    root=Path(args.project_root).expanduser().resolve()
    output=Path(args.output)
    if not output.is_absolute(): output=root/output
    output.mkdir(parents=True,exist_ok=True)
    selected=[canonical(m) for m in args.methods] if args.methods else METHODS
    selected=list(dict.fromkeys(selected))
    grids=parse_kv(args.grid)
    parameters=parse_kv(args.params)
    error_columns=parse_kv(args.error_column)
    combined=root/'outputs/23_full_next_method_grid/23_all_next_methods_combined.csv'
    raw_new=root/'outputs/23_full_next_method_grid'
    original_grid=root/'outputs/grid_all_runs/grid_results.csv'
    script15_root=root/'outputs/15_full_additional_method_grid'
    script15_slugs={'Continuous-ECD':'continuous_ecd','MxNE':'mxne',
                    'RAP-MUSIC':'rap_music','LCMV':'lcmv'}
    out_audit=[]; out_config=[]; out_subject=[]; out_loso=[]; out_global=[]
    conflicts=[]; sources=[]; run_sets={}; eligible_data={}

    # Discover canonical cases ONLY from the data actually loaded. In a strict 12-
    # method run, these must agree and include the requested number of 61 runs.
    input_frames={}
    for method in selected:
        if method in grids:
            paths=[Path(p).expanduser() for p in grids[method]]
            paths=[p if p.is_absolute() else root/p for p in paths]
        elif method in NEW_METHODS:
            paths=[combined] if combined.exists() else [raw_new / {
                'irMxNE':'irmxne','Gamma-MAP':'gamma_map','Champagne':'champagne','SESAME':'sesame'}[method]/'grid_results.csv']
        elif method in {'MNE','dSPM','sLORETA','eLORETA'}:
            paths=[original_grid]
        else:
            paths=[script15_root/script15_slugs[method]/'grid_results.csv']
        if not all(path.is_file() for path in paths) and args.allow_missing_methods:
            print(f'SKIP {method}: grid file missing: {[str(p) for p in paths if not p.is_file()]}')
            continue
        frames=[]
        for path in paths:
            if not path.is_file(): raise FileNotFoundError(f'{method}: {path}')
            frame=pathload(path,method)
            if ('source_experiment' in frame and method in ALLOWED_SOURCE_EXPERIMENTS
                    and not args.include_unapproved_extensions):
                frame=frame[frame.source_experiment.isin(ALLOWED_SOURCE_EXPERIMENTS[method])].copy()
            frames.append(frame)
            sources.append({'method':method,'path':str(path),'rows_in_method':len(frame)})
        df=pd.concat(frames,ignore_index=True)
        must=['subject','run','status']
        missing=[x for x in must if x not in df]
        if missing: raise ValueError(f'{method}: missing required columns {missing}')
        for field in ['subject','run','status']:
            df[field]=df[field].astype(str).str.strip()
        df['status']=df.status.str.upper()
        df['_run_key']=df.subject+'|'+df.run
        runs=set(zip(df.subject,df.run))
        run_sets[method]=runs
        if len(runs)!=args.expected_runs:
            raise ValueError(f'{method}: {len(runs)} distinct runs, expected {args.expected_runs}. '+
                             'Full grid missing or wrong input. Do not select best parameters.')
        if len(set(df.subject))!=args.expected_subjects:
            raise ValueError(f'{method}: found {df.subject.nunique()} subjects, expected {args.expected_subjects}')
        error_col=error_columns.get(method,[None])[0]
        ec=pick_error_column(df,error_col)
        df['_error']=pd.to_numeric(df[ec],errors='coerce')
        fields=parameters.get(method,[None])[0]
        if fields:
            fields=[x.strip() for x in fields.split(',') if x.strip()]
        else:
            fields=[c for c in PARAMS[method] if c in df and df[c].notna().any()]
        if not fields: raise ValueError(f'{method}: cannot identify configuration fields; provide --params')
        if 'montage' in df and 'montage' not in fields:
            raise ValueError(f'{method}: montage is present but not in configuration fields')
        # Catch unintentionally excluded columns likely to be method settings.
        likely=[c for c in df if (c.lower().startswith(('sesame_','gamma_','irmxne_','champagne_','mxne_'))
                  and c not in fields and df[c].notna().any() and c not in PARAMS[method]
                  and not any(token in c for token in ('active_','runtime','noise_std','target_peak','estimated_',
                  'gof','dispersion','explained_','source_','signature')))]
        if likely: print(f'WARNING {method}: review possible untracked parameter fields: {likely}')
        for col in fields:
            if col not in df: raise ValueError(f'{method}: requested parameter column absent: {col}')
        # Repair missing fixed input values on FAILs only where unambiguous within
        # same file/configuration/montage. This follows Script 25's repair logic.
        if 'configuration_id' in df:
            groupkeys=['_input_file','configuration_id']+(['montage'] if 'montage' in df else [])
            if 'source_file' in df: groupkeys=['source_file','configuration_id']+(['montage'] if 'montage' in df else [])
            for col in fields:
                tmp=df.groupby(groupkeys,dropna=False)[col]
                num=tmp.transform('nunique')
                exemplar=tmp.transform('first')
                mask=df[col].isna() & (num==1) & exemplar.notna()
                df.loc[mask,col]=exemplar.loc[mask]
        df['_params_json']=[json.dumps({k:normalize(v) for k,v in zip(fields,row)},sort_keys=True)
                            for row in df[fields].itertuples(index=False,name=None)]
        df['_config_key']=df._params_json.map(lambda x:hashlib.sha256(x.encode()).hexdigest()[:20])
        # An absent parameter must NOT create an eligible configuration.
        df['_missing_parameters']=df[fields].isna().any(axis=1)
        input_frames[method]=df
        (output/f'39_{method.replace("/","_").replace("-","_")}_configuration_fields.json').write_text(
             json.dumps({'method':method,'input_paths':[str(x) for x in paths],
                         'configuration_fields':fields,'error_column':ec},indent=2)+'\n')

    if not input_frames: raise RuntimeError('No grid inputs were loaded')
    canonical_cases=next(iter(run_sets.values()))
    for method,cases in run_sets.items():
        if cases!=canonical_cases:
            raise ValueError(f'{method}: run identity differs from other input grids; '+
                             f'missing {sorted(canonical_cases-cases)[:4]}, extra {sorted(cases-canonical_cases)[:4]}')
    expected_by_subject={s:set(r for ss,r in canonical_cases if ss==s)
                         for s in sorted(set(s for s,_ in canonical_cases))}
    subjects=sorted(expected_by_subject)
    all_parts=[]; archived_canonical=[]; archived_full_pass=[]
    for method,df in input_frames.items():
        # Conservative duplicate checking: identical duplicate rows are collapsed;
        # duplicates with discrepancies invalidate that config/run in every fold.
        canonical_rows=[]
        for (ck,s,r),g in df.groupby(['_config_key','subject','run'],sort=False):
            if len(g)>1:
                signatures={(str(st),normalize(err),str(params))
                            for st,err,params in g[['status','_error','_params_json']].itertuples(index=False,name=None)}
                conflict=len(signatures)>1
                conflicts.append({'method':method,'configuration_key':ck,'subject':s,'run':r,
                                  'rows':len(g),'conflict':conflict})
                if conflict: continue
            canonical_rows.append(g.index[0])
        canonical_df=df.loc[canonical_rows].copy()
        ambiguous={(x['configuration_key'],x['subject'],x['run']) for x in conflicts
                   if x['method']==method and x['conflict']}
        canonical_df['_valid']=(canonical_df.status.eq('PASS') &
                               canonical_df._error.notna() & np.isfinite(canonical_df._error) &
                               (canonical_df._error>=0) & ~canonical_df._missing_parameters)
        # Preserve a reusable self-contained copy, with source fields and stable IDs.
        # Conflicting duplicated run keys remain excluded from the canonical copy.
        archived_canonical.append(canonical_df.assign(
            selection_method=method, selection_configuration_key=canonical_df['_config_key'],
            selection_configuration_parameters=canonical_df['_params_json'],
            selection_error_mm=canonical_df['_error']))
        configmap=canonical_df.drop_duplicates('_config_key').set_index('_config_key')['_params_json'].to_dict()
        for ck,parameters_json in configmap.items():
            group=canonical_df[canonical_df._config_key==ck]
            per_sub={}
            for subject in subjects:
                sub=group[group.subject==subject]
                found=set(sub.run)
                missing=expected_by_subject[subject]-found
                bad=sub[~sub._valid]
                bad_runs=set(bad.run)
                ambiguous_runs={r for c,s,r in ambiguous if c==ck and s==subject}
                ok=(not missing and not bad_runs and not ambiguous_runs and
                    len(sub)==len(expected_by_subject[subject]))
                per_sub[subject]={
                    'eligible':bool(ok), 'median':float(sub._error.median()) if ok else np.nan,
                    'errors':sub._error.to_list() if ok else [],
                    'missing':len(missing), 'bad':len(bad_runs), 'conflicting':len(ambiguous_runs),
                }
                out_audit.append({'method':method,'configuration_key':ck,'subject':subject,
                                  'expected_runs':len(expected_by_subject[subject]),
                                  'observed_runs':len(sub),'missing_runs':len(missing),
                                  'nonpass_or_invalid_runs':len(bad_runs),
                                  'duplicate_conflict_runs':len(ambiguous_runs),
                                  'eligible_all_pass':ok,'median_error_mm':per_sub[subject]['median'],
                                  'configuration_parameters':parameters_json})
            if all(per_sub[s]['eligible'] for s in subjects):
                archived_full_pass.append(group.assign(
                    selection_method=method, selection_configuration_key=ck,
                    selection_configuration_parameters=parameters_json,
                    selection_error_mm=group['_error']))
            out_config.append({'method':method,'configuration_key':ck,
                               'configuration_parameters':parameters_json,
                               'fully_all_pass_61':all(per_sub[s]['eligible'] for s in subjects),
                               'eligible_subject_count':sum(per_sub[s]['eligible'] for s in subjects)})
            for strategy,test,refs in ([('single_reference',None,[s]) for s in subjects]+
                                       [('loso',s,[r for r in subjects if r!=s]) for s in subjects]+
                                       [('global',None,subjects)]):
                if not all(per_sub[s]['eligible'] for s in refs): continue
                medians=[per_sub[s]['median'] for s in refs]
                errors=[x for s in refs for x in per_sub[s]['errors']]
                all_parts.append({'method':method,'strategy':strategy,
                                  'test_subject':test if test else '',
                                  'reference_subject':refs[0] if strategy=='single_reference' else '',
                                  'reference_subjects':';'.join(refs),
                                  'configuration_key':ck,'configuration_parameters':parameters_json,
                                  'reference_subject_count':len(refs),
                                  'reference_run_count':len(errors),
                                  'score_equal_subject_median_mm':float(np.mean(medians)),
                                  'pooled_median_mm':float(np.median(errors)),
                                  'pooled_mean_mm':float(np.mean(errors))})

    scores=pd.DataFrame(all_parts)
    if scores.empty: raise RuntimeError('No eligible all-PASS configurations found')
    groupby=['method','strategy','test_subject','reference_subject']
    scores=scores.sort_values(groupby+['score_equal_subject_median_mm','pooled_median_mm',
                                       'pooled_mean_mm','configuration_parameters','configuration_key'],
                              kind='stable')
    scores['selection_rank']=scores.groupby(groupby,dropna=False).cumcount()+1
    winners=scores[scores.selection_rank==1].copy()
    # Explicitly include six test rows for every single-reference selection
    # without selecting using any test outcomes.
    single=winners[winners.strategy=='single_reference'].copy()
    pairs=[]
    for row in single.to_dict('records'):
        for test in subjects:
            if test!=row['reference_subject']:
                pairs.append({**row,'test_subject':test})
    single_pairs=pd.DataFrame(pairs)
    for method in input_frames:
        required_single=len(subjects)
        required_loso=len(subjects)
        counts=winners[winners.method==method].strategy.value_counts().to_dict()
        if counts.get('single_reference',0)!=required_single or counts.get('loso',0)!=required_loso:
            print(f'WARNING {method}: selected single-reference {counts.get("single_reference",0)}/{required_single} '+
                  f'and LOSO {counts.get("loso",0)}/{required_loso}; see audit output')
    def save(df,name):
        df.to_csv(output/name,index=False)
    save(pd.DataFrame(sources),'00_input_sources.csv')
    save(pd.DataFrame(conflicts,columns=['method','configuration_key','subject','run','rows','conflict']),
         '01_duplicate_audit.csv')
    save(pd.DataFrame(out_audit),'02_subject_configuration_audit.csv')
    save(pd.DataFrame(out_config),'03_configuration_summary.csv')
    save(scores,'04_all_eligible_candidate_scores.csv')
    save(winners[winners.strategy=='single_reference'],'05_single_reference_selected.csv')
    save(single_pairs,'06_single_reference_42_pairs.csv')
    save(winners[winners.strategy=='loso'],'07_loso_selected.csv')
    save(winners[winners.strategy=='global'],'08_global_fixed_baseline_LEAKY.csv')
    # All 12 methods in one reusable archive: only 61-run complete, all-PASS
    # configurations are included. Original per-run columns and provenance kept.
    if archived_full_pass:
        full_pass=pd.concat(archived_full_pass,ignore_index=True,sort=False)
        if not full_pass['_valid'].all():
            raise AssertionError('Archive contains a non-PASS or invalid row')
        save(full_pass,'10_all_12_methods_full_61_all_pass.csv')
    else:
        raise RuntimeError('No 61-run all-PASS configurations to archive')
    # Important for Script 40: some reference-eligible LOSO configs may FAIL
    # on test subject; retain their test rows as well, not only global all-PASS.
    save(pd.concat(archived_canonical,ignore_index=True,sort=False),
         '11_all_12_methods_canonical_grid_backup.csv')
    manifest={'version':1,'script':'39_loso_parameter_selection.py',
              'methods_loaded':list(input_frames),'subjects':subjects,
              'expected_run_counts':{s:len(v) for s,v in expected_by_subject.items()},
              'expected_runs':args.expected_runs,'selection_score':'mean(subject run-level medians)',
              'require_all_pass':True, 'input_sources':sources,
              'full_61_all_pass_archive':'10_all_12_methods_full_61_all_pass.csv',
              'full_canonical_archive':'11_all_12_methods_canonical_grid_backup.csv',
              'warning':'Global fixed baseline is selected using all subjects; NOT held-out',
              'warning2':'Only precomputed configurations in supplied CSV are eligible.'}
    (output/'09_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'\nScript 39: selected {len(winners)} method/strategy/fold winners; output: {output}')
    print(winners.groupby(['method','strategy']).size().unstack(fill_value=0).to_string())


def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[2])
    ap.add_argument('--output',type=Path,default=Path('cross_subject/outputs/39_loso_parameter_selection'))
    ap.add_argument('--grid',action='append',default=[],metavar='METHOD=PATH',
                    help='Repeat for each method; may repeat a method to concatenate full-grid CSVs')
    ap.add_argument('--params',action='append',default=[],metavar='METHOD=COL1,COL2',
                    help='Override configuration fields after reviewing actual grid schema')
    ap.add_argument('--error-column',action='append',default=[],metavar='METHOD=COL')
    ap.add_argument('--methods',nargs='+',default=METHODS)
    ap.add_argument('--expected-runs',type=int,default=61)
    ap.add_argument('--expected-subjects',type=int,default=7)
    ap.add_argument('--allow-missing-methods',action='store_true')
    ap.add_argument('--include-unapproved-extensions',action='store_true')
    run(ap.parse_args())
if __name__=='__main__': main()

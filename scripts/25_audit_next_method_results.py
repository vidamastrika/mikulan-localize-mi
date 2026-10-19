#!/usr/bin/env python3
"""Audit Script 24's master grid for use in a 12-method comparison.

Does not edit input results. Outputs an audit folder and two eligibility datasets:
  complete_any_status: all 61 distinct reference run keys present, FAIL allowed
  complete_all_pass:  all 61 distinct reference run keys present, every status PASS

Default experiment allowlist excludes gamma_depth_test but includes the
 completed gamma_xyz_true full 61-run experiment. Source experiment names are preserved from Script 24.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
from pathlib import Path
import pandas as pd

DEFAULT_INPUT = Path('outputs/23_full_next_method_grid/23_all_next_methods_combined.csv')
DEFAULT_OUTPUT = Path('outputs/25_next_method_audit')
# Curated FULL-GRID experiments only. Add a folder after verification, not automatically.
ALLOWED = {
    'champagne': {'23_full_next_method_grid', '23_champagne_pruning_extension'},
    'irmxne': {'23_full_next_method_grid', '23_irmxne_alpha15_extension', '23_irmxne_alpha20_extension'},
    'gamma_map': {'23_full_next_method_grid', '23_gamma_alpha_extension', '23_gamma_alpha_low_extension', '23_gamma_xyz_true'},
    'sesame': {'23_full_next_method_grid', '23_sesame_fs_extension'},
}
# Configuration fields: use observed intersection with master CSV. Avoid runtime,
# localization output, fit quality, and covariance measurements, which vary by run.
# Only requested/controlled parameters belong in configuration identities.
# IMPORTANT: do not include realized covariance_method, fit quality, or any
# metric calculated from the data (explained variance, GOF, noise std).
COMMON_PARAMETERS = ['montage']
METHOD_PARAMETERS = {
    'irmxne': ['irmxne_alpha', 'irmxne_loose', 'irmxne_depth',
               'irmxne_iterations', 'irmxne_random_state'],
    'champagne': ['champagne_alpha', 'champagne_prune',
                  'champagne_pruning_thresh',
                  'champagne_convergence_criterion', 'champagne_max_iter',
                  'champagne_noise_covariance_supplied'],
    'gamma_map': ['gamma_alpha', 'gamma_loose', 'gamma_depth',
                  'gamma_xyz_same_gamma', 'gamma_update_mode',
                  'gamma_maxit', 'gamma_tol'],
    'sesame': ['sesame_fs', 'sesame_n_parts', 'sesame_max_n_dips',
               'sesame_lam', 'sesame_hyper_q', 'sesame_random_seed',
               'sesame_noise_covariance_supplied'],
}
# Explicitly map experiment settings, not outcomes. Additional genuinely
# varied settings must be reviewed and added here if present in source scripts.

def normalize_method(x):
    return str(x).strip().lower().replace('-', '_')

def value(x):
    if pd.isna(x): return '<NA>'
    s = str(x).strip()
    if s.lower() in ('true', 'false'): return s.lower()
    try:
        if re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?', s):
            return format(float(s), '.12g')
    except (ValueError, OverflowError): pass
    return s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    ap.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument('--expected-runs', type=int, default=61)
    args = ap.parse_args()
    df = pd.read_csv(args.input, low_memory=False)
    needed = {'subject','run','method','status','source_experiment','source_file'}
    missing = needed - set(df.columns)
    if missing: raise SystemExit(f'Missing required columns: {sorted(missing)}')
    args.output.mkdir(parents=True, exist_ok=True)
    df['method'] = df.method.map(normalize_method)
    df['status'] = df.status.astype(str).str.upper().str.strip()
    df['source_experiment'] = df.source_experiment.astype(str).str.strip()
    # Keep every raw row in source summary, including excluded testing folders.
    origin = (df.groupby(['method','source_experiment','status'], dropna=False)
              .size().rename('rows').reset_index())
    origin['included_in_full_grid'] = origin.apply(
        lambda r: r.source_experiment in ALLOWED.get(r.method, set()), axis=1)
    origin.to_csv(args.output/'01_source_inventory.csv', index=False)
    included = df[df.apply(lambda r: r.source_experiment in ALLOWED.get(r.method, set()), axis=1)].copy()
    excluded = df.drop(index=included.index)
    excluded.to_csv(args.output/'02_excluded_rows.csv', index=False)
    if included.empty: raise SystemExit('No allowed rows found. Check source_experiment names.')

    # Construct run identity independently of montage and config.
    for field in ['subject','run']:
        included[field] = included[field].astype(str).str.strip()
    included['_run_key'] = included['subject'] + '\x1f' + included['run']
    reference = set(included.loc[included.method=='irmxne', '_run_key'].unique())
    if len(reference) != args.expected_runs:
        raise SystemExit(f'irMxNE reference has {len(reference)} run keys, not {args.expected_runs}. '
                         'Review subject/run formatting or pass --expected-runs.')
    # Never silently erase missing/failed rows.
    for method in ALLOWED:
        observed = set(included.loc[included.method==method, '_run_key'].unique())
        if not observed <= reference:
            print(f'WARNING: {method} has {len(observed-reference)} run keys outside irMxNE reference')

    # Fill parameter fields missing on failed runs ONLY when the same
    # source file + configuration_id + montage uniquely identifies them.
    # Source path is essential: configuration_id may be reused across jobs.
    used_fields = {}
    for method, subset in included.groupby('method', sort=False):
        cols = [c for c in COMMON_PARAMETERS + METHOD_PARAMETERS[method]
                if c in subset.columns and subset[c].notna().any()]
        if 'montage' not in cols or len(cols) == 1:
            raise SystemExit(f'Cannot identify configuration input parameters for {method}: {cols}')
        used_fields[method] = cols

    # Record parameter columns, so analysis can verify the intended identity.
    (args.output/'00_configuration_fields.json').write_text(json.dumps(used_fields, indent=2)+'\n')

    # Conservative repair for missing input parameters on failed observations.
    # If a grouping has multiple distinct values, do not infer a missing value.
    repair_log = []
    for method, subset in included.groupby('method', sort=False):
        cols = [c for c in used_fields[method] if c != 'montage']
        if 'configuration_id' not in included.columns:
            break
        keys = ['source_file', 'configuration_id', 'montage']
        for col in cols:
            non_null = included.loc[subset.index, keys + [col]]
            counts = non_null.groupby(keys, dropna=False)[col].transform('nunique')
            values = non_null.groupby(keys, dropna=False)[col].transform('first')
            target = subset.index[included.loc[subset.index, col].isna() & (counts == 1) & values.notna()]
            if len(target):
                included.loc[target, col] = values.loc[target]
                repair_log.append({'method': method, 'field': col, 'repaired_rows': len(target)})
    pd.DataFrame(repair_log, columns=['method','field','repaired_rows']).to_csv(
        args.output/'00_missing_parameter_repairs.csv', index=False)

    # Vectorized canonical fingerprints: efficient for >100k rows.
    signatures = []
    for method, subset in included.groupby('method', sort=False):
        cols = used_fields[method]
        normalized = subset[cols].apply(lambda series: series.map(value))
        payloads = normalized.to_dict(orient='records')
        signatures.extend(
            (idx,
             hashlib.sha256(json.dumps({'method': method, **payload},
                                       sort_keys=True).encode()).hexdigest()[:20],
             json.dumps({'method': method, **payload}, sort_keys=True))
            for idx, payload in zip(subset.index, payloads)
        )
    sigdf = pd.DataFrame(signatures, columns=['index','configuration_key',
                                              'configuration_parameters']).set_index('index')
    included = included.join(sigdf)

    # One (method, parameter combination, subject, run) should occur exactly once.
    key = ['method','configuration_key','_run_key']
    counts = included.groupby(key).size().rename('repetitions').reset_index()
    dupkeys = counts[counts.repetitions > 1][key]
    duplicates = included.merge(dupkeys, on=key, how='inner') if len(dupkeys) else included.iloc[:0].copy()
    duplicates.drop(columns=['_run_key'],errors='ignore').to_csv(args.output/'03_duplicate_rows.csv',index=False)
    # Conflict check: same config/run repeated but not matching status or distances.
    error_cols = [c for c in ['selected_localization_error_mm','localization_error_mm',
                               'distance_mm','error_mm','localization_distance_mm'] if c in included.columns]
    check_cols = ['status'] + error_cols
    collisions = []
    if not duplicates.empty:
        for groupkey, sub in duplicates.groupby(key):
            conflicting = any(sub[c].map(value).nunique(dropna=False)>1 for c in check_cols)
            collisions.append(dict(method=groupkey[0],configuration_key=groupkey[1],
                                   run_key=groupkey[2].replace('\x1f',' / '),
                                   repetitions=len(sub),conflicting_result=conflicting,
                                   source_files='; '.join(sorted(sub.source_file.astype(str).unique()))))
    pd.DataFrame(collisions,columns=['method','configuration_key','run_key','repetitions',
                                     'conflicting_result','source_files']).to_csv(
        args.output/'04_duplicate_conflicts.csv',index=False)

    # Eligibility is assessed BEFORE collapsing repetitions. An unresolved conflict
    # makes a configuration ineligible for either report until reviewed.
    conflict_pairs = {(x['method'],x['configuration_key']) for x in collisions if x['conflicting_result']}
    rows = []
    for (method,config), sub in included.groupby(['method','configuration_key']):
        found = set(sub['_run_key'])
        # all observed statuses at a run must be PASS for all-pass eligibility.
        run_status = sub.groupby('_run_key')['status'].apply(lambda s: set(s))
        failed_keys = sum('FAIL' in s for s in run_status)
        other_keys = sum(not s <= {'PASS','FAIL'} for s in run_status)
        not_all_pass = sum(s != {'PASS'} for s in run_status)
        missing_ref = reference - found
        foreign = found - reference
        full = len(missing_ref)==0 and len(foreign)==0 and len(found)==args.expected_runs
        conflict = (method,config) in conflict_pairs
        rows.append({
            'method':method,'configuration_key':config,
            'configuration_parameters':sub.configuration_parameters.iloc[0],
            'source_experiments':'; '.join(sorted(sub.source_experiment.unique())),
            'raw_rows':len(sub),'distinct_run_keys':len(found),
            'missing_reference_runs':len(missing_ref),'foreign_runs':len(foreign),
            'failed_run_keys':failed_keys,'other_status_run_keys':other_keys,
            'not_all_pass_run_keys':not_all_pass,
            'repeated_rows':len(sub)-len(found),
            'unresolved_duplicate_conflict':conflict,
            'complete_61_runs':full,
            'eligible_complete_any_status':full and not conflict and other_keys==0,
            'eligible_complete_all_pass':full and not conflict and not_all_pass==0,
            'missing_runs':'; '.join(k.replace('\x1f',' / ') for k in sorted(missing_ref)),
        })
    audit = pd.DataFrame(rows).sort_values(['method','configuration_key'])
    audit.to_csv(args.output/'05_configuration_completeness.csv', index=False)
    print('\nCONFIGURATION ELIGIBILITY')
    print(audit.groupby('method')[['complete_61_runs','eligible_complete_any_status',
                                   'eligible_complete_all_pass']].sum().to_string())
    print(f'\nDuplicate rows: {len(duplicates):,}; conflicting duplicated run groups: '
          f'{sum(x["conflicting_result"] for x in collisions)}')

    # For consistent downstream results, choose a single row from equivalent
    # duplicates (prefer PASS, then stable source path); unresolved conflicts are
    # excluded by eligibility anyway. Do not overwrite raw results.
    eligible = audit.set_index(['method','configuration_key'])
    included['_all_any'] = [bool(eligible.loc[(m,c),'eligible_complete_any_status'])
                            for m,c in zip(included.method,included.configuration_key)]
    included['_all_pass'] = [bool(eligible.loc[(m,c),'eligible_complete_all_pass'])
                             for m,c in zip(included.method,included.configuration_key)]
    canonical = included.sort_values(['source_file','source_row'] if 'source_row' in included else ['source_file'])
    canonical = canonical.drop_duplicates(key,keep='first')
    canonical = canonical.drop(columns=['_run_key','_all_any','_all_pass'],errors='ignore')
    accepted = audit.set_index(['method','configuration_key'])
    for label, col in [('06_complete_any_status.csv','eligible_complete_any_status'),
                       ('07_complete_all_pass.csv','eligible_complete_all_pass')]:
        mask = [bool(accepted.loc[(m,c),col]) for m,c in zip(canonical.method,canonical.configuration_key)]
        result = canonical.loc[mask].copy()
        result.to_csv(args.output/label,index=False)
        print(f'{label}: {len(result):,} rows; {result.configuration_key.nunique()} parameter signatures')
    print(f'\nOutputs: {args.output}')
    print('Important: verify 00_configuration_fields.json; any unmapped parameter '
          'must be added before conclusions are drawn.')

if __name__=='__main__': main()

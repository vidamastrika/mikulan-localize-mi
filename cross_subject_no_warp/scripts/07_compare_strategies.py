#!/usr/bin/env python3
"""Script 07 — reproducible master comparison of historical baseline and no-warp strategies.

Run from project root (virtual environment activated):
  python cross_subject_no_warp/scripts/07_compare_strategies.py --inspect-baseline
  python cross_subject_no_warp/scripts/07_compare_strategies.py
  python cross_subject_no_warp/scripts/07_compare_strategies.py --overwrite

Historical snapshot (copied manually once, immutable):
  cross_subject_no_warp/source/07_historical_baseline/13_selected_configuration_results_long.csv
Input: outputs/06_evaluate_localization/03_primary_localization.csv and
04_sensitivity_localization.csv; one historical CSV with 61 selected-configuration
run-level results per method (not aggregate medians, not the full parameter grid).
Historical selected configurations must have 61 PASS / 0 FAIL per method and be
selected strictly by previous Script 25/26 all-pass policy. Historical results
must include an error in mm, subject, run, and method; configure column names
with --baseline-*-col as needed. This script DOES NOT independently verify which
configuration Script 25/26 selected: use the historical selected-results export.

No anatomical alignment, registration, affine/SyN, head-coordinate adjustment,
source recomputation, or interpolation occurs. Cross-subject raw Euclidean
head-coordinate separation is descriptive and NOT validated localization error.
Do not rank all rows by the same numeric metric. Subject dependence and repeated
runs mean naive across-run statistics are descriptive, not inferential.

Outputs (Script 07 directory only):
  01_master_run_level.csv          all cases, harmonized numeric and semantic columns
  02_master_method_summary.csv    mean/median/std/IQR/min/max, counts and valid metric
  03_master_overall_summary.csv   all-method descriptive summary (not pooled accuracy ranking)
  04_baseline_vs_loso_within.csv  paired run-level valid anatomical errors
  05_baseline_vs_loso_within_summary.csv paired descriptive differences
  06_cross_no_warp_pair_summary.csv  paired cross differences where same key exists
  07_validation_report.json      integrity checks, inputs, SHA256, methodological notes

--overwrite atomically backs up previous Script 07 output directory and replaces
only this output directory after complete successful computation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import datetime
import numpy as np
import pandas as pd

ROOT = Path('cross_subject_no_warp')
DEFAULT_IN = ROOT / 'outputs/06_evaluate_localization'
DEFAULT_OUT = ROOT / 'outputs/07_compare_strategies'
DEFAULT_BASELINE = ROOT / 'source/07_historical_baseline/13_selected_configuration_results_long.csv'
CANDIDATE_ERROR_COLS = ('localization_error_mm','localization_distance_mm','localization_error','error_mm','distance_mm','error','localization_distance')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def choose_column(df, supplied, options, label):
    if supplied:
        if supplied not in df: raise ValueError(f'{label} column {supplied!r} not found; columns: {list(df.columns)}')
        return supplied
    for col in options:
        if col in df: return col
    raise ValueError(f'No {label} column among {options}. Columns: {list(df.columns)}')


def stats(x):
    a = pd.to_numeric(pd.Series(x), errors='coerce').dropna().to_numpy(float)
    a = a[np.isfinite(a)]
    if not len(a): return {'n':0,'mean_mm':np.nan,'median_mm':np.nan,'std_mm':np.nan,'q25_mm':np.nan,'q75_mm':np.nan,'iqr_mm':np.nan,'min_mm':np.nan,'max_mm':np.nan}
    q1,q3 = np.percentile(a,[25,75])
    return {'n':len(a),'mean_mm':float(np.mean(a)),'median_mm':float(np.median(a)),
            'std_mm':float(np.std(a,ddof=1)) if len(a)>1 else np.nan,
            'q25_mm':float(q1),'q75_mm':float(q3),'iqr_mm':float(q3-q1),
            'min_mm':float(np.min(a)),'max_mm':float(np.max(a))}


def label_run(s):
    v = str(s).strip()
    if v.startswith('run-'): return v
    if v.isdigit(): return f'run-{int(v):02d}'
    return v


def label_subject(s):
    v = str(s).strip()
    if v.startswith('sub-'): return v
    if v.isdigit(): return f'sub-{int(v):02d}'
    return v


def normalize_new(path):
    df = pd.read_csv(path,low_memory=False)
    required = ['strategy','scenario','method','test_subject','run','condition','status','localization_error_mm','raw_head_separation_mm']
    missing = sorted(set(required)-set(df.columns))
    if missing: raise ValueError(f'Missing columns in {path}: {missing}')
    if df.status.isna().any() or not df.status.isin(['PASS','FAIL']).all(): raise ValueError(f'Unexpected statuses in {path}')
    df['test_subject'] = df.test_subject.map(label_subject)
    df['run'] = df.run.map(label_run)
    df['reference_subject'] = df.get('reference_subject',pd.Series('',index=df.index)).fillna('').astype(str)
    df['metric'] = np.where(df['condition'].eq('within'),'anatomical_localization_error_mm','raw_head_coordinate_separation_mm')
    df['metric_interpretation'] = np.where(df['condition'].eq('within'),'within-subject anatomical distance','UNREGISTERED: diagnostic only, NOT anatomical localization error')
    df['value_mm'] = np.where(df['condition'].eq('within'),pd.to_numeric(df['localization_error_mm'],errors='coerce'),pd.to_numeric(df['raw_head_separation_mm'],errors='coerce'))
    df['metric_valid_as_anatomical_error'] = df['condition'].eq('within') & df['status'].eq('PASS') & np.isfinite(df['value_mm'])
    df['experiment'] = df['strategy'].astype(str) + '_' + df['condition'].astype(str)
    return df


def load_baseline(path,args):
    raw = pd.read_csv(path,low_memory=False)
    method_col = choose_column(raw,args.baseline_method_col,('method','Method'), 'method')
    subject_col = choose_column(raw,args.baseline_subject_col,('subject','test_subject','participant','subject_id'), 'subject')
    run_col = choose_column(raw,args.baseline_run_col,('run','run_id','run_label'), 'run')
    error_col = choose_column(raw,args.baseline_error_col,CANDIDATE_ERROR_COLS,'error (mm)')
    status_col = args.baseline_status_col or ('status' if 'status' in raw.columns else None)
    config_col = args.baseline_config_col or next((c for c in ('configuration_key','config_key','selected_configuration_key') if c in raw), None)
    if status_col and status_col not in raw: raise ValueError('Baseline status column absent')
    if config_col and config_col not in raw: raise ValueError('Baseline config column absent')
    cols = {method_col:'method',subject_col:'test_subject',run_col:'run',error_col:'value_mm'}
    b = raw.rename(columns=cols).copy()
    b['method'] = b.method.astype(str)
    b['test_subject'] = b.test_subject.map(label_subject)
    b['run'] = b.run.map(label_run)
    b['value_mm'] = pd.to_numeric(b.value_mm,errors='coerce')
    if not np.isfinite(b.value_mm).all(): raise ValueError('Baseline errors contain missing/nonfinite values. Provide selected run-level results only.')
    if (b.value_mm < 0).any(): raise ValueError('Negative localization errors in historical baseline')
    b['status'] = raw[status_col].astype(str).str.upper().to_numpy() if status_col else 'PASS'
    if (b.status != 'PASS').any(): raise ValueError('Historical all-pass input contains non-PASS rows')
    keys = ['method','test_subject','run']
    if b.duplicated(keys).any(): raise ValueError('Historical baseline has duplicate method+subject+run. Do not supply full parameter grid; select Script 25/26 config first.')
    if config_col:
        n = raw.groupby(raw[method_col].astype(str))[config_col].nunique(dropna=False)
        if (n>1).any(): raise ValueError(f'Multiple configurations per method: {n[n>1].to_dict()}')
    b['strategy']='Historical'
    b['scenario']='all_pass_selected'
    b['condition']='within'
    b['experiment']='Historical_within'
    b['reference_subject']=''
    b['metric']='anatomical_localization_error_mm'
    b['metric_interpretation']='historical within-subject anatomical distance; comparability depends on matching preprocessing and peak extraction'
    b['metric_valid_as_anatomical_error']=True
    return b,{'method':method_col,'subject':subject_col,'run':run_col,'error':error_col,'status':status_col,'configuration':config_col}


def validate_series(primary,variant,baseline):
    if len(primary)!=5856 or len(variant)!=5856: raise ValueError('Script 06 expected exactly 5856 rows per scenario')
    for df,name in [(primary,'primary'),(variant,'sensitivity')]:
        k=['strategy','method','test_subject','reference_subject','run','condition']
        if df.duplicated(k).any(): raise ValueError(f'{name} contains duplicate keys')
        for strat,n in [('LOSO',1464),('Single-reference',4392)]:
            if len(df[df.strategy.eq(strat)])!=n: raise ValueError(f'{name}/{strat} unexpected schedule')
    k=['strategy','method','test_subject','reference_subject','run','condition']
    a=primary.set_index(k).sort_index()
    b=variant.set_index(k).sort_index()
    if not a.index.equals(b.index): raise ValueError('Primary and sensitivity key sets differ')
    changed=a.status.ne(b.status)
    if changed.sum()!=4: raise ValueError(f'Expected exactly 4 sensitivity PASS restorations, got {changed.sum()}')
    if not ((a.loc[changed,'status']=='FAIL')&(b.loc[changed,'status']=='PASS')).all(): raise ValueError('Unexpected sensitivity status transitions')
    if len(baseline)!=732: raise ValueError(f'Historical selected baseline must have 732 records (12 methods x 61 runs); got {len(baseline)}')
    counts=baseline.groupby('method').size()
    if len(counts)!=12 or (counts!=61).any(): raise ValueError(f'Baseline must have 12 methods, exactly 61 rows each. Counts: {counts.to_dict()}')
    within=primary[(primary.strategy=='LOSO')&(primary.condition=='within')]
    if set(map(tuple,baseline[['method','test_subject','run']].to_numpy()))!=set(map(tuple,within[['method','test_subject','run']].to_numpy())):
        raise ValueError('Historical and LOSO within keys do not match; check method names and run IDs')


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input-root',type=Path,default=DEFAULT_IN)
    p.add_argument('--output-root',type=Path,default=DEFAULT_OUT)
    p.add_argument('--baseline',type=Path,default=DEFAULT_BASELINE,help='Historical selected all-pass snapshot in source/ by default')
    p.add_argument('--inspect-baseline',action='store_true',help='Inspect the default source snapshot (or --baseline override)')
    p.add_argument('--baseline-method-col')
    p.add_argument('--baseline-subject-col')
    p.add_argument('--baseline-run-col')
    p.add_argument('--baseline-error-col')
    p.add_argument('--baseline-status-col')
    p.add_argument('--baseline-config-col')
    p.add_argument('--overwrite',action='store_true')
    args=p.parse_args()
    if args.inspect_baseline:
        x=pd.read_csv(args.baseline,nrows=4)
        print('Baseline columns:',list(x.columns));print(x.to_string(index=False,max_colwidth=45));return
    if not args.baseline.is_file(): p.error(f'Historical baseline snapshot not found: {args.baseline}. Copy the selected all-pass Script 26 CSV into source/07_historical_baseline first.')
    paths={'historical_baseline':args.baseline,'primary':args.input_root/'03_primary_localization.csv','sensitivity':args.input_root/'04_sensitivity_localization.csv'}
    for name,path in paths.items():
        if not path.is_file():raise FileNotFoundError(f'{name}: {path}')
    if args.output_root.exists() and not args.overwrite: raise FileExistsError(f'{args.output_root} exists. Use --overwrite for SCRIPT 07 ONLY')
    primary=normalize_new(paths['primary']); variant=normalize_new(paths['sensitivity'])
    baseline,column_map=load_baseline(args.baseline,args)
    validate_series(primary,variant,baseline)
    main_cols=['strategy','scenario','experiment','method','test_subject','reference_subject','run','condition','status','metric','value_mm','metric_valid_as_anatomical_error','metric_interpretation']
    master=pd.concat([baseline[main_cols],primary[main_cols],variant[main_cols]],ignore_index=True)
    expected=732+5856+5856
    if len(master)!=expected:raise AssertionError('Unexpected total')
    method_rows=[]
    for k,g in master.groupby(['scenario','experiment','method','metric'],sort=True):
        passed=g.status.eq('PASS'); observed= g.value_mm.where(passed)
        method_rows.append(dict(zip(['scenario','experiment','method','metric'],k),scheduled=len(g),pass_count=int(passed.sum()),fail_count=int((~passed).sum()),pass_rate=float(passed.mean()),anatomical_error_valid=bool(g.metric_valid_as_anatomical_error.all() if len(g) else False),**stats(observed)))
    method_df=pd.DataFrame(method_rows)
    all_rows=[]
    for k,g in master.groupby(['scenario','experiment','metric'],sort=True):
        passed=g.status.eq('PASS')
        all_rows.append(dict(zip(['scenario','experiment','metric'],k),scheduled=len(g),pass_count=int(passed.sum()),fail_count=int((~passed).sum()),pass_rate=float(passed.mean()),**stats(g.value_mm.where(passed))))
    overall=pd.DataFrame(all_rows)
    historical=baseline[['method','test_subject','run','value_mm']].rename(columns={'value_mm':'baseline_error_mm'})
    loso=primary[(primary.strategy=='LOSO') & (primary.condition=='within')][['method','test_subject','run','value_mm']].rename(columns={'value_mm':'loso_within_error_mm'})
    paired=historical.merge(loso,on=['method','test_subject','run'],validate='one_to_one')
    if len(paired)!=732:raise ValueError('Baseline-within pairing is not 732')
    paired['loso_minus_baseline_mm']=paired.loso_within_error_mm-paired.baseline_error_mm
    paired_summary=pd.DataFrame([{'method':method,**stats(g.loso_minus_baseline_mm)} for method,g in paired.groupby('method')])
    # No-warp pairs: LOSO has a specific reference per test. Single-reference
    # includes that same pair among six; pairing avoids comparing unmatched references.
    cross=primary[primary.condition.eq('single_reference_no_warp')].copy()
    left=cross[cross.strategy.eq('LOSO')][['method','test_subject','reference_subject','run','value_mm']].rename(columns={'value_mm':'loso_raw_mm'})
    right=cross[cross.strategy.eq('Single-reference')][['method','test_subject','reference_subject','run','value_mm']].rename(columns={'value_mm':'single_reference_raw_mm'})
    cross_pairs=left.merge(right,on=['method','test_subject','reference_subject','run'],validate='one_to_one')
    cross_pairs['single_minus_loso_raw_mm']=cross_pairs.single_reference_raw_mm-cross_pairs.loso_raw_mm
    cross_pairs['interpretation']='DIAGNOSTIC ONLY: coordinate frame mismatch; parameter strategies differ; NOT accuracy'
    rep={'input_paths':{k:str(v) for k,v in paths.items()},'sha256':{k:digest(v) for k,v in paths.items()},'baseline_columns':column_map,'row_counts':{'historical':len(baseline),'primary':len(primary),'sensitivity':len(variant),'master':len(master),'baseline_within_paired':len(paired),'cross_pair_rows':len(cross_pairs)},'methodological_notes':['Historical baseline was selected in-sample; LOSO parameters held out each test subject.','Within baseline and LOSO differences are meaningful only when historical preprocessing and peak extraction match.','Cross no-warp metric is raw head-coordinate separation, NOT anatomically registered localization error.','Paired cross coordinate differences cannot establish accuracy superiority.','Primary and sensitivity duplicate largely the same observations; do not pool as independent samples.','Sensitivity is mixed peak policy: only failed reconstructions are overridden.','Summary statistics pool correlated runs and are descriptive only.','No coordinate transformations, registrations or modifications applied.']}
    parent=args.output_root.parent;parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='07_tmp_',dir=parent) as temp:
        out=Path(temp)
        files={'01_master_run_level.csv':master,'02_master_method_summary.csv':method_df,'03_master_overall_summary.csv':overall,'04_baseline_vs_loso_within.csv':paired,'05_baseline_vs_loso_within_summary.csv':paired_summary,'06_cross_no_warp_pair_summary.csv':cross_pairs}
        for name,df in files.items():df.to_csv(out/name,index=False)
        (out/'07_validation_report.json').write_text(json.dumps(rep,indent=2)+'\n')
        (out/'README.txt').write_text(__doc__+'\n')
        if args.output_root.exists():
            stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            backup=args.output_root.with_name(args.output_root.name+'_backup_'+stamp)
            if backup.exists():raise FileExistsError(backup)
            args.output_root.rename(backup)
            print('Backed up old Script07 output:',backup)
        shutil.move(str(out),str(args.output_root))
    print('MASTER COMPLETE:',len(master),'rows; methods',method_df.method.nunique())
    print('Baseline paired:',len(paired),'Cross reference-matched pairs:',len(cross_pairs))
    print('Outputs:',args.output_root)

if __name__=='__main__':main()

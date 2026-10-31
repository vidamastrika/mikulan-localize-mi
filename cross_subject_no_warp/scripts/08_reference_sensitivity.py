#!/usr/bin/env python3
"""Script 08 — Reference sensitivity, strictly NO WARP / NO ALIGNMENT.

Purpose
-------
For each of 12 methods and each test run, compare estimates obtained using
six different reference subjects from Script 05. Test EEG and test subject are
held fixed; the reference forward/head model and reference-selected parameters
can both change. Therefore reference variation is descriptive, NOT an isolated
head-model effect and NOT anatomical localization accuracy.

Input files (read-only)
-----------------------
  cross_subject_no_warp/outputs/06_evaluate_localization/03_primary_localization.csv
  cross_subject_no_warp/outputs/06_evaluate_localization/04_sensitivity_localization.csv
  cross_subject_no_warp/outputs/07_compare_strategies/02_master_method_summary.csv
    (optional, historical within-subject baseline context only)

Requirements: Python 3, pandas, numpy. Run from repository root.

Examples
--------
  python cross_subject_no_warp/scripts/08_reference_sensitivity.py --phase audit
  python cross_subject_no_warp/scripts/08_reference_sensitivity.py
  python cross_subject_no_warp/scripts/08_reference_sensitivity.py --overwrite
  python cross_subject_no_warp/scripts/08_reference_sensitivity.py --output-root /tmp/08_test

Outputs
-------
  01_reference_run_level.csv       4,392 x 2 scenarios; original observations
  02_reference_method_summary.csv  per method / scenario / reference
  03_test_run_sensitivity.csv      per method / scenario / test / run, across 6 refs
  04_method_sensitivity_summary.csv across test-run groups (61 per method)
  05_reference_pairing_matrix.csv  method / scenario / test / reference
  06_primary_vs_sensitivity.csv    paired PASS and raw-distance changes
  07_baseline_context.csv          historical within baseline, kept incomparable
  08_validation_report.json        counts, warnings, input hashes and limitations

OVERWRITE POLICY: only this script's output directory. With --overwrite,
existing files created by this script are MOVED into a timestamped backup
folder inside its output directory before saving new outputs. Other directories
and inputs are NEVER modified. No MNE transforms, no intersubject registration,
no coordinate changes, no reconstruction calls.

Statistics: std=sample SD (ddof=1); quartiles use pandas linear interpolation;
missing raw distances for FAIL are excluded from distance statistics but all
scheduled jobs stay in success-rate denominators. Do NOT regard 6 references for
one test run as independent test observations in inferential analyses.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path('cross_subject_no_warp')
EXPECTED_METHODS = 12
EXPECTED_TEST_RUNS = 61
EXPECTED_REFERENCES = 6
EXPECTED_SINGLE_ROWS = EXPECTED_METHODS * EXPECTED_TEST_RUNS * EXPECTED_REFERENCES
KEY = ['method', 'test_subject', 'run', 'reference_subject']
OUTPUTS = [
    '01_reference_run_level.csv','02_reference_method_summary.csv',
    '03_test_run_sensitivity.csv','04_method_sensitivity_summary.csv',
    '05_reference_pairing_matrix.csv','06_primary_vs_sensitivity.csv',
    '07_baseline_context.csv','08_validation_report.json'
]

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()

def stats(values):
    a = pd.to_numeric(pd.Series(values), errors='coerce').dropna()
    a = a[np.isfinite(a)]
    if not len(a):
        return dict(n=0, mean_mm=np.nan, median_mm=np.nan, std_mm=np.nan,
                    q25_mm=np.nan, q75_mm=np.nan, iqr_mm=np.nan,
                    min_mm=np.nan, max_mm=np.nan)
    q25, q75 = a.quantile([.25,.75]).tolist()
    return dict(n=int(len(a)), mean_mm=float(a.mean()), median_mm=float(a.median()),
                std_mm=float(a.std(ddof=1)) if len(a)>1 else np.nan,
                q25_mm=float(q25),q75_mm=float(q75),iqr_mm=float(q75-q25),
                min_mm=float(a.min()),max_mm=float(a.max()))

def single_frame(df, scenario):
    required = set(KEY + ['strategy','condition','status','raw_head_separation_mm'])
    missing=required-set(df.columns)
    if missing: raise ValueError(f'{scenario} input lacks {sorted(missing)}')
    d = df.loc[(df.strategy=='Single-reference') &
               (df.condition=='single_reference_no_warp')].copy()
    if len(d)!=EXPECTED_SINGLE_ROWS:
        raise ValueError(f'{scenario}: expected {EXPECTED_SINGLE_ROWS} Single-reference rows; got {len(d)}')
    if d[KEY].isna().any().any() or d.duplicated(KEY).any():
        raise ValueError(f'{scenario}: null or duplicate keys')
    d['raw_head_separation_mm'] = pd.to_numeric(d['raw_head_separation_mm'],errors='coerce')
    if (d.status.isin(['PASS','FAIL'])==False).any(): raise ValueError(f'{scenario}: unknown status')
    if d.loc[d.status=='PASS','raw_head_separation_mm'].isna().any():
        raise ValueError(f'{scenario}: PASS without raw coordinate diagnostic')
    if (d.loc[d.status=='FAIL','raw_head_separation_mm'].notna()).any():
        raise ValueError(f'{scenario}: FAIL unexpectedly has distance')
    if 'localization_error_valid' in d and d['localization_error_valid'].astype(str).str.lower().eq('true').any():
        raise ValueError('Cross no-warp anatomical error marked valid; refusing')
    if 'cross_anatomical_error_available' in d and d['cross_anatomical_error_available'].astype(str).str.lower().eq('true').any():
        raise ValueError('Cross no-warp anatomical error available; refusing')
    per_test = d.groupby(['method','test_subject','run']).agg(n_refs=('reference_subject','nunique'),rows=('reference_subject','size'))
    if len(per_test)!=EXPECTED_METHODS*EXPECTED_TEST_RUNS or (per_test.n_refs!=6).any() or (per_test.rows!=6).any():
        raise ValueError(f'{scenario}: incomplete reference coverage per test run')
    if (d.reference_subject==d.test_subject).any(): raise ValueError('Self reference in cross design')
    if d.method.nunique()!=EXPECTED_METHODS: raise ValueError('Expected 12 methods')
    if d[['test_subject','run']].drop_duplicates().shape[0]!=EXPECTED_TEST_RUNS:
        raise ValueError('Expected 61 distinct test runs')
    d['scenario']=scenario
    keep=KEY+['scenario','status','raw_head_separation_mm']
    for col in ['configuration_key','sensitivity_override','peak_selection_policy','failure_reason','hemisphere']:
        if col in d: keep.append(col)
    return d[keep].copy()

def summarize_references(df):
    rows=[]
    for (scenario,method,ref),g in df.groupby(['scenario','method','reference_subject'],sort=True):
        rows.append(dict(scenario=scenario,method=method,reference_subject=ref,
                    scheduled=len(g),pass_count=int((g.status=='PASS').sum()),
                    fail_count=int((g.status=='FAIL').sum()),
                    pass_rate=float((g.status=='PASS').mean()),
                    **stats(g.raw_head_separation_mm)))
    return pd.DataFrame(rows)

def sensitivity_per_run(df):
    rows=[]
    for (scenario,method,test,run),g in df.groupby(['scenario','method','test_subject','run'],sort=True):
        ok=g.loc[g.status=='PASS','raw_head_separation_mm']
        ss=stats(ok)
        rows.append(dict(scenario=scenario,method=method,test_subject=test,run=run,
                scheduled_references=6,reference_count=g.reference_subject.nunique(),
                pass_count=len(ok),fail_count=int((g.status=='FAIL').sum()),
                pass_rate=len(ok)/6,range_mm=(ss['max_mm']-ss['min_mm']) if ss['n'] else np.nan,
                cv=(ss['std_mm']/ss['mean_mm']) if ss['n']>1 and ss['mean_mm']!=0 else np.nan,
                **ss))
    return pd.DataFrame(rows)

def summarize_test_runs(perrun):
    rows=[]
    for (scenario,method),g in perrun.groupby(['scenario','method'],sort=True):
        row=dict(scenario=scenario,method=method,test_runs=len(g),scheduled=int(g.scheduled_references.sum()),
                 pass_count=int(g.pass_count.sum()),fail_count=int(g.fail_count.sum()),
                 pass_rate=float(g.pass_count.sum()/g.scheduled_references.sum()),
                 complete_six_reference_runs=int((g.pass_count==6).sum()),
                 incomplete_test_runs=int((g.pass_count<6).sum()),
                 median_across_reference_sd_mm=float(g.std_mm.median()) if g.std_mm.notna().any() else np.nan,
                 mean_across_reference_sd_mm=float(g.std_mm.mean()) if g.std_mm.notna().any() else np.nan,
                 median_across_reference_range_mm=float(g.range_mm.median()) if g.range_mm.notna().any() else np.nan,
                 **{f'pooled_raw_{k}':v for k,v in stats(perrun.loc[(perrun.scenario==scenario)&(perrun.method==method),'median_mm']).items()})
        # 'pooled_raw_...' above summarizes test-run medians, not individual runs.
        rows.append(row)
    return pd.DataFrame(rows)

def pairing_matrix(df):
    rows=[]
    for (scenario,method,test,ref),g in df.groupby(['scenario','method','test_subject','reference_subject'],sort=True):
        rows.append(dict(scenario=scenario,method=method,test_subject=test,reference_subject=ref,
                        scheduled=len(g),pass_count=int((g.status=='PASS').sum()),
                        fail_count=int((g.status=='FAIL').sum()),
                        pass_rate=float((g.status=='PASS').mean()),**stats(g.raw_head_separation_mm)))
    return pd.DataFrame(rows)

def variants(primary, variant):
    cols=KEY+['status','raw_head_separation_mm']
    a=primary[cols].rename(columns={'status':'primary_status','raw_head_separation_mm':'primary_raw_mm'})
    b=variant[cols].rename(columns={'status':'sensitivity_status','raw_head_separation_mm':'sensitivity_raw_mm'})
    v=a.merge(b,on=KEY,how='outer',validate='one_to_one',indicator=True)
    if not v['_merge'].eq('both').all(): raise ValueError('Primary/sensitivity mismatched keys')
    v=v.drop(columns='_merge')
    v['changed_status']=v.primary_status!=v.sensitivity_status
    v['changed_raw_mm']=(~np.isclose(v.primary_raw_mm,v.sensitivity_raw_mm,rtol=1e-11,atol=1e-9,equal_nan=True))
    v['sensitivity_minus_primary_raw_mm']=v.sensitivity_raw_mm-v.primary_raw_mm
    changed=v.loc[v.changed_status]
    if ((changed.primary_status!='FAIL')|(changed.sensitivity_status!='PASS')).any():
        raise ValueError('Unexpected changed status direction')
    if len(changed)!=3 or set(changed.method)!={'irMxNE'}:
        raise ValueError(f'Expected 3 irMxNE overrides, found {len(changed)}')
    if (v.loc[~v.changed_status,'changed_raw_mm']).any():
        raise ValueError('A non-overridden original distance changed')
    return v

def baseline_context(path):
    if not path.exists(): return pd.DataFrame(columns=['scenario','experiment','method','metric','scheduled','pass_count','fail_count','mean_mm','median_mm','std_mm','q25_mm','q75_mm','iqr_mm','interpretation'])
    df=pd.read_csv(path)
    req={'scenario','experiment','method','metric','scheduled','pass_count','fail_count','mean_mm','median_mm'}
    if not req.issubset(df.columns): raise ValueError('Script 07 master method summary has unexpected schema')
    d=df.loc[(df.experiment=='Historical_within')&(df.scenario=='all_pass_selected')].copy()
    if len(d)!=12 or d.method.nunique()!=12: raise ValueError('Historical baseline must have 12 method summaries')
    d['interpretation']='Historical within-subject anatomical localization error; NOT directly comparable to unregistered cross-subject raw head-coordinate separation'
    return d

def prepare_outputs(out,overwrite):
    out.mkdir(parents=True,exist_ok=True)
    existing=[out/n for n in OUTPUTS if (out/n).exists()]
    if existing and not overwrite: raise FileExistsError('Outputs exist; re-run with --overwrite. Files: '+', '.join(str(p) for p in existing))
    if existing:
        folder=out/'backups'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        folder.mkdir(parents=True,exist_ok=False)
        for p in existing: shutil.move(str(p),str(folder/p.name))
        print('Backed up existing Script 08 outputs to:',folder)

def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--phase',choices=['audit','evaluate'],default='evaluate')
    p.add_argument('--primary',type=Path,default=ROOT/'outputs/06_evaluate_localization/03_primary_localization.csv')
    p.add_argument('--sensitivity',type=Path,default=ROOT/'outputs/06_evaluate_localization/04_sensitivity_localization.csv')
    p.add_argument('--baseline-summary',type=Path,default=ROOT/'outputs/07_compare_strategies/02_master_method_summary.csv')
    p.add_argument('--output-root',type=Path,default=ROOT/'outputs/08_reference_sensitivity')
    p.add_argument('--overwrite',action='store_true')
    args=p.parse_args()
    for x in (args.primary,args.sensitivity):
        if not x.is_file():p.error(f'Input missing: {x}')
    primary=single_frame(pd.read_csv(args.primary,low_memory=False),'primary')
    sensitivity=single_frame(pd.read_csv(args.sensitivity,low_memory=False),'sensitivity')
    variant=variants(primary,sensitivity)
    print('Audit PASS: 4392 primary + 4392 sensitivity; 6 references per method/test-run; 3 irMxNE sensitivity overrides')
    print('Primary PASS:',int((primary.status=='PASS').sum()), 'Sensitivity PASS:',int((sensitivity.status=='PASS').sum()))
    if args.phase=='audit': return
    both=pd.concat([primary,sensitivity],ignore_index=True)
    method_summary=summarize_references(both)
    perrun=sensitivity_per_run(both)
    overall=summarize_test_runs(perrun)
    matrix=pairing_matrix(both)
    baseline=baseline_context(args.baseline_summary)
    manifest=dict(script='08_reference_sensitivity.py',no_warp=True,no_alignment=True,
        metric='raw_head_coordinate_separation_mm_DIAGNOSTIC_ONLY',
        inferential_unit='test_subject/run; reference results are repeated measurements',
        caveats=['Reference forward/anatomy AND reference-selected parameter can differ; no causal isolation',
                 'No cross-subject anatomical localization error has been validated',
                 'Historical baseline is within-subject localization error and is not a cross-distance comparator',
                 'Only 7 subjects and 61 test runs, so repeated runs per subject are not independent'],
        input_sha256={str(x):sha256(x) for x in [args.primary,args.sensitivity]+([args.baseline_summary] if args.baseline_summary.exists() else [])},
        totals={'primary':{'rows':len(primary),'pass':int((primary.status=='PASS').sum()),'fail':int((primary.status=='FAIL').sum())},
                'sensitivity':{'rows':len(sensitivity),'pass':int((sensitivity.status=='PASS').sum()),'fail':int((sensitivity.status=='FAIL').sum())}},
        test_run_groups=len(perrun),changed_status=int(variant.changed_status.sum()),
        baseline_context_present=bool(len(baseline)))
    prepare_outputs(args.output_root,args.overwrite)
    outputs={OUTPUTS[0]:both,OUTPUTS[1]:method_summary,OUTPUTS[2]:perrun,OUTPUTS[3]:overall,
             OUTPUTS[4]:matrix,OUTPUTS[5]:variant,OUTPUTS[6]:baseline}
    for n,d in outputs.items():
        d.to_csv(args.output_root/n,index=False)
        print(f'Written {n}: {len(d)} rows')
    (args.output_root/OUTPUTS[7]).write_text(json.dumps(manifest,indent=2)+'\n')
    print('Done. Outputs:',args.output_root)

if __name__=='__main__':main()

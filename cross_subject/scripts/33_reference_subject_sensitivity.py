#!/usr/bin/env python3
"""Script 33: descriptive sensitivity of cross-subject source localization to reference choice.

Uses Script 32 outputs only. No EEG fitting, registration or inverse modeling.
Inference resamples seven test subjects (NOT 366 nominal rows) and is exploratory.

Example:
 python cross_subject/scripts/33_reference_subject_sensitivity.py
 python cross_subject/scripts/33_reference_subject_sensitivity.py --bootstrap 5000
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_INPUT = Path('cross_subject/outputs/32_all_reference_localization_evaluation/cross_subject_localization_errors.csv')
DEFAULT_OUTPUT = Path('cross_subject/outputs/33_reference_subject_sensitivity')
METRICS = {'affine':'cross_affine_error_mm', 'syn':'cross_syn_error_mm'}
KEY = ['test_subject','run']


def describe(values):
    a = np.asarray(values, float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return {'n':0, 'mean_mm':np.nan, 'median_mm':np.nan, 'sd_mm':np.nan, 'min_mm':np.nan, 'max_mm':np.nan}
    return {'n':int(len(a)), 'mean_mm':float(np.mean(a)), 'median_mm':float(np.median(a)),
            'sd_mm':float(np.std(a, ddof=1)) if len(a)>1 else np.nan,
            'min_mm':float(a.min()),'max_mm':float(a.max())}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=DEFAULT_INPUT)
    p.add_argument('--output-root',type=Path,default=DEFAULT_OUTPUT)
    p.add_argument('--bootstrap',type=int,default=3000)
    p.add_argument('--seed',type=int,default=2026)
    args=p.parse_args()
    if args.bootstrap < 1: p.error('--bootstrap must be >= 1')
    df=pd.read_csv(args.input)
    required=set(KEY+['reference_subject','within_subject_error_mm','status','runtime_seconds',*METRICS.values()])
    missing=required-set(df.columns)
    if missing: p.error('Missing CSV columns: '+', '.join(sorted(missing)))
    if df.duplicated(KEY+['reference_subject']).any(): p.error('Duplicate test/run/reference combinations')
    if (df.test_subject==df.reference_subject).any(): p.error('Same-subject reference found')
    for c in [*METRICS.values(),'within_subject_error_mm','runtime_seconds']:
        df[c]=pd.to_numeric(df[c],errors='coerce')
    df['complete']=(df['status']=='PASS_PROVISIONAL') & df[[*METRICS.values(),'within_subject_error_mm']].notna().all(axis=1)
    complete=df[df.complete].copy()
    if complete.empty: p.error('No complete evaluations available')
    refs=sorted(df.test_subject.unique())
    counts=complete.groupby(KEY)['reference_subject'].nunique()
    if len(df)!=366 or len(complete)!=366 or len(refs)!=7 or len(counts)!=61 or not (counts==6).all():
        print('WARNING: not a full 366/61/7/6 complete design; exploratory summaries use available complete records.')
    # Ensure the within baseline is truly constant across the six references for each EEG run.
    variability=complete.groupby(KEY)['within_subject_error_mm'].agg(lambda x:float(x.max()-x.min()))
    if (variability>1e-6).any(): p.error('Subject-specific baseline differs across references for same EEG run')
    args.output_root.mkdir(parents=True,exist_ok=True)
    # One row per test+run; variability due to reference choice, with common baseline once.
    run_rows=[]
    for (test,run),g in complete.groupby(KEY,sort=True):
        row={'test_subject':test,'run':run,'n_references':len(g),
             'within_subject_error_mm':float(g.within_subject_error_mm.iloc[0])}
        for short,col in METRICS.items():
            v=g[col].to_numpy(float)
            row.update({f'{short}_reference_mean_mm':float(np.mean(v)),
                        f'{short}_reference_median_mm':float(np.median(v)),
                        f'{short}_reference_sd_mm':float(np.std(v,ddof=1)) if len(v)>1 else np.nan,
                        f'{short}_reference_range_mm':float(v.max()-v.min()),
                        f'{short}_reference_iqr_mm':float(np.percentile(v,75)-np.percentile(v,25)),
                        f'{short}_mean_minus_within_mm':float(np.mean(v)-row['within_subject_error_mm']),
                        f'{short}_best_reference_ORACLE':str(g.iloc[np.argmin(v)].reference_subject),
                        f'{short}_worst_reference_ORACLE':str(g.iloc[np.argmax(v)].reference_subject),
                        f'{short}_best_error_ORACLE_mm':float(v.min()),
                        f'{short}_worst_error_ORACLE_mm':float(v.max())})
        row['syn_minus_affine_reference_mean_mm']=row['syn_reference_mean_mm']-row['affine_reference_mean_mm']
        run_rows.append(row)
    run_df=pd.DataFrame(run_rows).sort_values(KEY)
    run_df.to_csv(args.output_root/'per_eeg_run_reference_sensitivity.csv',index=False)

    # Subject summaries: each test subject is one independent observational unit for inference.
    subject_rows=[]
    for test,g in run_df.groupby('test_subject',sort=True):
        row={'test_subject':test,'n_runs':len(g),'within_median_mm':float(g.within_subject_error_mm.median())}
        for short in METRICS:
            for k in ['reference_mean_mm','reference_sd_mm','reference_range_mm','mean_minus_within_mm']:
                row[f'{short}_{k}_across_runs_mean']=float(g[f'{short}_{k}'].mean())
        row['syn_minus_affine_across_run_mean_mm']=float(g.syn_minus_affine_reference_mean_mm.mean())
        subject_rows.append(row)
    subject_df=pd.DataFrame(subject_rows)
    subject_df.to_csv(args.output_root/'per_test_subject_sensitivity.csv',index=False)

    # Per directed pair, aggregate across EEG runs. This is descriptive, not a test of which model wins.
    pairs=[]
    for (test,reference),g in complete.groupby(['test_subject','reference_subject'],sort=True):
        row={'test_subject':test,'reference_subject':reference,'n_runs':len(g),
             'runtime_mean_seconds':float(g.runtime_seconds.mean())}
        for short,col in METRICS.items():
            row[f'{short}_mean_error_mm']=float(g[col].mean())
            row[f'{short}_median_error_mm']=float(g[col].median())
        pairs.append(row)
    pair_df=pd.DataFrame(pairs)
    pair_df.to_csv(args.output_root/'per_reference_test_pair_summary.csv',index=False)

    # Reference effects adjusted for unequal test-run totals: first average within each test,
    # then average equally over the six test subjects for a given reference.
    reference_rows=[]
    for ref,g in pair_df.groupby('reference_subject',sort=True):
        row={'reference_subject':ref,'n_test_subjects':len(g),'n_eeg_runs_total':int(g.n_runs.sum())}
        for short in METRICS:
            row[f'{short}_equal_test_weighted_mean_mm']=float(g[f'{short}_mean_error_mm'].mean())
            row[f'{short}_pooled_median_mm']=float(complete.loc[complete.reference_subject==ref,METRICS[short]].median())
        reference_rows.append(row)
    pd.DataFrame(reference_rows).to_csv(args.output_root/'per_reference_subject_summary.csv',index=False)

    # Hierarchical description: run-specific mean across six references then equal-weight test subjects.
    # Cluster bootstrap resamples WHOLE test subjects (n=7); no naive row-level bootstrap.
    subject_stats=subject_df.set_index('test_subject')
    subject_keys=subject_stats.index.to_list()
    measures={
        'affine_minus_within_mm':'affine_mean_minus_within_mm_across_runs_mean',
        'syn_minus_within_mm':'syn_mean_minus_within_mm_across_runs_mean',
        'syn_minus_affine_mm':'syn_minus_affine_across_run_mean_mm',
        'affine_reference_sd_mm':'affine_reference_sd_mm_across_runs_mean',
        'syn_reference_sd_mm':'syn_reference_sd_mm_across_runs_mean',
        'affine_reference_range_mm':'affine_reference_range_mm_across_runs_mean',
        'syn_reference_range_mm':'syn_reference_range_mm_across_runs_mean'}
    rng=np.random.default_rng(args.seed)
    indices=rng.integers(0,len(subject_keys),size=(args.bootstrap,len(subject_keys)))
    inference=[]
    for label,column in measures.items():
        vals=subject_stats[column].to_numpy(float)
        samples=vals[indices].mean(axis=1)
        inference.append({'metric':label,'estimate_mm':float(vals.mean()),
                          'cluster_bootstrap_ci95_low_mm':float(np.percentile(samples,2.5)),
                          'cluster_bootstrap_ci95_high_mm':float(np.percentile(samples,97.5)),
                          'independent_test_subjects':len(vals),'bootstrap_resamples':args.bootstrap})
    pd.DataFrame(inference).to_csv(args.output_root/'subject_cluster_bootstrap.csv',index=False)

    summary={'status':'PROVISIONAL_REGISTRATION_DEPENDENT',
             'input_file':str(args.input),'n_rows':len(df),'n_complete':len(complete),
             'n_distinct_eeg_runs':len(run_df),'n_test_subjects':len(subject_df),
             'n_directed_reference_test_pairs':len(pair_df),
             'overall_complete_case':{c:describe(complete[c]) for c in [*METRICS.values(),'runtime_seconds']},
             'per_run_reference_range':{k:describe(run_df[f'{k}_reference_range_mm']) for k in METRICS},
             'subject_cluster_bootstrap':inference,
             'interpretation':{'unit_of_inference':'test subject (7 clusters)',
                               'reference_best_worst':'Retrospective oracle descriptive only; do not use for predictive inference',
                               'confidence_intervals':'Exploratory percentile cluster bootstrap; seven clusters is small',
                               'limitations':['Anatomical registration and RAS coordinate conventions are provisional',
                                              'Cross-subject errors include transformation effects',
                                              'SESAME hyperparameters selected using these seven subjects',
                                              'Test subjects are not an independently held-out cohort']}}
    (args.output_root/'sensitivity_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print('Script 33 finished:',args.output_root)
    print('Rows:',len(df),'complete:',len(complete),'EEG runs:',len(run_df),'pairs:',len(pair_df))
    for r in inference: print(f"{r['metric']}: {r['estimate_mm']:.2f} mm (cluster CI {r['cluster_bootstrap_ci95_low_mm']:.2f}, {r['cluster_bootstrap_ci95_high_mm']:.2f})")

if __name__=='__main__': main()

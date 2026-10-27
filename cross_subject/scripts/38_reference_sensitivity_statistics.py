#!/usr/bin/env python3
"""Reference sensitivity and exploratory uncertainty, using Script 37's merged CSV.
Run from the project root. No reconstructions or registrations are repeated.
"""
import argparse
import json
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

GROUPS=[('Distributed / minimum-norm',['MNE','dSPM','sLORETA','eLORETA']),('Sparse',['MxNE','irMxNE']),('Dipole-based',['Continuous-ECD','RAP-MUSIC']),('Beamformer',['LCMV']),('Bayesian / hierarchical',['Gamma-MAP','Champagne','SESAME'])]
YEARS={'MNE':1994,'dSPM':2000,'sLORETA':2002,'eLORETA':2007,'MxNE':2012,'irMxNE':2016,'Continuous-ECD':1985,'RAP-MUSIC':1999,'LCMV':1997,'Gamma-MAP':2009,'Champagne':2010,'SESAME':2014}
KEYS=['test_subject','reference_subject','run']
RUN=['method','test_subject','run']
METRICS=['within_subject_error_mm','cross_affine_error_mm','cross_syn_error_mm']
CONTRASTS={'affine_minus_within_mm':(METRICS[1],METRICS[0]),'syn_minus_within_mm':(METRICS[2],METRICS[0]),'syn_minus_affine_mm':(METRICS[2],METRICS[1])}


def interval(values,rng,n_boot):
    """Equal-test-participant estimand; resample participant summaries as clusters."""
    x=np.asarray(values,dtype=float);x=x[np.isfinite(x)]
    if not len(x): return np.nan,np.nan,np.nan,0
    if len(x)<2: return float(x.mean()),np.nan,np.nan,len(x)
    samples=x[rng.integers(0,len(x),size=(n_boot,len(x)))].mean(axis=1)
    lo,hi=np.quantile(samples,[.025,.975])
    return float(x.mean()),float(lo),float(hi),len(x)


def validate(d):
    required=set(['method','status']+KEYS+METRICS)
    if required-set(d): raise ValueError(f'Missing columns: {sorted(required-set(d))}')
    if d[['method','status']+KEYS].isna().any().any(): raise ValueError('Missing case identifiers or status')
    if set(d.method)!=set(YEARS): raise ValueError('Expected all 12 methods from Script 37')
    if d.duplicated(['method']+KEYS).any(): raise ValueError('Duplicate method/case rows')
    schedule=None
    for method,g in d.groupby('method'):
        keys=set(map(tuple,g[KEYS].values));subjects=set(g.test_subject)
        counts=g.groupby(['test_subject','run']).reference_subject.nunique()
        if (len(keys)!=366 or len(subjects)!=7 or set(g.reference_subject)!=subjects
                or len(counts)!=61 or not counts.eq(6).all()
                or g.test_subject.eq(g.reference_subject).any()):
            raise ValueError(f'{method}: expected 61 runs x 6 references among 7 subjects')
        if schedule is not None and keys!=schedule: raise ValueError('Case schedules differ')
        schedule=keys
    for c in METRICS:d[c]=pd.to_numeric(d[c],errors='raise')
    for key,g in d.groupby(RUN):
        x=g[METRICS[0]].dropna()
        if len(x) and x.max()-x.min()>1e-6: raise ValueError(f'Inconsistent baseline: {key}')
    d['analysis_complete']=(d.status.isin(['PASS','PASS_PROVISIONAL'])
        & np.isfinite(d[METRICS]).all(axis=1) & d[METRICS].ge(0).all(axis=1))
    return d


def summarize_runs(frame):
    rows=[]
    for key,g in frame.groupby(RUN):
        row=dict(zip(RUN,key));row['n_references']=len(g)
        row.update(g[METRICS].mean().to_dict())
        for contrast,(a,b) in CONTRASTS.items():row[contrast]=float((g[a]-g[b]).mean())
        for label,column in [('affine',METRICS[1]),('syn',METRICS[2])]:
            row[f'{label}_reference_sd_mm']=float(g[column].std(ddof=1)) if len(g)>1 else np.nan
            row[f'{label}_reference_range_mm']=float(g[column].max()-g[column].min()) if len(g)>1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def forest(summary,methods,path,scope):
    fig,axes=plt.subplots(1,3,figsize=(15,8),sharey=True)
    for ax,metric in zip(axes,CONTRASTS):
        sub=summary[summary.metric.eq(metric)].set_index('method').reindex(methods)
        for y,(_,r) in enumerate(sub.iterrows()):
            if np.isfinite(r.estimate_mm):ax.plot(r.estimate_mm,y,'o',color='#376C99',ms=5)
            if np.isfinite(r.ci_low_mm):ax.plot([r.ci_low_mm,r.ci_high_mm],[y,y],color='#376C99',lw=1.6)
        ax.axvline(0,color='#888888',ls='--',lw=1)
        ax.set_title(metric.replace('_mm','').replace('_',' '),fontsize=11)
        ax.set_xlabel('Mean paired difference (mm)');ax.grid(axis='x',alpha=.2)
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_yticks(range(len(methods)),[f'{m} ({YEARS[m]})' for m in methods]);axes[0].invert_yaxis()
    fig.suptitle(f'{scope.capitalize()} cases: paired localization differences',fontsize=15)
    fig.text(.5,.025,'Equal test-participant weighting; 95% percentile bootstrap intervals. Fixed reference pool. Exploratory; no multiplicity adjustment.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.055,1,.95));fig.savefig(path,dpi=240);plt.close(fig)


def heatmaps(frame,methods,root):
    subjects=sorted(frame.test_subject.unique())
    for label,col in [('affine',METRICS[1]),('syn',METRICS[2])]:
        # Mean per test/reference pair across available runs; counts accompany values.
        values=frame.groupby(['method','test_subject','reference_subject'])[col].mean().unstack('reference_subject')
        counts=frame.groupby(['method','test_subject','reference_subject']).size().unstack('reference_subject')
        values.to_csv(root/f'09_{label}_reference_pair_means.csv')
        counts.to_csv(root/f'09_{label}_reference_pair_counts.csv')
        fig,axes=plt.subplots(4,3,figsize=(13,16),layout='constrained')
        vmax=float(values.max().max());im=None
        for ax,m in zip(axes.flat,methods):
            if m in values.index.get_level_values('method'):
                mat=values.loc[m].reindex(index=subjects,columns=subjects)
            else:mat=pd.DataFrame(np.nan,index=subjects,columns=subjects)
            im=ax.imshow(mat,vmin=0,vmax=max(1,vmax),cmap='viridis')
            ax.set_title(f'{m} ({YEARS[m]})',fontsize=10)
            ax.set_xticks(range(7),subjects,rotation=45,ha='right',fontsize=7)
            ax.set_yticks(range(7),subjects,fontsize=7)
            ax.set_xlabel('Reference subject',fontsize=8);ax.set_ylabel('Test subject',fontsize=8)
        fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.6,label='Mean localization error (mm)')
        fig.suptitle(f'{label.capitalize()}: available-case error by reference and test subject\nBlank cells: self-pairs or no successful cases',fontsize=13)
        fig.savefig(root/f'10_{label}_reference_heatmaps.png',dpi=220);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=Path('cross_subject/outputs/37_cross_subject_comparison/01_all_methods_evaluation.csv'))
    p.add_argument('--output-root',type=Path,default=Path('cross_subject/outputs/38_reference_sensitivity_statistics'))
    p.add_argument('--bootstrap',type=int,default=10000)
    p.add_argument('--seed',type=int,default=38)
    p.add_argument('--overwrite',action='store_true')
    a=p.parse_args()
    if a.bootstrap<1000:raise ValueError('Use at least 1000 bootstrap replicates')
    d=validate(pd.read_csv(a.input));root=a.output_root
    if root.exists() and any(root.iterdir()) and not a.overwrite:raise FileExistsError('Output folder not empty; use --overwrite')
    root.mkdir(parents=True,exist_ok=True);rng=np.random.default_rng(a.seed)
    methods=[m for _,members in GROUPS for m in members]
    good=d[d.analysis_complete].copy()
    sets=[set(map(tuple,good.loc[good.method.eq(m),KEYS].values)) for m in methods]
    common=set.intersection(*sets)
    matched=good[[tuple(row) in common for row in good[KEYS].values]].copy()
    coverage=d.groupby('method').agg(attempted=('status','size'),complete=('analysis_complete','sum'))
    coverage['incomplete']=coverage.attempted-coverage.complete
    coverage['incomplete_percent']=100*coverage.incomplete/coverage.attempted
    coverage.to_csv(root/'01_completeness.csv')
    d.groupby(['method','status']).size().rename('count').to_csv(root/'01_original_status_counts.csv')
    d.loc[~d.analysis_complete].to_csv(root/'02_incomplete_cases.csv',index=False)
    pd.DataFrame(sorted(common),columns=KEYS).to_csv(root/'03_common_successful_cases.csv',index=False)
    d.groupby(['method','test_subject','reference_subject']).analysis_complete.agg(['size','sum']).to_csv(root/'03_reference_pair_coverage.csv')
    for scope,frame in [('available',good),('matched',matched)]:
        if frame.empty:
            print(f'No {scope} cases; skipped.');continue
        runs=summarize_runs(frame);runs.to_csv(root/f'04_{scope}_run_level.csv',index=False)
        metric_cols=METRICS+list(CONTRASTS)+['affine_reference_sd_mm','syn_reference_sd_mm','affine_reference_range_mm','syn_reference_range_mm']
        participant=runs.groupby(['method','test_subject'])[metric_cols].mean()
        participant.to_csv(root/f'05_{scope}_participant_level.csv')
        summary=[]
        for method,g in participant.groupby(level='method'):
            for metric in metric_cols:
                estimate,low,high,n=interval(g[metric],rng,a.bootstrap)
                summary.append(dict(method=method,metric=metric,estimate_mm=estimate,ci_low_mm=low,ci_high_mm=high,n_test_subjects=n))
        summary=pd.DataFrame(summary);summary.to_csv(root/f'06_{scope}_cluster_intervals.csv',index=False)
        forest(summary,methods,root/f'07_{scope}_paired_differences.png',scope)
        if scope=='matched':
            pairs=[]
            for metric in METRICS:
                wide=participant[metric].unstack('method')
                for first,second in combinations(methods,2):
                    if first not in wide or second not in wide:continue
                    diff=(wide[first]-wide[second]).dropna()
                    estimate,lo,hi,n=interval(diff,rng,a.bootstrap)
                    pairs.append(dict(method_a=first,method_b=second,metric=metric,a_minus_b_mm=estimate,ci_low_mm=lo,ci_high_mm=hi,n_test_subjects=n))
            pd.DataFrame(pairs).to_csv(root/'08_matched_method_pair_differences.csv',index=False)
    if not good.empty:heatmaps(good,methods,root)
    (root/'analysis_settings.json').write_text(json.dumps(dict(input=str(a.input),bootstrap=a.bootstrap,seed=a.seed,common_cases=len(common),estimand='equal test-participant mean; references fixed'),indent=2))
    (root/'README.txt').write_text('''Script 38: exploratory reference sensitivity and statistical uncertainty.
Input: Script 37 merged 12-method evaluation CSV. No solvers are rerun.
Available: all successful finite, nonnegative evaluations per method.
Matched: identical successful test/run/reference cases shared by all 12 methods.
Failures and PASS_PROVISIONAL statuses are preserved; analysis does not validate provisional results.
Each run contributes its mean across successful references. Each test participant contributes its mean across runs. Participants receive equal weight.
Paired differences are formed on the same cases. Baseline is not counted six times as independent data.
Reference SD uses ddof=1; reference range is max-minus-min. Both require at least two successful references. See n_references in run tables: reduced coverage can change sensitivity estimates.
Confidence intervals: 95% percentile bootstrap resampling test-participant summaries, 10,000 replicates by default, fixed seed. The small pool of seven test participants limits precision.
The reference pool is held fixed. Intervals do not model additional dependence from reusing the same reference heads across test participants, or generalize uncertainty to new reference heads.
The estimand is participant-weighted, so it can differ from Script 37 run-weighted descriptive summaries and prior Script 33 summaries.
Method-pair intervals are exploratory and not adjusted for multiple comparisons; no p-values or significance declarations are produced. An interval containing zero does not establish equivalence.
Heatmaps use available cases; corresponding pair means/counts and coverage CSVs expose missingness. Blank self-pairs are not zero error.
Original global fixed parameters may include test-subject data in selection; this is not LOSO parameter validation. Scripts 39/40 address that question.
''')
    print(coverage.to_string());print(f'Common successful cases: {len(common)}/366');print('Saved:',root)

if __name__=='__main__':main()

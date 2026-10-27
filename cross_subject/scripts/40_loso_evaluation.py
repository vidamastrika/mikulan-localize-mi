#!/usr/bin/env python3
"""Script 40: CSV-only evaluation of Script 39 selections, Localize-MI 12 methods.

Run from repository root:
  python cross_subject/scripts/40_loso_evaluation.py

No EEG inverse solutions are computed. Distinguish all-PASS selection on
reference subjects from failures on held-out test subjects. Global selection
is in-sample, not independent validation.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

METHODS = ['MNE','dSPM','sLORETA','eLORETA','Continuous-ECD','RAP-MUSIC',
           'LCMV','MxNE','irMxNE','Gamma-MAP','Champagne','SESAME']
SUBJECTS = [f'sub-{i:02d}' for i in range(1,8)]
STRATEGIES = ['script26_fixed_in_sample','single_reference','loso']


def get_args():
    root = Path(__file__).resolve().parents[2]
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection-root',type=Path,default=root/'cross_subject/outputs/39_loso_parameter_selection')
    p.add_argument('--output-root',type=Path,default=root/'cross_subject/outputs/40_loso_evaluation')
    p.add_argument('--script26-baseline',type=Path,default=root/'outputs/26_twelve_methods_comparison/complete_all_pass/tables/13_selected_configuration_results_long.csv')
    p.add_argument('--script26-definitions',type=Path,default=root/'outputs/26_twelve_methods_comparison/complete_all_pass/tables/01_selected_configuration_definitions.csv')
    p.add_argument('--chunk-size',type=int,default=50000)
    p.add_argument('--dpi',type=int,default=180)
    p.add_argument('--no-figures',action='store_true')
    return p.parse_args()


def get_selected(folder):
    parts=[]
    for filename,expected in [('05_single_reference_selected.csv','single_reference'),
                              ('07_loso_selected.csv','loso')]:
        f=folder/filename
        if not f.is_file(): raise FileNotFoundError(f)
        d=pd.read_csv(f,low_memory=False,dtype={'configuration_key':str})
        required={'method','configuration_key','configuration_parameters','reference_subjects'}
        if expected=='loso': required.add('test_subject')
        if expected=='single_reference': required.add('reference_subject')
        if missing:=required-set(d.columns): raise ValueError(f'{f}: missing {missing}')
        if not d['strategy'].eq(expected).all(): raise ValueError(f'{f}: unexpected strategy')
        for _,r in d.iterrows():
            method=str(r.method); config=str(r.configuration_key)
            if method not in METHODS: raise ValueError(f'Unexpected method: {method}')
            testlist=([str(r.test_subject)] if expected=='loso' else
                      [s for s in SUBJECTS if s!=str(r.reference_subject)] if expected=='single_reference'
                      else SUBJECTS)
            if expected=='loso' and (str(r.test_subject) in str(r.reference_subjects).split(';')):
                raise ValueError(f'LOSO leakage: {method}/{r.test_subject}')
            for test in testlist:
                if test not in SUBJECTS: raise ValueError(f'Unexpected test subject {test}')
                parts.append({'method':method,'strategy': expected,
                              'test_subject':test,'reference_subject':str(r.reference_subject) if expected=='single_reference' else '',
                              'reference_subjects':str(r.reference_subjects),
                              'configuration_key':config,'configuration_parameters':str(r.configuration_parameters)})
    s=pd.DataFrame(parts)
    if s.duplicated(['method','strategy','test_subject','reference_subject']).any():
        raise ValueError('Duplicate selections for method/strategy/test/reference')
    for method,g in s.groupby('method'):
        counts=g.groupby('strategy').size().to_dict()
        if counts!={'loso':7,'single_reference':42}:
            raise ValueError(f'{method} incorrect selection counts {counts}')
    return s


def load_script26_baseline(path, definitions_path):
    """Preserve Script 26's original complete_all_pass selected run results verbatim."""
    if not path.is_file():
        raise FileNotFoundError(f'Script 26 baseline unavailable: {path}')
    if not definitions_path.is_file():
        raise FileNotFoundError(f'Script 26 selection definitions unavailable: {definitions_path}')
    d = pd.read_csv(path, low_memory=False)
    defs = pd.read_csv(definitions_path, low_memory=False)
    needed = {'method','subject','run','status','localization_error_mm','configuration_key'}
    if missing := needed-set(d.columns):
        raise ValueError(f'Script 26 selected results missing {sorted(missing)}')
    if missing := {'method','configuration_key','configuration_parameters'}-set(defs.columns):
        raise ValueError(f'Script 26 configuration definitions missing {sorted(missing)}')
    if d.duplicated(['method','subject','run']).any():
        raise ValueError('Script 26 baseline has duplicate method/subject/run')
    if defs.duplicated('method').any():
        raise ValueError('Script 26 definitions have duplicate method')
    if set(d.method)!=set(METHODS) or set(defs.method)!=set(METHODS):
        raise ValueError('Script 26 baseline/definitions do not contain exactly 12 methods')
    counts=d.groupby('method').size()
    if not counts.eq(61).all():
        raise ValueError(f'Script 26 baseline must have 61 rows/method: {counts.to_dict()}')
    if not d.status.astype(str).str.upper().eq('PASS').all():
        raise ValueError('Complete-all-PASS Script 26 baseline has FAIL rows')
    runset=set(map(tuple,d[d.method==METHODS[0]][['subject','run']].to_numpy()))
    for method,g in d.groupby('method'):
        if set(map(tuple,g[['subject','run']].to_numpy()))!=runset:
            raise ValueError(f'Script 26 method {method} has mismatched runs')
        defrow=defs.loc[defs.method.eq(method)].iloc[0]
        if not g.configuration_key.astype(str).eq(str(defrow.configuration_key)).all():
            raise ValueError(f'Script 26 selected config mismatches definition for {method}')
    d['localization_error_mm']=pd.to_numeric(d.localization_error_mm,errors='coerce')
    if not np.isfinite(d.localization_error_mm).all() or d.localization_error_mm.lt(0).any():
        raise ValueError('Invalid numeric Script 26 baseline errors')
    d=d.rename(columns={'subject':'test_subject'})
    d['strategy']='script26_fixed_in_sample'
    d['reference_subject']=''
    d['reference_subjects']=';'.join(SUBJECTS)
    d=d.merge(defs[['method','configuration_key','configuration_parameters']],
              on=['method','configuration_key'],validate='many_to_one')
    cols=['method','strategy','test_subject','run','reference_subject','reference_subjects',
          'configuration_key','configuration_parameters','status','localization_error_mm']
    return d[cols].copy(),defs


def scan_grid(path, chosen_keys, chunk_size):
    if not path.is_file():raise FileNotFoundError(path)
    use=['method','_canonical_method','subject','run','_run_key','_config_key','status','_error',
         'localization_distance_mm','_valid','_missing_parameters','_params_json']
    cols=pd.read_csv(path,nrows=0).columns.tolist()
    for c in ['_canonical_method','subject','run','_config_key','status']:
        if c not in cols:raise ValueError(f'Grid missing {c}')
    selected=[c for c in use if c in cols]
    seen=[]
    for chunk in pd.read_csv(path,usecols=selected,chunksize=chunk_size,low_memory=False,
                             dtype={'_config_key':str,'subject':str,'run':str,'_canonical_method':str}):
        sub=chunk.loc[chunk['_config_key'].isin(chosen_keys)].copy()
        if not sub.empty: seen.append(sub)
    if not seen:raise ValueError('No selected configurations found in backup CSV')
    grid=pd.concat(seen,ignore_index=True)
    grid['status']=grid['status'].astype(str).str.upper().str.strip()
    grid['_error']=pd.to_numeric(grid.get('_error',pd.Series(np.nan,index=grid.index)),errors='coerce')
    if 'localization_distance_mm' in grid.columns:
        fallback=pd.to_numeric(grid['localization_distance_mm'],errors='coerce')
        grid['_error']=grid['_error'].fillna(fallback)
    valid=grid['status'].eq('PASS')
    if ((~np.isfinite(grid.loc[valid,'_error'])) | (grid.loc[valid,'_error']<0)).any():
        raise ValueError('PASS row with missing/nonfinite/negative error')
    if not set(grid['status'].unique())<= {'PASS','FAIL'}:
        raise ValueError('Unexpected status in selected rows')
    key=['_canonical_method','_config_key','subject','run']
    if grid.duplicated(key).any():raise ValueError('Duplicate canonical results for method/config/subject/run')
    return grid


def expand_evaluations(selection,grid):
    matches=[]
    for (method,config,test), sel in selection.groupby(['method','configuration_key','test_subject'],sort=False):
        g=grid.loc[(grid['_canonical_method']==method)&(grid['_config_key']==config)&(grid['subject']==test)].copy()
        if g.empty:raise ValueError(f'Missing test results: {method} {config} {test}')
        for _,r in sel.iterrows():
            for _,v in g.iterrows():
                matches.append({'method':method,'strategy':r.strategy,'test_subject':test,
                                'run':str(v['run']),'reference_subject':r.reference_subject,
                                'reference_subjects':r.reference_subjects,'configuration_key':config,
                                'configuration_parameters':r.configuration_parameters,
                                'status':str(v.status),
                                'localization_error_mm':float(v['_error']) if v.status=='PASS' else np.nan})
    out=pd.DataFrame(matches)
    runsets=out.groupby(['method','test_subject','strategy','reference_subject'],dropna=False)['run'].nunique()
    if (runsets<=0).any():raise ValueError('Empty evaluation group')
    per_subject=out.groupby(['method','test_subject','strategy','reference_subject'],dropna=False).agg(
        n_runs=('run','nunique'),n_pass=('status',lambda x:int(x.eq('PASS').sum())),
        n_fail=('status',lambda x:int(x.eq('FAIL').sum())),
        mean_error_mm=('localization_error_mm','mean'),median_error_mm=('localization_error_mm','median'),
        sd_error_mm=('localization_error_mm','std')).reset_index()
    per_subject['pass_rate']=per_subject.n_pass/per_subject.n_runs
    return out,per_subject


def summarize(df):
    rows=[]
    for (method,strategy),g in df.groupby(['method','strategy']):
        vals=g['localization_error_mm'].dropna()
        subject_medians=g.groupby('test_subject')['localization_error_mm'].median()
        rows.append(dict(method=method,strategy=strategy,n_total=len(g),n_pass=int(g.status.eq('PASS').sum()),
                         n_fail=int(g.status.eq('FAIL').sum()),pass_rate=float(g.status.eq('PASS').mean()),
                         mean_error_mm=vals.mean(),sd_error_mm=vals.std(),median_error_mm=vals.median(),
                         q25_error_mm=vals.quantile(.25),q75_error_mm=vals.quantile(.75),
                         median_subject_medians_mm=subject_medians.median(),
                         mean_subject_medians_mm=subject_medians.mean()))
    return pd.DataFrame(rows)


def paired(df):
    # Pair LOSO versus global on identical held-out subject/run, including failures.
    a=df.loc[df.strategy=='loso',['method','test_subject','run','status','localization_error_mm']].rename(
        columns={'status':'loso_status','localization_error_mm':'loso_error_mm'})
    b=df.loc[df.strategy=='script26_fixed_in_sample',['method','test_subject','run','status','localization_error_mm']].rename(
        columns={'status':'global_status','localization_error_mm':'global_error_mm'})
    if a.duplicated(['method','test_subject','run']).any() or b.duplicated(['method','test_subject','run']).any():
        raise ValueError('Paired baseline contains duplicate run')
    p=a.merge(b,on=['method','test_subject','run'],how='outer',validate='one_to_one',indicator=True)
    if not p['_merge'].eq('both').all():raise ValueError('Unmatched paired test runs')
    p=p.drop(columns='_merge')
    p['both_pass']=p.loso_status.eq('PASS') & p.global_status.eq('PASS')
    p['delta_loso_minus_global_mm']=np.where(p.both_pass,p.loso_error_mm-p.global_error_mm,np.nan)
    rows=[]
    for method,g in p.groupby('method'):
        matched=g.delta_loso_minus_global_mm.dropna()
        bysub=g.groupby('test_subject').delta_loso_minus_global_mm.median().dropna()
        rows.append({'method':method,'total_pairs':len(g),'both_pass_pairs':len(matched),
                     'loso_only_fail':int((g.loso_status.eq('FAIL') & g.global_status.eq('PASS')).sum()),
                     'global_only_fail':int((g.global_status.eq('FAIL') & g.loso_status.eq('PASS')).sum()),
                     'both_fail':int((g.global_status.eq('FAIL') & g.loso_status.eq('FAIL')).sum()),
                     'median_paired_delta_mm':matched.median(),
                     'mean_paired_delta_mm':matched.mean(),
                     'median_of_subject_median_deltas_mm':bysub.median(),
                     'n_subjects_with_matched_pairs':len(bysub)})
    return p,pd.DataFrame(rows)


def pair_summary(df):
    x=df[df.strategy=='single_reference'].copy()
    z=x.groupby(['method','reference_subject','test_subject'],dropna=False).agg(
        n_runs=('run','nunique'),n_pass=('status',lambda v:int(v.eq('PASS').sum())),
        n_fail=('status',lambda v:int(v.eq('FAIL').sum())),
        median_error_mm=('localization_error_mm','median'),
        mean_error_mm=('localization_error_mm','mean')).reset_index()
    z['pass_rate']=z.n_pass/z.n_runs
    return z


def figures(summary,subject_summary,pairs,delta,out,dpi):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=out/'figures';folder.mkdir(exist_ok=True)
    # Absolute within-strategy results, not naive unpaired inference.
    data=summary.pivot(index='method',columns='strategy',values='median_subject_medians_mm').reindex(METHODS)
    fig,ax=plt.subplots(figsize=(13,6))
    x=np.arange(len(data));width=.26
    for j,s in enumerate(STRATEGIES):
        ax.bar(x+(j-1)*width,data[s],width,label=s.replace('_',' '))
    ax.set_xticks(x,[display_method(m) for m in data.index],rotation=40,ha='right');ax.set_ylabel('Median of subject medians (mm)')
    ax.set_title('Source localization error by parameter selection strategy');ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(folder/'01_method_strategy_medians.png',dpi=dpi);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5))
    dat=delta.set_index('method').reindex(METHODS)
    ax.bar(dat.index,dat['median_of_subject_median_deltas_mm'])
    ax.axhline(0,color='black',lw=.8)
    ax.set_ylabel('LOSO − global fixed, median of subject deltas (mm)')
    ax.set_title('Paired LOSO penalty (only pairs where both PASS)')
    ax.set_xticks(range(len(dat.index)),[display_method(m) for m in dat.index],rotation=40,ha='right')
    fig.tight_layout();fig.savefig(folder/'02_loso_paired_deltas.png',dpi=dpi);plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,5))
    data=summary.pivot(index='method',columns='strategy',values='pass_rate').reindex(METHODS)
    for j,s in enumerate(STRATEGIES):ax.bar(x+(j-1)*width,data[s],width,label=s.replace('_',' '))
    ax.set_xticks(x,[display_method(m) for m in data.index],rotation=40,ha='right');ax.set_ylim(0,1.04);ax.set_ylabel('PASS fraction')
    ax.set_title('Evaluation success rates');ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(folder/'03_success_rates.png',dpi=dpi);plt.close(fig)
    for method in METHODS:
        sub=pairs.loc[pairs.method==method]
        mat=sub.pivot(index='reference_subject',columns='test_subject',values='median_error_mm').reindex(index=SUBJECTS,columns=SUBJECTS)
        fig,ax=plt.subplots(figsize=(7,5));im=ax.imshow(mat.to_numpy(float),aspect='auto')
        ax.set_xticks(range(7),SUBJECTS,rotation=45,ha='right');ax.set_yticks(range(7),SUBJECTS)
        ax.set_xlabel('Test subject');ax.set_ylabel('Reference subject')
        ax.set_title(f'{display_method(method)}: single-reference transfer (median error mm)')
        fig.colorbar(im,ax=ax,label='Median error (mm)')
        fig.tight_layout();safe=method.lower().replace('-','_')
        fig.savefig(folder/f'04_heatmap_{safe}.png',dpi=dpi);plt.close(fig)



# Visual comparisons use individual PASS run-level localization errors, never
# synthetic distributions reconstructed from group summary statistics.
METHOD_FAMILIES = [
    ('Minimum norm', ['MNE', 'dSPM', 'sLORETA', 'eLORETA']),
    ('Dipole fitting', ['Continuous-ECD']),
    ('Subspace', ['RAP-MUSIC']),
    ('Beamformer', ['LCMV']),
    ('Sparse', ['MxNE', 'irMxNE']),
    ('Bayesian / probabilistic', ['Gamma-MAP', 'Champagne', 'SESAME']),
]
# Publication labels matched to Script 26 (reference years, not first-invention claims).
METHOD_YEARS = {
    'MNE':1994, 'dSPM':2000, 'sLORETA':2002, 'eLORETA':2007,
    'Continuous-ECD':1985, 'RAP-MUSIC':1999, 'LCMV':1997,
    'MxNE':2012, 'irMxNE':2016, 'Gamma-MAP':2009,
    'Champagne':2010, 'SESAME':2014,
}
def display_method(method):
    return f'{method} ({METHOD_YEARS[method]})'

STRATEGY_LABELS = {
    'script26_fixed_in_sample': 'Script 26 fixed baseline (in-sample)',
    'loso': 'LOSO (held-out test)',
    'single_reference': 'Single-reference transfer',
}


def strategy_boxplots(evaluations, out, dpi):
    """Run-level boxplots (including all valid test runs for each transfer)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder = out / 'figures' / 'boxplots'
    folder.mkdir(parents=True, exist_ok=True)
    x = evaluations.copy()
    needed = {'method','strategy','status','localization_error_mm'}
    missing = needed - set(x.columns)
    if missing:
        raise ValueError(f'Boxplot input missing columns: {sorted(missing)}')
    x['localization_error_mm'] = pd.to_numeric(x['localization_error_mm'], errors='coerce')
    x = x.loc[(x['status'].astype(str).str.upper() == 'PASS') &
              np.isfinite(x['localization_error_mm'])].copy()
    if x.empty:
        raise ValueError('No valid PASS localization errors for boxplots.')
    strategies = ['script26_fixed_in_sample', 'loso', 'single_reference']
    ordered_methods = [m for _, names in METHOD_FAMILIES for m in names]
    if set(ordered_methods) != set(METHODS):
        raise ValueError('Method families do not cover METHODS exactly.')

    def observations(method, strategy):
        return x.loc[(x.method == method) & (x.strategy == strategy),
                     'localization_error_mm'].to_numpy(float)

    def draw(ax, methods, title, tick_size=9):
        # Group boxes by method, with 3 strategy boxes around each method index.
        offsets = (-0.27, 0.0, 0.27)
        for j, strategy in enumerate(strategies):
            series = [observations(m, strategy) for m in methods]
            nonempty = [(i, values) for i, values in enumerate(series) if values.size]
            if not nonempty:
                continue
            locs = [i + offsets[j] for i, _ in nonempty]
            bp = ax.boxplot([v for _, v in nonempty], positions=locs,
                            widths=0.22, patch_artist=True, showfliers=False,
                            manage_ticks=False, whis=(5, 95))
            # Matplotlib default color cycle: stable color by strategy.
            color = plt.rcParams['axes.prop_cycle'].by_key()['color'][j]
            for box in bp['boxes']:
                box.set_facecolor(color)
                box.set_alpha(0.55)
            for median in bp['medians']:
                median.set_color('black')
                median.set_linewidth(1.3)
            for element in bp['whiskers'] + bp['caps']:
                element.set_color(color)
        ax.set_xticks(range(len(methods)), [display_method(m) for m in methods], rotation=35, ha='right', fontsize=tick_size)
        ax.set_xlim(-0.6, len(methods)-0.4)
        ax.set_ylabel('Localization error (mm)')
        ax.set_title(title)
        ax.grid(axis='y', alpha=0.2)

    def legend(fig, ax):
        from matplotlib.patches import Patch
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        handles = [Patch(facecolor=colors[j], alpha=0.55,
                         label=STRATEGY_LABELS[v]) for j, v in enumerate(strategies)]
        ax.legend(handles=handles, frameon=False, fontsize=9, loc='upper right')

    # Single overview ordered by method family, with thin family boundaries.
    fig, ax = plt.subplots(figsize=(17, 7))
    draw(ax, ordered_methods, 'Localization error by method family: Script 26 fixed vs LOSO vs single-reference')
    mark = 0
    for group, names in METHOD_FAMILIES[:-1]:
        mark += len(names)
        ax.axvline(mark-0.5, color='0.6', linewidth=0.8, linestyle=':')
    legend(fig, ax)
    fig.tight_layout()
    fig.savefig(folder/'01_all_methods_by_family_boxplot.png', dpi=dpi)
    plt.close(fig)

    # One panel per family (six panels in one document-sized figure).
    fig, axs = plt.subplots(3, 2, figsize=(16, 14), sharey=False)
    for ax, (group, names) in zip(axs.flat, METHOD_FAMILIES):
        draw(ax, names, group)
    from matplotlib.patches import Patch
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig.legend(handles=[Patch(facecolor=colors[j], alpha=.55,
                               label=STRATEGY_LABELS[st]) for j, st in enumerate(strategies)],
               loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(.5, .995))
    fig.suptitle('Localization error by method family', y=1.02)
    fig.tight_layout(rect=[0, 0, 1, .96])
    fig.savefig(folder/'02_method_family_panels_boxplot.png', dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    # One standalone boxplot for each method.
    for method in ordered_methods:
        fig, ax = plt.subplots(figsize=(7.5, 5.3))
        vals = [observations(method, st) for st in strategies]
        nonempty = [(j, v) for j, v in enumerate(vals) if len(v)]
        if not nonempty:
            plt.close(fig)
            continue
        bp = ax.boxplot([v for j, v in nonempty], positions=[j for j, _ in nonempty],
                        widths=.55, patch_artist=True, showfliers=False,
                        whis=(5, 95), manage_ticks=False)
        for (j, v), b in zip(nonempty, bp['boxes']):
            b.set_facecolor(colors[j]); b.set_alpha(.55)
        ax.set_xticks(range(3), [STRATEGY_LABELS[v] for v in strategies],
                      rotation=15, ha='right')
        ax.set_xlim(-.6, 2.6)
        ax.set_title(f'{display_method(method)}: parameter-transfer evaluation')
        ax.set_ylabel('Localization error (mm)')
        ax.grid(axis='y', alpha=.2)
        counts = ' | '.join(f'{len(v)} PASS' for v in vals)
        ax.text(.5, -.27, counts, transform=ax.transAxes, ha='center', fontsize=8)
        fig.tight_layout()
        safe = method.lower().replace('-', '_').replace(' ', '_')
        fig.savefig(folder/f'03_method_{safe}_boxplot.png', dpi=dpi, bbox_inches='tight')
        plt.close(fig)

    manifest = (x.groupby(['method','strategy']).agg(
        pass_observations=('localization_error_mm','size'),
        median_error_mm=('localization_error_mm','median')).reset_index())
    manifest.to_csv(out/'09_boxplot_observation_counts.csv', index=False)
    print(f'Boxplots saved to {folder}')

def main():
    a=get_args();src=a.selection_root;out=a.output_root
    out.mkdir(parents=True,exist_ok=True)
    selected=get_selected(src)
    selected.to_csv(out/'01_expanded_selection_map.csv',index=False)
    selectedkeys=set(selected.configuration_key)
    grid=scan_grid(src/'11_all_12_methods_canonical_grid_backup.csv',selectedkeys,a.chunk_size)
    result,per_subject=expand_evaluations(selected,grid)
    baseline,baseline_defs=load_script26_baseline(a.script26_baseline,a.script26_definitions)
    if set(map(tuple,baseline[['method','test_subject','run']].to_numpy())) != set(map(tuple,result.loc[result.strategy.eq('loso'),['method','test_subject','run']].to_numpy())):
        raise ValueError('Script 26 baseline and LOSO test run identities do not match')
    result=pd.concat([result,baseline],ignore_index=True,sort=False)
    bs=baseline.groupby(['method','test_subject','strategy','reference_subject'],dropna=False).agg(
        n_runs=('run','nunique'),n_pass=('status',lambda x:int(x.eq('PASS').sum())),
        n_fail=('status',lambda x:int(x.eq('FAIL').sum())),
        mean_error_mm=('localization_error_mm','mean'),median_error_mm=('localization_error_mm','median'),
        sd_error_mm=('localization_error_mm','std')).reset_index()
    bs['pass_rate']=bs.n_pass/bs.n_runs
    per_subject=pd.concat([per_subject,bs],ignore_index=True)
    if set(result.method)!=set(METHODS):raise ValueError('Not all 12 methods present')
    result.to_csv(out/'02_all_strategy_evaluations.csv',index=False)
    per_subject.to_csv(out/'03_subject_strategy_summary.csv',index=False)
    summary=summarize(result);summary.to_csv(out/'04_method_strategy_summary.csv',index=False)
    p,d=paired(result);p.to_csv(out/'05_loso_vs_global_paired_runs.csv',index=False)
    d.to_csv(out/'06_loso_vs_global_paired_summary.csv',index=False)
    pairs=pair_summary(result);pairs.to_csv(out/'07_single_reference_42_pairs.csv',index=False)
    baseline_manifest=baseline[['method','strategy','test_subject','reference_subject','configuration_key','configuration_parameters']].drop_duplicates()
    chosen=pd.concat([selected[['method','strategy','test_subject','reference_subject','configuration_key','configuration_parameters']],baseline_manifest],ignore_index=True)
    chosen.to_csv(out/'08_chosen_configuration_manifest.csv',index=False)
    if not a.no_figures:
        figures(summary,per_subject,pairs,d,out,a.dpi)
        strategy_boxplots(result,out,a.dpi)
    report={'methods':len(result.method.unique()),'all_rows':len(result),
            'loso_rows':int((result.strategy=='loso').sum()),
            'single_reference_rows':int((result.strategy=='single_reference').sum()),
            'global_rows':int((result.strategy=='script26_fixed_in_sample').sum()),
            'source_csv':str(src/'11_all_12_methods_canonical_grid_backup.csv'),
            'baseline_csv':str(a.script26_baseline),
            'warning':'Script 26 fixed baseline was selected using all 7 subjects; descriptive/in-sample, not held-out.',
            'paired_analysis':'Only jointly PASS results have numeric paired deltas; failures separately tabulated.'}
    (out/'00_analysis_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print('SCRIPT 40 COMPLETE\n'+json.dumps(report,indent=2))
    print('Output:',out)

if __name__=='__main__':main()

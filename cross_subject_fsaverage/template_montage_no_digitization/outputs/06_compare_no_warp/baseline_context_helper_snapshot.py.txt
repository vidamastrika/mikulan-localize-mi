"""Historical Script 26 baseline context for Script 06; no reconstruction/MRI IO.

Baseline is the 732 complete_all_pass globally selected within-subject errors.
These use a different metric/selection design from the cross-model diagnostics.
Collapse repeated reference pairs within each run before attaching the baseline;
never count its 732 observations six times or call mixed-metric gaps improvements.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RUN_KEY=['method','test_subject','run']
VALUES=['baseline_localization_error_mm','no_warp_run_diagnostic_mm','template_run_diagnostic_mm']
LABELS=['Historical baseline: within-subject error','No-warp: unregistered HEAD distance',
        'Template/fsaverage: unregistered HEAD distance']


def require(condition,message):
    if not condition:raise ValueError(message)


def prepare(baseline_path,definitions_path,runs,methods):
    baseline=pd.read_csv(baseline_path,keep_default_na=False).rename(
        columns={'subject':'test_subject','localization_error_mm':'baseline_localization_error_mm',
                 'configuration_key':'baseline_configuration_key'})
    definitions=pd.read_csv(definitions_path,keep_default_na=False)
    require(len(baseline)==732 and not baseline.duplicated(RUN_KEY).any()
            and set(baseline.method)==set(methods) and baseline.status.eq('PASS').all(),
            'Historical baseline must contain 732 unique all-PASS method/test-run values')
    require(len(definitions)==12 and definitions.method.is_unique and set(definitions.method)==set(methods)
            and definitions.report_mode.eq('complete_all_pass').all(),
            'Expected 12 Script 26 all-pass selected configuration definitions')
    selected=dict(zip(definitions.method,definitions.configuration_key))
    require(baseline.baseline_configuration_key.eq(baseline.method.map(selected)).all(),
            'Historical baseline coordinates do not match selected definitions')
    baseline['baseline_localization_error_mm']=pd.to_numeric(baseline.baseline_localization_error_mm,errors='raise')
    require(np.isfinite(baseline.baseline_localization_error_mm).all()
            and baseline.baseline_localization_error_mm.ge(0).all(),'Invalid baseline errors')
    expected=set(map(tuple,runs[RUN_KEY].drop_duplicates().to_numpy()))
    require(set(map(tuple,baseline[RUN_KEY].to_numpy()))==expected,'Baseline and comparison test-run/method keys differ')
    require(baseline.groupby('method').size().eq(61).all(),'Baseline requires 61 runs per method')
    baseline['metric']='within_subject_anatomical_localization_error_mm'
    baseline['selection']='historical_in_sample_best_fixed_configuration_across_61_runs'
    context=runs.merge(baseline[RUN_KEY+['baseline_localization_error_mm','baseline_configuration_key']],
                       on=RUN_KEY,how='left',validate='many_to_one')
    context=context.rename(columns={'no_warp_paired_median_mm':'no_warp_run_diagnostic_mm',
                                    'template_paired_median_mm':'template_run_diagnostic_mm'})
    context['common_run_available']=context[VALUES].notna().all(axis=1)
    context['metric_comparability']='context_only_mixed_metric_definitions_and_selection_designs'
    def summarize(groups):
        rows=[]
        for values,g in context.groupby(groups,sort=True):
            if not isinstance(values,tuple):values=(values,)
            common=g.loc[g.common_run_available]
            row=dict(zip(groups,values))|dict(n_runs_planned=len(g),n_common_runs=len(common),
                baseline_all_runs_median_error_mm=g.baseline_localization_error_mm.median())
            for column in VALUES:
                row[column.replace('_mm','_median_mm')]=common[column].median()
                row[column.replace('_mm','_q25_mm')]=common[column].quantile(.25)
                row[column.replace('_mm','_q75_mm')]=common[column].quantile(.75)
            rows.append(row)
        return pd.DataFrame(rows)
    return baseline,definitions,context,summarize(['strategy','method']),summarize(['strategy','method','test_subject'])


def figures(context,method,participant,out,colors,styles,methods,dpi):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    strategies=['LOSO','Single-reference'];written=[]
    for strategy,number in [('LOSO',7),('Single-reference',8)]:
        g=context.loc[context.strategy.eq(strategy)&context.common_run_available]
        fig,ax=plt.subplots(figsize=(20,9))
        for offset,column,alpha,hatch in [(-.25,VALUES[0],.3,'//'),(0,VALUES[1],.55,None),(.25,VALUES[2],.9,None)]:
            arrays=[g.loc[g.method.eq(m),column].to_numpy(float) for m in methods]
            boxes=ax.boxplot(arrays,positions=np.arange(12)+offset,widths=.22,manage_ticks=False,patch_artist=True,
                             flierprops=dict(marker='o',markersize=2,alpha=.25),medianprops=dict(color='#222222',linewidth=1))
            for m,box in zip(methods,boxes['boxes']):box.set(facecolor=colors[m],alpha=alpha,hatch=hatch)
        ax.set_xticks(np.arange(12),methods,rotation=35,ha='right')
        ax.set_ylabel('Distance (mm); metric differs by condition')
        ax.set_title(f'{strategy}: historical 12-method baseline context\nCommon test runs; Single Reference collapsed to median over paired successful references')
        ax.grid(axis='y',alpha=.2);ax.legend(handles=[Patch(facecolor='#555555',alpha=a,hatch=h,label=label)
            for a,h,label in zip([.3,.55,.9],['//',None,None],LABELS)],loc='upper right',fontsize=9,frameon=False)
        fig.text(.5,.01,'Baseline: in-sample selected within-subject errors. Other conditions: unregistered diagnostics. Not a three-way anatomical accuracy ranking.',
                 ha='center',fontsize=10)
        fig.tight_layout(rect=(0,.04,1,1))
        name=f'{number:02d}_{strategy.lower()}_baseline_context.png'
        fig.savefig(out/name,dpi=dpi,bbox_inches='tight');plt.close(fig);written.append(name)
    fig,axes=plt.subplots(2,3,figsize=(22,12),sharey=True)
    for i,strategy in enumerate(strategies):
        selected=participant.loc[participant.strategy.eq(strategy)]
        for j,column in enumerate(VALUES):
            ax=axes[i,j]
            pivot=selected.pivot(index='test_subject',columns='method',values=column.replace('_mm','_median_mm'))
            for m in methods:
                marker,line=styles[m]
                ax.plot(np.arange(len(pivot)),pivot[m],color=colors[m],marker=marker,linestyle=line,lw=1.2,alpha=.82,label=m)
            ax.set_xticks(np.arange(len(pivot)),pivot.index,rotation=30);ax.set_title(f'{strategy}\n{LABELS[j]}',fontsize=11)
            ax.grid(axis='y',alpha=.2);ax.set_xlabel('Participant')
        axes[i,0].set_ylabel('Participant median distance (mm)\nMetric differs between columns')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=6,loc='lower center',frameon=False)
    fig.suptitle('All 12 methods, including sub-03: historical baseline context\nCommon runs within each method/strategy; baseline selection and metric differ')
    fig.tight_layout(rect=(0,.1,1,.94));name='09_baseline_context_by_participant.png'
    fig.savefig(out/name,dpi=dpi,bbox_inches='tight');plt.close(fig);written.append(name)
    return written


def report_lines(method,methods):
    lines=['','## Historical 12-method baseline context','',
        'Baseline source: Script 26 complete_all_pass globally selected fixed configurations, 732 unique method/run results. '
        'This is an in-sample within-subject anatomical error baseline, not the same metric or selection design as cross-model unregistered HEAD distances. '
        'No MRI mapping is added to either cross-model condition. No baseline-relative improvement percentages or paired error deltas are calculated.','',
        'Each Single Reference recording is collapsed to the median distance over the same successful reference pairs before adding one baseline observation. '
        'All three distributions use common available test runs per method/strategy. Baseline is stored once per method/run; repeating its value in two strategy views does not add independent observations.','',
        '| Strategy | Method | Common runs | Historical within-subject baseline median (mm) | No-warp run diagnostic median (mm) | Template run diagnostic median (mm) |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    for strategy in ['LOSO','Single-reference']:
        for m in methods:
            r=method.loc[method.strategy.eq(strategy)&method.method.eq(m)].iloc[0]
            lines.append(f'| {strategy} | {m} | {int(r.n_common_runs)} | {r.baseline_localization_error_median_mm:.2f} | '
                         f'{r.no_warp_run_diagnostic_median_mm:.2f} | {r.template_run_diagnostic_median_mm:.2f} |')
    lines += ['', 'Tables 11–15 supply the original baseline rows/definitions, per-run context, method context and participant context. '
        'Figures 07/08 show the three-condition distributions; 09 shows participant profiles. Metric labels must remain visible when sharing these figures.']
    return lines

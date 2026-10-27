#!/usr/bin/env python3
"""Descriptive cross-subject comparison. Run from repository root.
Merges Script 34, Script 35, and the four-method Script 36 CSV first.
All figures and descriptive tables use the validated merged result.
Years and groups reproduce Script 26, not claims about invention dates.
PASS_PROVISIONAL remains provisional. No inferential tests are performed.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

GROUPS=[('Distributed / minimum-norm',['MNE','dSPM','sLORETA','eLORETA']),('Sparse',['MxNE','irMxNE']),('Dipole-based',['Continuous-ECD','RAP-MUSIC']),('Beamformer',['LCMV']),('Bayesian / hierarchical',['Gamma-MAP','Champagne','SESAME'])]
YEARS={'MNE':1994,'dSPM':2000,'sLORETA':2002,'eLORETA':2007,'Continuous-ECD':1985,'MxNE':2012,'RAP-MUSIC':1999,'LCMV':1997,'irMxNE':2016,'Champagne':2010,'Gamma-MAP':2009,'SESAME':2014}
COLORS={'MNE':'#4C78A8','dSPM':'#E45756','sLORETA':'#54A24B','eLORETA':'#8E5C99','MxNE':'#72B7B2','irMxNE':'#2D6A9F','Continuous-ECD':'#F58518','RAP-MUSIC':'#B279A2','LCMV':'#ECA82C','Gamma-MAP':'#8561A8','Champagne':'#D65A31','SESAME':'#299C87'}
METRICS=['within_subject_error_mm','cross_affine_error_mm','cross_syn_error_mm']
KEYS=['test_subject','reference_subject','run']

def plot(tab, methods, out, subtitle):
    """Stack method families; three upright condition boxes per method."""
    groups=[(g,[m for m in members if m in methods]) for g,members in GROUPS]
    groups=[(g,members) for g,members in groups if members]
    condition_labels=['Baseline (within-subject)','Affine','SyN']
    condition_colors=['#5877A3','#E59B45','#399E88']
    fig,axes=plt.subplots(len(groups),1,figsize=(12,4.0*len(groups)+2.2),
                          sharey=True,squeeze=False)
    ymax=max(1,float(tab[METRICS].max().max())*1.06)
    max_members=max(len(members) for _,members in groups)
    for ax,(group,members) in zip(axes[:,0],groups):
        shift=(max_members-len(members))/2
        centers=np.arange(len(members),dtype=float)+shift
        for i,method in enumerate(members):
            if i%2==0: ax.axvspan(centers[i]-.47,centers[i]+.47,color='#F3F5F7',zorder=0)
            for offset,metric,color in zip([-.24,0,.24],METRICS,condition_colors):
                values=tab.loc[tab.method.eq(method),metric].dropna()
                if len(values):
                    bp=ax.boxplot([values],positions=[centers[i]+offset],vert=True,
                                  widths=.20,patch_artist=True,showfliers=True,
flierprops=dict(
    marker='o',
    markersize=3,
    markerfacecolor=color,
    markeredgecolor='none',
    alpha=0.5,
),
                                  manage_ticks=False)
                    bp['boxes'][0].set(facecolor=color,edgecolor='#39434D',linewidth=.9)
                    bp['medians'][0].set(color='#17212B',linewidth=1.5)
                    for artist in bp['whiskers']+bp['caps']:
                        artist.set(color='#39434D',linewidth=.9)
        ax.set_xticks(centers,[f'{m} ({YEARS[m]})' for m in members],fontsize=10)
        ax.set_xlim(-.55,max_members-.45)
        ax.set_ylim(0,ymax)
        ax.set_ylabel('Localization error (mm)',fontsize=9)
        ax.set_title(group,loc='left',fontsize=12,fontweight='bold',pad=9)
        ax.grid(axis='y',alpha=.23);ax.set_axisbelow(True)
        ax.spines[['top','right']].set_visible(False)
        ax.tick_params(axis='x',length=0,pad=8)
    fig.suptitle('EEG source localization across subjects',fontsize=17,y=.985)
    fig.text(.5,.958,subtitle,ha='center',va='top',fontsize=9)
    fig.legend(handles=[Patch(facecolor=c,label=l) for c,l in zip(condition_colors,condition_labels)],
               loc='upper center',bbox_to_anchor=(.5,.925),ncol=3,frameon=False,fontsize=10)
    fig.text(.5,.018,'Each method: baseline, affine, SyN (left to right). Common error scale across all groups.\n'
             'Boxes: median and IQR; whiskers: 1.5 × IQR. Outlier shown as dots; full data range retained. Years follow Script 26.',
             ha='center',fontsize=8,color='#555555')
    fig.subplots_adjust(left=.10,right=.98,top=.88,bottom=.065,hspace=.50)
    fig.savefig(out,dpi=240,facecolor='white');plt.close(fig)

def merge_evaluation_csvs(sources):
    """Keep all original columns and failures; reject missing/overlapping methods."""
    frames=[]
    aliases={'C-ECD':'Continuous-ECD','ECD':'Continuous-ECD',
             'continuous_ecd':'Continuous-ECD','rap_music':'RAP-MUSIC',
             'gamma_map':'Gamma-MAP','irmxne':'irMxNE','mxne':'MxNE',
             'lcmv':'LCMV','champagne':'Champagne','sesame':'SESAME'}
    for label,path,expected_methods in sources:
        if not path.is_file():
            raise FileNotFoundError(f'{label}: {path}. Supply the correct path using its input option.')
        frame=pd.read_csv(path)
        required=['method','status']+KEYS+METRICS
        missing=set(required)-set(frame)
        if missing: raise ValueError(f'{path}: missing columns {sorted(missing)}')
        if frame[['method','status']+KEYS].isna().any().any():
            raise ValueError(f'{path}: missing method/status/case identifiers')
        frame['original_method_label']=frame['method']
        frame['method']=frame['method'].astype(str).str.strip().replace(aliases)
        if set(frame.method)!=set(expected_methods):
            raise ValueError(f'{label}: expected {sorted(expected_methods)}, found {sorted(set(frame.method))}')
        frame['comparison_source_file']=str(path)
        frame['comparison_source_script']=label
        frames.append(frame)
    merged=pd.concat(frames,ignore_index=True,sort=False)
    duplicates=merged.duplicated(['method']+KEYS,keep=False)
    if duplicates.any():
        raise ValueError('Duplicate method/case rows: '+merged.loc[duplicates,['method']+KEYS].head().to_string(index=False))
    return merged


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', '--script36', dest='script36',type=Path,default=Path('cross_subject/outputs/36_four_methods_combined/02_localization_errors.csv'),help='Four-method merged CSV from Script 36')
    p.add_argument('--script34',type=Path,default=Path('cross_subject/outputs/34_cross_subject_minimum_norm/02_localization_errors.csv'))
    p.add_argument('--script35',type=Path,default=Path('cross_subject/outputs/35_cross_subject_ecd_music_lcmv_mxne/02_localization_errors.csv'))
    p.add_argument('--output-root',type=Path,default=Path('cross_subject/outputs/37_cross_subject_comparison'))
    p.add_argument('--overwrite',action='store_true')
    a=p.parse_args()
    d=merge_evaluation_csvs([
        ('34',a.script34,['MNE','dSPM','sLORETA','eLORETA']),
        ('35',a.script35,['Continuous-ECD','RAP-MUSIC','LCMV','MxNE']),
        ('36',a.script36,['irMxNE','Gamma-MAP','Champagne','SESAME']),
    ])
    methods=[m for _,ms in GROUPS for m in ms if m in set(d.method)]
    expected=None
    for m,g in d.groupby('method'):
        keys=set(map(tuple,g[KEYS].values))
        subjects=set(g.test_subject)
        reference_counts=g.groupby(['test_subject','run']).reference_subject.nunique()
        if (len(keys)!=366 or len(subjects)!=7 or set(g.reference_subject)!=subjects
                or len(reference_counts)!=61 or not reference_counts.eq(6).all()
                or (g.test_subject==g.reference_subject).any()):
            raise ValueError(f'{m}: expected 61 EEG runs, 7 subjects and 6 references per run (366 cases)')
        if expected is None: expected=keys
        if keys!=expected: raise ValueError('Methods have different case schedules')
    for c in METRICS:d[c]=pd.to_numeric(d[c],errors='raise')
    valid=d.status.isin(['PASS','PASS_PROVISIONAL']) & np.isfinite(d[METRICS]).all(axis=1) & d[METRICS].ge(0).all(axis=1)
    d['comparison_complete']=valid
    # Within-subject value must be identical across references of a test run.
    for key,g in d.groupby(['method','test_subject','run']):
        v=g.within_subject_error_mm.dropna()
        if len(v) and v.max()-v.min()>1e-6: raise ValueError(f'Within-subject inconsistency: {key}')
    coverage=d.groupby('method').agg(attempted=('status','size'),complete=('comparison_complete','sum'))
    coverage['incomplete']=coverage.attempted-coverage.complete
    good=d[valid].copy()
    common=set.intersection(*[set(map(tuple,g[KEYS].values)) for _,g in good.groupby('method')]) if good.method.nunique()==len(methods) else set()
    matched=good[[tuple(row) in common for row in good[KEYS].values]].copy()
    root=a.output_root
    if root.exists() and any(root.iterdir()) and not a.overwrite: raise FileExistsError('Output folder not empty. Use --overwrite.')
    root.mkdir(parents=True,exist_ok=True)
    d.to_csv(root/'01_all_methods_evaluation.csv',index=False)
    # Reload the authoritative merged output: all downstream products use it.
    d=pd.read_csv(root/'01_all_methods_evaluation.csv')
    good=d.loc[d['comparison_complete']].copy()
    matched=good[[tuple(row) in common for row in good[KEYS].values]].copy()
    d.loc[~d['comparison_complete']].to_csv(root/'07_incomplete_cases.csv',index=False)
    d.groupby(['comparison_source_script','comparison_source_file','method']).size().rename('rows').to_csv(root/'08_input_provenance.csv')
    coverage.reindex(methods).to_csv(root/'02_completeness.csv')
    d.groupby(['method','status']).size().rename('count').to_csv(root/'03_status_counts.csv')
    for name,frame in [('available',good),('matched',matched)]:
        if frame.empty: print(f'No {name} cases; figure skipped');continue
        runs=frame.groupby(['method','test_subject','run'],as_index=False)[METRICS].mean()
        counts=frame.groupby(['method','test_subject','run']).size().rename('n_references').reset_index()
        runs=runs.merge(counts,validate='one_to_one')
        runs.to_csv(root/f'04_{name}_run_level.csv',index=False)
        runs.groupby('method')[METRICS].agg(['count','mean','median','std']).to_csv(root/f'05_{name}_summary.csv')
        n=runs.groupby('method').size()
        caption=f'{len(methods)} methods · one value per EEG run · cross errors averaged across successful references'
        if name=='matched':caption+=f'\nMatched reference–test cases: {len(common)}/366; EEG runs per method: {n.min()}–{n.max()}'
        else:caption+='\nAvailable successful cases; reference coverage may differ between methods'
        plot(runs,methods,root/f'06_{name}_boxplot.png',caption)
    (root/'README.txt').write_text('Descriptive comparison, not inferential testing.\nAll failure rows and provisional statuses retained in 01_all_methods_evaluation.csv.\nCross-subject boxes show per-EEG-run means over available successful references, not 366 independent observations.\nWithin-subject values are deduplicated per EEG run and validated for consistency.\nMatched figure uses identical successful reference–test cases for all loaded methods.\nMissing values are never replaced with zero. Outlier points hidden; original values retained.\nGroups and reference-publication years copied from Script 26.\nAll 12 methods are required. Inputs: Script 34, Script 35, and the four-method Script 36 merge. Script 36 remains unchanged. Use validated Script 35 results after the LCMV audit.\nScript 38 is reserved for reference sensitivity and statistical inference with participant-level dependence.\n')
    print(coverage.to_string());print('Common successful cases:',len(common));print('Saved:',root)

if __name__=='__main__':main()

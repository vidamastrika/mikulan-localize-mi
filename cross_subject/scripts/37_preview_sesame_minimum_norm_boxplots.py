#!/usr/bin/env python3
"""Read-only plots: SESAME (Script 32) vs four minimum-norm methods (Script 34)."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

METHODS=['SESAME','MNE','dSPM','sLORETA','eLORETA']
METRICS=[('Within','within_subject_error_mm'),('Cross Affine','cross_affine_error_mm'),('Cross SyN','cross_syn_error_mm')]
ALIASES={'within_subject_error_mm':['within_subject_error_mm','within_error_mm'],
         'cross_affine_error_mm':['cross_affine_error_mm','affine_error_mm'],
         'cross_syn_error_mm':['cross_syn_error_mm','syn_error_mm']}

def find_csv(root,number,explicit):
    if explicit:
        p=Path(explicit)
        if not p.is_file():raise FileNotFoundError(p)
        return p
    base=root/'cross_subject'/'outputs'
    preferred=base/'32_all_reference_localization_evaluation'/'cross_subject_localization_errors.csv'
    if number==32 and preferred.is_file():return preferred
    files=list(base.glob(f'{number}*/**/*localization_errors*.csv'))
    if len(files)!=1:raise ValueError(f'Script {number}: found {len(files)} files. Set --sesame-csv or --minimum-norm-csv explicitly: {files}')
    return files[0]

def load(path,is_sesame):
    df=pd.read_csv(path,low_memory=False)
    if is_sesame:df['method']='SESAME'
    else:
        col=next((x for x in ['method','inverse_method','method_name'] if x in df),None)
        if col is None:raise ValueError(f'{path}: missing method column. Found: {list(df)}')
        mapper={'mne':'MNE','dspm':'dSPM','sloreta':'sLORETA','eloreta':'eLORETA'}
        df['method']=df[col].astype(str).str.lower().str.replace('-','',regex=False).str.replace('_','',regex=False).map(mapper)
        df=df[df.method.isin(METHODS[1:])].copy()
    for col in ['test_subject','reference_subject','run']:
        if col not in df:raise ValueError(f'{path}: missing {col}')
    if 'status' in df:df=df[df.status.astype(str).str.upper().str.startswith('PASS')].copy()
    for _,canonical in METRICS:
        found=next((c for c in ALIASES[canonical] if c in df),None)
        if found is None:raise ValueError(f'{path}: no {canonical}. Found: {list(df)}')
        df[canonical]=pd.to_numeric(df[found],errors='coerce')
    return df

def chart(arrays,labels,title,path,showfliers=True):
    fig,ax=plt.subplots(figsize=(9,5.3))
    ax.boxplot(arrays,tick_labels=labels,showfliers=showfliers)
    ax.set_ylabel('Localization error (mm)')
    ax.set_title(title)
    ax.grid(axis='y',alpha=.2)
    fig.savefig(path,dpi=220,bbox_inches='tight')
    plt.close(fig)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--repo-root',type=Path,default=Path.cwd())
    p.add_argument('--sesame-csv',type=Path)
    p.add_argument('--minimum-norm-csv',type=Path)
    p.add_argument('--output-root',type=Path,default=Path('cross_subject/outputs/37_preview_sesame_minimum_norm'))
    p.add_argument('--allow-incomplete',action='store_true')
    a=p.parse_args()
    paths=[find_csv(a.repo_root,32,a.sesame_csv),find_csv(a.repo_root,34,a.minimum_norm_csv)]
    all_df=pd.concat([load(paths[0],True),load(paths[1],False)],ignore_index=True)
    within={};cross={};records=[]
    for method in METHODS:
        sub=all_df[all_df.method==method].copy()
        if sub.duplicated(['test_subject','run','reference_subject']).any():
            raise ValueError(f'{method}: duplicate test/run/reference')
        spread=sub.groupby(['test_subject','run'])['within_subject_error_mm'].agg(lambda v:v.max()-v.min())
        if (spread>1e-4).any():raise ValueError(f'{method}: within values differ across references')
        counts=sub.groupby(['test_subject','run']).reference_subject.nunique()
        if not a.allow_incomplete and (len(sub)!=366 or len(counts)!=61 or (counts!=6).any()):
            raise ValueError(f'{method}: expected 366 cross rows, 61 unique runs, 6 references/run. Got {len(sub)} rows and {len(counts)} runs')
        cross[method]=sub
        within[method]=sub.drop_duplicates(['test_subject','run'])
        for j,(label,field) in enumerate(METRICS):
            group=within[method] if j==0 else cross[method]
            v=group[field].dropna()
            records.append(dict(method=method,scenario=label,n=len(v),median_mm=v.median(),mean_mm=v.mean(),sd_mm=v.std()))
    a.output_root.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(records).to_csv(a.output_root/'01_boxplot_summary.csv',index=False)
    for j,(label,field) in enumerate(METRICS,1):
        groups=within if j==1 else cross
        chart([groups[m][field].dropna() for m in METHODS],METHODS,
              f'{label}: SESAME and minimum-norm methods',a.output_root/f'0{j}_{label.lower().replace(" ","_")}.png')
    # Grouped boxplot with 3 scenarios for each method
    fig,ax=plt.subplots(figsize=(12,5.7));positions=[];values=[]
    for i,m in enumerate(METHODS):
        for j,(_,field) in enumerate(METRICS):
            positions.append(4*i+j+1)
            values.append((within[m] if j==0 else cross[m])[field].dropna())
    bp=ax.boxplot(values,positions=positions,widths=.72,patch_artist=True,showfliers=False)
    shades=['#dddddd','#91b2c9','#b6cdb0']
    for k,patch in enumerate(bp['boxes']):patch.set_facecolor(shades[k%3])
    ax.set_xticks([4*i+2 for i in range(5)],METHODS)
    ax.set_ylabel('Localization error (mm)');ax.set_title('Within vs cross-subject — SESAME and minimum-norm methods')
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=c,label=m) for c,(m,_) in zip(shades,METRICS)])
    ax.grid(axis='y',alpha=.2)
    fig.savefig(a.output_root/'04_grouped_comparison.png',dpi=220,bbox_inches='tight');plt.close(fig)
    for i,m in enumerate(METHODS,5):
        chart([within[m][METRICS[0][1]].dropna(),cross[m][METRICS[1][1]].dropna(),cross[m][METRICS[2][1]].dropna()],
              ['Within','Affine','SyN'],f'{m}: within vs cross-subject',a.output_root/f'{i:02d}_{m.lower()}_boxplot.png')
    print('Data sources:',*[str(v) for v in paths],sep='\n  ')
    print(pd.DataFrame(records).to_string(index=False,float_format=lambda x:f'{x:.3f}'))
    print('Output:',a.output_root)
    print('Within=61 unique EEG runs; cross=366 test/reference pairs per method (if complete). Descriptive, not independent samples.')

if __name__=='__main__':main()

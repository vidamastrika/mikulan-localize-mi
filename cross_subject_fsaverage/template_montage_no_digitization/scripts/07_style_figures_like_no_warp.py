#!/usr/bin/env python3
"""Restyle completed Script 06 figures with no-warp method years/families.

Figure-only stage: reads saved comparison tables, verifies their manifest hashes,
then invokes existing numerical-free plotting functions. No MRI, inverse fits,
metric computation, parameter selection or summary-table rewriting. Reuses exact
METHOD_META, METHOD_ORDER and GROUP_ORDER literals from no-warp Script 09, with
Script 26 colors/markers already recorded by Script 06. Years are historical
report labels copied for consistency, not assertions of first invention dates.

Outputs replace the nine existing comparison PNGs only after all renders pass.
First originals and their manifest are preserved in figures_before_no_warp_style/.
Adds figure_method_metadata.csv and figure_style_manifest.json; updates only PNG
hashes/style provenance in comparison_manifest.json. All CSV analysis tables and
REPORT.md remain byte-identical. Code and input provenance is retained.

Dependencies: Python >=3.10, NumPy, pandas, Matplotlib. No MNE/ANTs imports.
Run manually from repository root, or invoke for a requested figure refresh:
 .venv/bin/python cross_subject_fsaverage/template_montage_no_digitization/scripts/07_style_figures_like_no_warp.py
Rerender a previously styled report: same command with --overwrite.
--comparison-root overrides the existing Script 06 output; --dpi defaults 300.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

VARIANT=Path(__file__).resolve().parents[1]
ROOT=VARIANT.parents[1]


def require(condition,message):
    if not condition:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);sys.modules[name]=result
    spec.loader.exec_module(result)
    return result


def no_warp_metadata(path):
    values={}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node,ast.AnnAssign) and isinstance(node.target,ast.Name):
            values[node.target.id]=node.value
        elif isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Name):values[target.id]=node.value
    meta=ast.literal_eval(values['METHOD_META'])
    methods=ast.literal_eval(values['METHOD_ORDER']);groups=ast.literal_eval(values['GROUP_ORDER'])
    require(len(methods)==12 and set(meta)==set(methods),'No-warp method metadata incomplete')
    ordered=[m for g in groups for m in methods if meta[m]['group']==g]
    require(ordered==methods,'No-warp families must be contiguous in plotting order')
    return meta,methods,groups


def decorate(fig,meta,methods,groups):
    """Add years and family bands to method axes; group method legends by family."""
    from matplotlib.ticker import FixedLocator
    method_set=set(methods)
    display_to_method={label:m for m in methods for label in
                       [m,f'{m} ({meta[m]["year"]})',f'{m}\n({meta[m]["year"]})']}
    horizontal=False;vertical=False
    for ax in fig.axes:
        for orientation,axis in [('x',ax.xaxis),('y',ax.yaxis)]:
            labels=[t.get_text() for t in axis.get_ticklabels()]
            # Right-hand panels can hide shared method tick labels. Still draw
            # their family bands using the method labels of a shared sibling.
            if orientation=='y' and not labels:
                for sibling in ax.get_shared_y_axes().get_siblings(ax):
                    other=[display_to_method.get(t.get_text(),t.get_text()) for t in sibling.get_yticklabels()]
                    if len(other)==12 and set(other)==method_set:
                        labels=other;break
            labels=[display_to_method.get(label,label) for label in labels]
            if len(labels)!=12 or set(labels)!=method_set:continue
            positions=axis.get_ticklocs()
            require(labels==methods,'Figure method order does not match no-warp families')
            axis.set_major_locator(FixedLocator(positions))
            if orientation=='x':
                vertical=True
                ax.set_xticks(positions,[f'{m}\n({meta[m]["year"]})' for m in methods],rotation=0,ha='center',fontsize=9)
                ax.tick_params(axis='x',pad=9,length=0)
                ax.set_xlim(positions[0]-.6,positions[-1]+.6)
            else:
                horizontal=True
                ax.set_yticks(positions,[f'{m} ({meta[m]["year"]})' for m in methods],fontsize=9)
            for i,group in enumerate(groups):
                indices=[j for j,m in enumerate(methods) if meta[m]['group']==group]
                lo,hi=positions[indices[0]]-.5,positions[indices[-1]]+.5
                centre=(lo+hi)/2
                if orientation=='x':
                    if i%2==0:ax.axvspan(lo,hi,color='#f5f6f8',zorder=-1)
                    if indices[-1]<11:ax.axvline(hi,color='#c8cbd0',lw=.9,zorder=0)
                    ax.text(centre,-.23,group,transform=ax.get_xaxis_transform(),
                            ha='center',va='top',fontsize=9.5,fontweight='bold',clip_on=False)
                else:
                    if i%2==0:ax.axhspan(lo,hi,color='#f5f6f8',zorder=-1)
                    if indices[-1]<11:ax.axhline(hi,color='#c8cbd0',lw=.9,zorder=0)
                    ax.text(1.025,centre,group,transform=ax.get_yaxis_transform(),
                            rotation=270,ha='left',va='center',fontsize=8.5,fontweight='bold',clip_on=False)
    # Participant plots use method legends instead of method axes. Preserve
    # condition legends, replacing only the complete 12-method figure legend.
    grouped=False
    for legend in list(fig.legends):
        names=[t.get_text() for t in legend.get_texts()]
        if set(names)!=method_set:continue
        handles=legend.legend_handles
        mapping=dict(zip(names,handles));legend.remove();grouped=True
        for i,group in enumerate(groups):
            names=[m for m in methods if meta[m]['group']==group]
            fig.legend([mapping[m] for m in names],[f'{m} ({meta[m]["year"]})' for m in names],
                       title=group,loc='lower center',bbox_to_anchor=((i+.5)/len(groups),.012),
                       frameon=False,ncol=1,fontsize=8.5,title_fontsize=9)
    if vertical:
        fig.subplots_adjust(left=.065,right=.985,bottom=.30,top=.84,wspace=.20)
        for text in fig.texts:
            if text.get_text().startswith('Baseline: in-sample'):
                text.set_y(.065)
    elif horizontal:
        fig.subplots_adjust(left=.13,right=.94,bottom=.14,top=.83,wspace=.48)
    elif grouped:
        # Keep the family-grouped legends below the participant panels.
        fig.subplots_adjust(bottom=.23,top=.88,hspace=.42,wspace=.18)
    return dict(method_axis_years=vertical or horizontal,grouped_family_legend=grouped)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison-root',type=Path,default=VARIANT/'outputs/06_compare_no_warp')
    parser.add_argument('--dpi',type=int,default=300)
    parser.add_argument('--overwrite',action='store_true')
    args=parser.parse_args(argv)
    comparison=args.comparison_root.expanduser().resolve()
    require((VARIANT/'outputs').resolve() in comparison.parents,'Comparison must be within this variant/outputs')
    require(args.dpi>0,'DPI must be positive')
    style_manifest=comparison/'figure_style_manifest.json'
    require(not style_manifest.exists() or args.overwrite,'Style already applied; use --overwrite to redraw')
    manifest_path=comparison/'comparison_manifest.json'
    manifest=json.loads(manifest_path.read_text())
    require(manifest['status']=='PRIMARY_NO_REGISTRATION_DIAGNOSTIC_COMPARISON_COMPLETE','Completed Script 06 required')
    hashes_before={}
    for name,expected in manifest['output_sha256'].items():
        actual=sha(comparison/name);require(actual==expected,f'Comparison output changed: {name}')
        hashes_before[name]=actual
    style_source=ROOT/'cross_subject_no_warp/scripts/09_visualize_results.py'
    meta,methods,groups=no_warp_metadata(style_source)
    colors_table=pd.read_csv(comparison/'method_styles.csv')
    colors=dict(zip(colors_table.method,colors_table.color))
    styles={r.method:(r.marker,r.linestyle) for r in colors_table.itertuples()}
    require(set(colors)==set(methods),'Saved method colors incomplete')
    # Import numerical-free plotting definitions only. Neither script main runs.
    compare_code=Path(__file__).with_name('06_compare_no_warp.py')
    baseline_code=Path(__file__).with_name('comparison_baseline_context.py')
    compare=module(compare_code,'template_existing_comparison_figures');compare.METHODS=methods
    baseline=module(baseline_code,'template_existing_baseline_figures')
    tables=comparison/'tables'
    paths={
        'pair':tables/'01_paired_jobs.csv','method':tables/'02_method_summary.csv',
        'participant':tables/'03_participant_summary.csv','baseline_runs':tables/'13_baseline_context_unique_runs.csv',
        'baseline_methods':tables/'14_baseline_context_method_summary.csv',
        'baseline_participants':tables/'15_baseline_context_participant_summary.csv'}
    data={name:pd.read_csv(path) for name,path in paths.items()}
    source_hashes={str(p):sha(p) for p in [style_source,compare_code,baseline_code,Path(__file__),*paths.values()]}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    original_save=Figure.savefig;checks=[]
    def save(fig,*a,**kw):
        checks.append(decorate(fig,meta,methods,groups))
        return original_save(fig,*a,**kw)
    # Render all nine figures before replacing any originals.
    with tempfile.TemporaryDirectory(prefix='.no_warp_style_',dir=comparison) as temp:
        stage=Path(temp)
        try:
            Figure.savefig=save
            names=compare.figures(data['pair'],data['method'],data['participant'],stage/'figures',colors,styles,args.dpi)
            names+=baseline.figures(data['baseline_runs'],data['baseline_methods'],data['baseline_participants'],
                                    stage/'figures',colors,styles,methods,args.dpi)
        finally:
            Figure.savefig=original_save;plt.close('all')
        require(len(names)==9 and set(names)==set(manifest['figures']),'Regenerated figure identities differ')
        require(len(checks)==9 and all(c['method_axis_years'] or c['grouped_family_legend'] for c in checks),
                'One or more figures lack method years/families')
        for name in names:require((stage/'figures'/name).stat().st_size>0,f'Missing image: {name}')
        for name,expected in hashes_before.items():require(sha(comparison/name)==expected,f'Input changed while rendering: {name}')
        backup=comparison/'figures_before_no_warp_style'
        if not backup.exists():
            backup.mkdir()
            shutil.copy2(manifest_path,backup/'comparison_manifest.json')
            for name in names:shutil.copy2(comparison/'figures'/name,backup/name)
        for name in names:os.replace(stage/'figures'/name,comparison/'figures'/name)
    metadata_path=comparison/'figure_method_metadata.csv'
    pd.DataFrame([dict(method=m,year=meta[m]['year'],family=meta[m]['group'],color=colors[m],
                       marker=styles[m][0],linestyle=styles[m][1]) for m in methods]).to_csv(metadata_path,index=False)
    report=dict(created_utc=datetime.now(timezone.utc).isoformat(),status='FIGURES_RESTYLED_NO_WARP_YEARS_AND_FAMILIES',
        source_sha256=source_hashes,method_order=methods,method_metadata=meta,family_order=groups,
        year_policy='Exact historical no-warp Script 09 display labels; not re-dated or claimed as first invention years',
        palette_policy='Retain Script 26 method colors and marker styles from completed comparison',
        figures=names,dpi=args.dpi,artist_checks=checks,original_backup=str(backup),
        comparison_tables_changed=False,metrics_changed=False,MRI_alignment_used=False,
        output_sha256={str(Path('figures')/name):sha(comparison/'figures'/name) for name in names})
    style_manifest.write_text(json.dumps(report,indent=2)+'\n')
    for name in names:manifest['output_sha256'][str(Path('figures')/name)]=sha(comparison/'figures'/name)
    for name in ['figure_method_metadata.csv','figure_style_manifest.json']:
        manifest['output_sha256'][name]=sha(comparison/name)
    manifest['figure_style']=dict(source=str(style_source),source_sha256=sha(style_source),
                                  method_order=methods,families=groups,dpi=args.dpi,
                                  metadata_csv='figure_method_metadata.csv',manifest='figure_style_manifest.json')
    manifest_path.write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    for name,expected in hashes_before.items():
        if not name.startswith('figures/') and name not in ['figure_method_metadata.csv','figure_style_manifest.json']:
            require(sha(comparison/name)==expected,f'Analysis output changed unexpectedly: {name}')
    print(f'FIGURES_RESTYLED_NO_WARP_YEARS_AND_FAMILIES: {comparison/"figures"}')
    print(f'{len(names)} figures; numerical tables and report unchanged; originals backed up in {backup}')


if __name__=='__main__':
    main()

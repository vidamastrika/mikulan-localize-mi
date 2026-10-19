#!/usr/bin/env python3
"""Script 26: 12-method Localize-MI comparison, using Script 17 plotting style.

Two independent reporting cohorts:
 A complete_any_status: one observation for every one of 61 runs, FAIL allowed.
 B complete_all_pass: the same 61 observations must all be PASS.

For each cohort, select a single fixed configuration per method by the LOWEST
median localization error among successful observations, breaking ties by more
successful runs, then lower mean error, then stable configuration ID. Report A
medians condition on successful observations and must be interpreted with PASS
coverage. Overall-grid values are within-run medians across eligible PASS
solutions, as in Script 17 (not pooled solution medians).

This is an in-sample descriptive comparison, not unseen-subject validation.
Script 25's 06/07 CSVs must first be validated as per its audit output.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ['MNE','dSPM','sLORETA','eLORETA']
OTHER = ['Continuous-ECD','MxNE','RAP-MUSIC','LCMV']
NEW = ['irMxNE','Champagne','Gamma-MAP','SESAME']
METHODS = ORIGINAL + OTHER + NEW
NEW_NAMES = {'irmxne':'irMxNE','champagne':'Champagne','gamma_map':'Gamma-MAP','sesame':'SESAME'}
COLORS = {'irMxNE':'#2D6A9F','Champagne':'#D65A31','Gamma-MAP':'#8561A8','SESAME':'#299C87'}

# Reference publication years, not necessarily first invention dates.
# Continuous-ECD: 1985 is an established dipole-model paper, not an origin claim.
METHOD_YEARS = {
    'MNE': 1994, 'dSPM': 2000, 'sLORETA': 2002, 'eLORETA': 2007,
    'Continuous-ECD': 1985, 'MxNE': 2012, 'RAP-MUSIC': 1999,
    'LCMV': 1997, 'irMxNE': 2016, 'Champagne': 2010,
    'Gamma-MAP': 2009, 'SESAME': 2014,
}

def add_year_labels_to_figure(fig):
    """Edit display-only method labels; preserve all underlying data names."""
    labels = {m: f'{m} ({METHOD_YEARS[m]})' for m in METHODS}
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            # Force label Text objects to reflect their formatters before editing.
            for tick in axis.get_major_ticks():
                for item in (tick.label1, tick.label2):
                    value = item.get_text()
                    if value in labels:
                        item.set_text(labels[value])
            # FixedLocator + FixedFormatter avoids a redraw overwriting the text.
            ticks = axis.get_ticklocs()
            texts = [t.get_text() for t in axis.get_ticklabels()]
            if any(t in labels for t in texts) and len(ticks) == len(texts):
                axis.set_ticks(ticks, labels=[labels.get(t, t) for t in texts])
        legend = ax.get_legend()
        if legend is not None:
            for item in legend.get_texts():
                item.set_text(labels.get(item.get_text(), item.get_text()))
    for legend in fig.legends:
        for item in legend.get_texts():
            item.set_text(labels.get(item.get_text(), item.get_text()))



def arguments():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('--script17', type=Path, default=ROOT/'scripts/17_compare_all_methods_selected_and_overall.py')
    a.add_argument('--original-grid', type=Path, default=ROOT/'outputs/grid_all_runs/grid_results.csv')
    a.add_argument('--script15-root', type=Path, default=ROOT/'outputs/15_full_additional_method_grid')
    a.add_argument('--mxne-extra-results', type=Path, action='append', default=[])
    a.add_argument('--new-any', type=Path, default=ROOT/'outputs/25_next_method_audit/06_complete_any_status.csv')
    a.add_argument('--new-pass', type=Path, default=ROOT/'outputs/25_next_method_audit/07_complete_all_pass.csv')
    a.add_argument('--output-root', type=Path, default=ROOT/'outputs/26_twelve_methods_comparison')
    a.add_argument('--expected-runs', type=int, default=61)
    a.add_argument('--dpi', type=int, default=240)
    a.add_argument('--no-figures', action='store_true')
    a.add_argument('--overwrite', action='store_true')
    return a.parse_args()


def script17_module(path):
    if not path.is_file():
        raise FileNotFoundError(f'Script 17 not found: {path}. Supply --script17 FILE')
    spec = importlib.util.spec_from_file_location('localizemi_script17', path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    mod.METHOD_ORDER = METHODS[:]
    mod.METHOD_COLORS.update(COLORS)
    return mod


def reference_keys(frame):
    return set(map(tuple, frame[['subject','run']].astype(str).drop_duplicates().to_numpy()))


def normalize(frame, label, mod):
    mod.require_columns(frame, ['method','subject','run','status','montage'], label)
    out = mod.standardize_grid_error_column(frame, label)
    out = mod.normalize_status(out)
    out['subject'] = out.subject.astype(str)
    out['run'] = out.run.astype(str)
    out['method'] = out.method.astype(str)
    if out.loc[out.status.eq('PASS'), 'localization_distance_mm'].isna().any():
        raise ValueError(f'{label}: PASS row with missing error')
    return out


def old_grid(args, mod):
    original = mod.load_original_grid(args.original_grid)
    additional = mod.load_additional_grids(args.script15_root,
        args.mxne_extra_results, 'all' if args.mxne_extra_results else 'old')
    return pd.concat([original,additional], ignore_index=True, sort=False)


def prepare_old(old, reference):
    """Identify full configurations for old methods, not row-specific outcomes."""
    out=[]
    for method, group in old.groupby('method', sort=False):
        if method in ORIGINAL:
            fields = [c for c in ['method_origin','montage','loose','depth','snr','lambda2'] if c in group.columns]
        else:
            fields = ['montage','configuration_id']
        if not all(c in group for c in fields):
            raise ValueError(f'Cannot identify old configuration inputs for {method}: {fields}')
        part = group.copy()
        # Source grids use stable configuration identifiers; normalizing numeric
        # strings avoids splitting alpha=0.1 vs alpha=0.10.
        def norm(v):
            if pd.isna(v): return '<NA>'
            try:
                if not isinstance(v,str) or v.replace('.','',1).replace('-','',1).isdigit():
                    return f'{float(v):.12g}'
            except (TypeError,ValueError): pass
            return str(v).strip()
        cfg = part[fields].apply(lambda s:s.map(norm))
        part['configuration_key'] = (method + ':' + cfg.agg('|'.join,axis=1))
        part['configuration_parameters'] = cfg.apply(lambda r:json.dumps(r.to_dict(), sort_keys=True),axis=1)
        out.append(part)
    return pd.concat(out, ignore_index=True, sort=False)


def eligibility(grid, ref, require_all_pass):
    frames=[]; records=[]
    for (method, cfg), group in grid.groupby(['method','configuration_key'], sort=False, dropna=False):
        keys=reference_keys(group)
        counts=group.groupby(['subject','run'], dropna=False).size()
        # Duplicates must be resolved, never bias medians as extra solutions.
        if counts.max() > 1:
            same=group.groupby(['subject','run'],dropna=False).apply(
                lambda g:g[['status','localization_distance_mm']].fillna(-1).drop_duplicates().shape[0]==1,
                include_groups=False)
            if not same.all():
                raise ValueError(f'Conflicting duplicate observations: {method}/{cfg}')
            group=group.drop_duplicates(['subject','run']).copy()
        complete=(keys == ref and len(group)==len(ref))
        passed=int(group.status.eq('PASS').sum())
        eligible=complete and (not require_all_pass or passed==len(ref))
        records.append({'method':method,'configuration_key':cfg,
            'configuration_parameters':group.configuration_parameters.iloc[0],
            'runs':len(keys),'pass':passed,'fail':int(group.status.eq('FAIL').sum()),
            'eligible':eligible})
        if eligible: frames.append(group)
    if not frames: raise ValueError('No complete eligible configurations')
    return pd.concat(frames,ignore_index=True,sort=False),pd.DataFrame(records)


def load_new(path, expected_methods, ref, mod):
    data=normalize(pd.read_csv(path,low_memory=False),str(path),mod)
    data['method']=data.method.str.lower().str.replace('-','_',regex=False).map(NEW_NAMES)
    if set(data.method.dropna()) != set(expected_methods):
        raise ValueError(f'{path}: methods mismatch: {data.method.unique()}')
    for name in ['configuration_key','configuration_parameters']:
        if name not in data.columns: raise ValueError(f'{path}: missing {name}; rerun Script 25')
    grouped=data.groupby(['method','configuration_key'],dropna=False)
    for (method,cfg), group in grouped:
        if reference_keys(group)!=ref or len(group)!=len(ref):
            raise ValueError(f'{path}: configuration not exactly 61 reference runs: {method}/{cfg}')
        if group.duplicated(['subject','run']).any():
            raise ValueError(f'{path}: duplicate run in {method}/{cfg}')
    return data


def select_and_overall(grid, methods, ref, mode):
    defs=[]; selected=[]
    for method in methods:
        m=grid.loc[grid.method.eq(method)]
        if m.empty: raise ValueError(f'{mode}: no eligible configurations: {method}')
        stats=[]
        for cfg,g in m.groupby('configuration_key',sort=False):
            v=g.loc[g.status.eq('PASS'),'localization_distance_mm'].dropna().to_numpy(float)
            stats.append({'key':cfg,'median':np.median(v) if len(v) else np.inf,
                          'mean':np.mean(v) if len(v) else np.inf,
                          'pass':len(v)})
        ranked=sorted(stats,key=lambda v:(v['median'],-v['pass'],v['mean'],str(v['key'])))
        best=ranked[0]
        if not np.isfinite(best['median']):
            raise ValueError(f'{mode}: {method} has no successful localization in eligible grid')
        chosen=m[m.configuration_key.eq(best['key'])].copy()
        chosen['localization_error_mm']=chosen['localization_distance_mm'].where(chosen.status.eq('PASS'))
        chosen['view']='selected_configuration'
        selected.append(chosen)
        defs.append({'method':method,'configuration_key':best['key'],
            'configuration_parameters':chosen.configuration_parameters.iloc[0],
            'candidate_configurations_searched':len(stats),
            'selected_median_mm':best['median'],
            'selected_pass':best['pass'],'selected_fail':len(ref)-best['pass'],
            'selection_label':'lowest-median observed configuration (in sample)'})
    chosen=pd.concat(selected,ignore_index=True)
    return chosen,pd.DataFrame(defs)


def make_report(mod, grid, audit, ref, root, args, mode):
    import matplotlib.pyplot as plt
    destination=root/mode
    if destination.exists() and not args.overwrite:
        raise FileExistsError(f'{destination} exists; pass --overwrite')
    # Never remove existing outputs; overwrite individual named files only.
    tables=destination/'tables'; figures=destination/'figures'
    tables.mkdir(parents=True,exist_ok=True); figures.mkdir(parents=True,exist_ok=True)
    selected, definitions=select_and_overall(grid,METHODS,ref,mode)
    selected_all=selected[['subject','run','method','status','localization_error_mm','configuration_key']].copy()
    selected_pass=selected_all[selected_all.status.eq('PASS')].copy()
    selected_pass['view']='selected_configuration'
    # Keep failure as missing value for plotting and pairwise statistics.
    selected_plot=selected_all[['subject','run','method','localization_error_mm']].copy()
    wide=mod.to_wide(selected_plot)
    overall=mod.create_overall_run_values(grid.iloc[:0],grid)
    overall_pass=overall.dropna(subset=['localization_error_mm']).copy()
    selected_summary=mod.method_summary(selected_pass)
    overall_summary=mod.method_summary(overall_pass)
    compare=mod.selected_vs_overall_summary(selected_summary,overall_summary)
    # Merge reliability information explicitly, without treating failed errors as zero.
    compare=compare.merge(definitions[['method','selected_pass','selected_fail']],on='method')
    compare['selected_success_rate']=compare.selected_pass/len(ref)
    pair=mod.pairwise_summary(wide)
    overall_wide=mod.to_wide(overall)
    overall_pair=mod.pairwise_summary(overall_wide)
    participant=mod.participant_summary(selected_pass)
    overall_participant=mod.participant_summary(overall_pass)
    run=mod.run_summary(wide)
    overall_run=mod.run_summary(overall_wide)
    lowest=mod.lowest_counts(run)
    overall_lowest=mod.lowest_counts(overall_run)
    outliers=mod.outlier_table(selected_pass,1.5)
    overall_outliers=mod.outlier_table(overall_pass,1.5)
    overview=pd.DataFrame([{'report_mode':mode,'participants':len({x[0] for x in ref}),
        'matched_runs':len(ref),'methods':len(METHODS),'eligible_configurations':grid[['method','configuration_key']].drop_duplicates().shape[0],
        'attempted_grid_solutions_all_methods':len(grid),
        'pass_grid_solutions_all_methods':int(grid.status.eq('PASS').sum()),
        'fail_grid_solutions_all_methods':int(grid.status.eq('FAIL').sum())}])
    outputs={
      '01_selected_configuration_definitions.csv':definitions,
      '02_selected_configuration_method_summary.csv':selected_summary,
      '03_overall_grid_method_summary.csv':overall_summary,
      '04_selected_vs_overall_summary.csv':compare,
      '05_selected_configuration_pairwise_differences.csv':pair,
      '06_overall_grid_pairwise_differences.csv':overall_pair,
      '07_selected_configuration_participant_summary.csv':participant,
      '08_overall_grid_participant_summary.csv':overall_participant,
      '09_selected_configuration_run_summary.csv':run,
      '10_overall_grid_run_summary.csv':overall_run,
      '11_selected_configuration_lowest_error_counts.csv':lowest,
      '12_overall_grid_lowest_typical_counts.csv':overall_lowest,
      '13_selected_configuration_results_long.csv':selected_all,
      '14_selected_configuration_results_wide.csv':wide.reset_index(),
      '15_overall_grid_run_values_long.csv':overall,
      '16_overall_grid_run_values_wide.csv':overall_wide.reset_index(),
      '17_selected_configuration_outliers.csv':outliers,
      '18_overall_grid_outliers.csv':overall_outliers,
      '19_dataset_summary.csv':overview,
      '20_configuration_completeness_audit.csv':audit,
      '21_grid_eligible_rows_by_method.csv':grid.groupby(['method','status']).size().rename('rows').reset_index(),
    }
    for name,df in outputs.items():df.to_csv(tables/name,index=False)
    if not args.no_figures:
        # All Script-17 plotting functions call Figure.savefig. Decorating that
        # call updates visible labels consistently in every PNG, without
        # altering METHOD_ORDER, merges, pivots, or analysis tables.
        from matplotlib.figure import Figure
        original_savefig = Figure.savefig
        def savefig_with_years(fig, *a, **kw):
            add_year_labels_to_figure(fig)
            return original_savefig(fig, *a, **kw)
        Figure.savefig = savefig_with_years
        mod.setup_plot_style()
        context=f'{len(ref)} runs | {len({x[0] for x in ref})} participants | 12 methods | {mode}'
        mod.save_distribution(selected_pass,figures/'01_selected_configuration_distribution.png',
            'Localization error for selected lowest-median observed configurations',context,
            'Localization error (mm)',args.dpi)
        mod.save_distribution(overall_pass,figures/'02_overall_grid_run_distribution.png',
            'Typical localization error across tested parameter grids',context,
            'Within-run median localization error (mm)',args.dpi)
        mod.save_selected_vs_overall(compare,figures/'03_selected_vs_overall_method_medians.png',args.dpi)
        mod.save_pairwise_heatmap(pair,figures/'04_selected_pairwise_median_differences.png',args.dpi)
        mod.save_participant(participant,figures/'05_selected_configuration_by_participant.png',
            'Selected fixed configurations: median error by participant',args.dpi)
        mod.save_lowest_counts(lowest,figures/'06_selected_configuration_lowest_error_counts.png',args.dpi)
        mod.save_participant(overall_participant,figures/'07_overall_grid_by_participant.png',
            'Typical within-grid run values by participant',args.dpi)
        mod.save_run_difficulty(run,figures/'08_selected_run_difficulty.png',args.dpi)
        mod.save_methods_by_run(wide,figures/'09_selected_configuration_by_run.png',
            'Selected fixed configurations: localization error by run',context,
            'Localization error (mm)',args.dpi)
        mod.save_methods_by_run(overall_wide,figures/'10_overall_grid_by_run.png',
            'Overall parameter grids: within-run median localization error',context,
            'Within-run median localization error (mm)',args.dpi)
        plt.close('all')
    print(f'\n{mode}: {len(grid):,} grid rows; {len(definitions)} methods; '
          f'PASS={int(grid.status.eq("PASS").sum()):,}, FAIL={int(grid.status.eq("FAIL").sum()):,}')
    print(compare[['method','selected_median_mm','selected_pass','selected_fail',
                   'overall_median_mm']].to_string(index=False))
    print(f'Wrote: {destination}')


def main():
    args=arguments()
    mod=script17_module(args.script17)
    old=old_grid(args,mod)
    old=prepare_old(old,reference_keys(old.loc[old.method.eq('MNE')]))
    ref=reference_keys(old.loc[old.method.eq('MNE')])
    if len(ref)!=args.expected_runs:
        raise ValueError(f'Original MNE reference contains {len(ref)} runs, expected {args.expected_runs}')
    for mode,path,all_pass in [('complete_any_status',args.new_any,False),('complete_all_pass',args.new_pass,True)]:
        accepted_old,audit_old=eligibility(old,ref,all_pass)
        incoming=load_new(path,NEW,ref,mod)
        if all_pass and not incoming.status.eq('PASS').all():
            raise ValueError('Script 25 all-pass dataset contains failures')
        all_data=pd.concat([accepted_old,incoming],ignore_index=True,sort=False)
        if set(all_data.method)!=set(METHODS):
            raise ValueError(f'Incomplete method set: {set(METHODS)-set(all_data.method)}')
        for meth,group in all_data.groupby('method'):
            if reference_keys(group)!=ref:
                raise ValueError(f'{mode}: {meth} run key mismatch')
        make_report(mod,all_data,audit_old,ref,args.output_root,args,mode)

if __name__=='__main__':main()

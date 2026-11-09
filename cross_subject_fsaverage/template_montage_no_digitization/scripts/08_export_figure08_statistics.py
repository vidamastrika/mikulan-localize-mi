#!/usr/bin/env python3
"""Export no-warp-style 00 statistics for the exact Figure 08 distributions.

Reads Script 06 tables/13_baseline_context_unique_runs.csv, filters strategy
Single-reference and common_run_available, just as comparison_baseline_context
figures() does. Each condition contributes one observation per test run/method;
no-warp/template values are medians over paired successful reference jobs.
Outputs tables/00_figure08_single_reference_statistics.csv with the exact 11
columns of no-warp's 00_master_three_strategies_statistics.csv, rounded to three
places, plus exact long plot values and SHA-256/source/filter/metric provenance.
No inverse, MRI, figure rendering, re-selection or original-table modification.
Command from repository root:
 .venv/bin/python cross_subject_fsaverage/template_montage_no_digitization/scripts/08_export_figure08_statistics.py
For both transfer strategies in one table, add --both-strategies. This creates
72 rows with an explicit transfer_strategy column (LOSO or Single-reference).
Requires NumPy/pandas. Fresh output files only; --overwrite allows this export's
own files to be regenerated. Adds their hashes to the comparison manifest.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

VARIANT=Path(__file__).resolve().parents[1]
ROOT=VARIANT.parents[1]
COLUMNS=['method','strategy','N','Mean_mm','Median_mm','SD_mm','Q1_mm','Q3_mm','IQR_mm','Min_mm','Max_mm']
SERIES=[('Baseline','baseline_localization_error_mm','within_subject_localization_error'),
        ('No-warp','no_warp_run_diagnostic_mm','unregistered_HEAD_distance'),
        ('Template/fsaverage','template_run_diagnostic_mm','unregistered_HEAD_distance')]


def require(condition,message):
    if not condition:raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison-root',type=Path,default=VARIANT/'outputs/06_compare_no_warp')
    parser.add_argument('--overwrite',action='store_true')
    parser.add_argument('--both-strategies',action='store_true',help='Combine Figure 07 LOSO and Figure 08 Single Reference in one table')
    args=parser.parse_args(argv)
    root=args.comparison_root.expanduser().resolve()
    require((VARIANT/'outputs').resolve() in root.parents,'Comparison must be within this variant/outputs')
    manifest_path=root/'comparison_manifest.json';manifest=json.loads(manifest_path.read_text())
    source=root/'tables/13_baseline_context_unique_runs.csv'
    source_sha=sha(source)
    require(source_sha==manifest['output_sha256']['tables/13_baseline_context_unique_runs.csv'],'Source table hash differs')
    data=pd.read_csv(source)
    require(set(data.common_run_available.astype(str).str.lower())<={'true','false'},'Unexpected common-run flag')
    transfers=['LOSO','Single-reference'] if args.both_strategies else ['Single-reference']
    data=data.loc[data.strategy.isin(transfers)&data.common_run_available.astype(str).str.lower().eq('true')].copy()
    require(not data.duplicated(['strategy','method','test_subject','run']).any(),'Duplicate strategy/run/method')
    style_path=root/'figure_method_metadata.csv'
    if style_path.is_file():
        methods=pd.read_csv(style_path).method.tolist()
    else:
        methods=pd.read_csv(root/'method_styles.csv').method.tolist()
    require(len(methods)==12 and set(data.method)==set(methods),'Figure 08 requires all 12 methods')
    schema=ROOT/'cross_subject_no_warp/outputs/09_visualize_results/00_master_three_strategies_statistics.csv'
    require(pd.read_csv(schema,nrows=0).columns.tolist()==COLUMNS,'Original no-warp 00 schema changed')
    stats=[];long=[]
    for transfer in transfers:
        for method in methods:
            group=data.loc[data.strategy.eq(transfer)&data.method.eq(method)]
            require(len(group)>0,'Missing method/transfer observations')
            for condition,column,metric in SERIES:
                values=pd.to_numeric(group[column],errors='raise').to_numpy(float)
                require(np.isfinite(values).all() and np.all(values>=0),'Invalid plot values')
                q1,q3=np.quantile(values,[.25,.75])
                record=dict(method=method,strategy=condition,N=len(values),Mean_mm=float(values.mean()),
                    Median_mm=float(np.median(values)),SD_mm=float(values.std(ddof=1)) if len(values)>1 else np.nan,
                    Q1_mm=float(q1),Q3_mm=float(q3),IQR_mm=float(q3-q1),Min_mm=float(values.min()),Max_mm=float(values.max()))
                if args.both_strategies:record['transfer_strategy']=transfer
                stats.append(record)
                for row,value in zip(group.to_dict('records'),values):
                    record=dict(method=method,strategy=condition,test_subject=row['test_subject'],run=row['run'],
                        value_mm=float(value),metric=metric,
                        n_paired_references=int(row['n_paired']) if condition!='Baseline' else 1)
                    if args.both_strategies:record['transfer_strategy']=transfer
                    long.append(record)
    if args.both_strategies:
        stats_path=root/'tables/00_master_three_conditions_statistics.csv'
        values_path=root/'tables/00_master_three_conditions_plot_values.csv'
        audit_path=root/'tables/00_master_three_conditions_statistics_manifest.json'
    else:
        stats_path=root/'tables/00_figure08_single_reference_statistics.csv'
        values_path=root/'tables/00_figure08_plot_values.csv'
        audit_path=root/'tables/00_figure08_statistics_manifest.json'
    for path in [stats_path,values_path,audit_path]:
        require(not path.exists() or args.overwrite,f'Export exists: {path}; use --overwrite')
    require(sha(source)==source_sha,'Source changed during export')
    pd.DataFrame(stats,columns=(['method','transfer_strategy']+COLUMNS[1:] if args.both_strategies else COLUMNS)).to_csv(stats_path,index=False,float_format='%.3f')
    pd.DataFrame(long).to_csv(values_path,index=False)
    audit=dict(created_utc=datetime.now(timezone.utc).isoformat(),figures=(['figures/07_loso_baseline_context.png','figures/08_single-reference_baseline_context.png'] if args.both_strategies else ['figures/08_single-reference_baseline_context.png']),
        source_csv=str(source),source_sha256=source_sha,
        filter=f"strategy in {transfers} and common_run_available == True",
        conditions={label:column for label,column,_ in SERIES},schema_source=str(schema),schema_sha256=sha(schema),
        export_script_sha256=sha(Path(__file__)),statistics_rows=len(stats),plot_value_rows=len(long),
        SD_definition='sample SD, ddof=1',quartiles='NumPy linear percentile convention',CSV_decimal_places=3,
        N_definition='unique test-run observations per method/condition after paired-reference median reduction, not reference jobs',
        baseline_metric='within-subject localization error; in-sample selected configurations',
        other_metrics='unregistered HEAD-coordinate distances; not validated anatomical errors',
        output_sha256={str(p.relative_to(root)):sha(p) for p in [stats_path,values_path]})
    audit_path.write_text(json.dumps(audit,indent=2,allow_nan=False)+'\n')
    for path in [stats_path,values_path,audit_path]:manifest['output_sha256'][str(path.relative_to(root))]=sha(path)
    manifest['figure07_08_statistics' if args.both_strategies else 'figure08_statistics']=dict(statistics_table=str(stats_path.relative_to(root)),
        exact_plot_values=str(values_path.relative_to(root)),provenance=str(audit_path.relative_to(root)),
        schema='no-warp 00 statistics columns; combined mode adds transfer_strategy')
    manifest_path.write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    print(f'Wrote {len(stats)} rows: {stats_path}')
    print(f'Exact plot values ({len(long)} rows): {values_path}')
    print('Figures, existing tables, metric definitions and reconstruction are unchanged.')


if __name__=='__main__':
    main()

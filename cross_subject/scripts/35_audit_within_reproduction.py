#!/usr/bin/env python3
"""Read-only audit of Script 35's 61 within-subject reconstructions against the original all-PASS grid.

Run from the repository root:
 python cross_subject/scripts/35_audit_within_reproduction.py \
   --grid outputs/15_full_additional_method_grid/grid_results.csv \
   --selected outputs/16_mxne_selected/tables/10_selected_configurations.csv

Multiple --grid arguments are allowed. Provide the grid containing the 61 selected
runs for Continuous-ECD, RAP-MUSIC, LCMV and MxNE. No inverse fits are rerun.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

METHODS=('Continuous-ECD','RAP-MUSIC','LCMV','MxNE')
EXPECTED={'Continuous-ECD':('128',15.,None,None,None),
          'RAP-MUSIC':('128',None,1.,None,None),
          'LCMV':('32',None,None,0.,None),
          'MxNE':('128',None,None,None,35.)}

def main():
 ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
 ap.add_argument('--grid',type=Path,action='append',required=True,help='Original Script 15 grid_results.csv (repeat if split)')
 ap.add_argument('--selected',type=Path,default=None,help='10_selected_configurations.csv, with alpha35 for MxNE')
 ap.add_argument('--execution',type=Path,default=Path('cross_subject/outputs/35_cross_subject_ecd_music_lcmv_mxne/01_execution_status.csv'))
 ap.add_argument('--evaluation',type=Path,default=Path('cross_subject/outputs/35_cross_subject_ecd_music_lcmv_mxne/02_localization_errors.csv'))
 ap.add_argument('--output',type=Path,default=Path('cross_subject/outputs/35_cross_subject_ecd_music_lcmv_mxne/audit'))
 ap.add_argument('--distance-tolerance-mm',type=float,default=.05)
 args=ap.parse_args()
 for f in args.grid+[args.execution,args.evaluation]+([args.selected] if args.selected else []):
  if not f.is_file():ap.error(f'Missing input: {f}')
 selected={
 'Continuous-ECD':{'montage':'128','ecd_min_dist_mm':15},
 'RAP-MUSIC':{'montage':'128','rap_music_n_dipoles':1},
 'LCMV':{'montage':'32','lcmv_reg':0,'lcmv_window_half_ms':10,
         'lcmv_data_covariance_method':'diagonal_fixed','lcmv_weight_norm':'unit-noise-gain-invariant'},
 'MxNE':{'montage':'128','mxne_alpha':35,'mxne_loose':.7,'mxne_depth':1},
 }
 if args.selected:
  tab=pd.read_csv(args.selected)
  for meth,cfg in selected.items():
   sel=tab[tab.method.astype(str)==meth]
   if len(sel)!=1:ap.error(f'Expected exactly one selected configuration for {meth}, got {len(sel)}')
   row=sel.iloc[0]
   for col,value in cfg.items():
    if col not in row:ap.error(f'{meth}: selection file missing {col}')
    found=row[col]
    if col=='montage' or isinstance(value,str):
     if str(found)!=str(value):ap.error(f'{meth}: selection differs for {col}: {found} != {value}')
    elif not np.isclose(float(found),float(value)):
     ap.error(f'{meth}: selection differs for {col}: {found} != {value}')
   if not (int(row['runs_pass'])==61 and int(row['runs_fail'])==0):
    ap.error(f'{meth}: selected config not 61/61 PASS')

 execution=pd.read_csv(args.execution,low_memory=False)
 ev=pd.read_csv(args.evaluation,low_memory=False)
 own=execution[execution.is_within.astype(str).str.lower()=='true'].copy()
 if len(own)!=244 or own[['method','test_subject','run']].duplicated().any():
  ap.error('Script 35 must contain exactly 244 unique subject-specific jobs')
 evbase=ev.drop_duplicates(['method','test_subject','run'])[['method','test_subject','run','within_subject_error_mm']]
 if len(evbase)!=244:ap.error('Evaluation missing unique 244 subject-specific baselines')
 own=own.merge(evbase,on=['method','test_subject','run'],validate='one_to_one')
 grids=[]
 for p in args.grid:
  g=pd.read_csv(p,low_memory=False)
  must=['method','subject','run','status','montage','localization_distance_mm']
  if set(must)-set(g):ap.error(f'Original grid missing {set(must)-set(g)}: {p}')
  g=g[g.method.isin(METHODS)].copy()
  g['source_grid']=str(p)
  grids.append(g)
 grid=pd.concat(grids,ignore_index=True)
 # Select by *all* method-specific parameter values to avoid selecting 61 runs from wrong configuration.
 selection=[]
 for method,cfg in selected.items():
  g=grid[grid.method==method].copy()
  for col,value in cfg.items():
   if col not in g:ap.error(f'Grid missing {col} for {method}')
   if isinstance(value,str):
    g=g[g[col].astype(str)==value]
   else:
    g=g[np.isclose(pd.to_numeric(g[col],errors='coerce'),value,atol=1e-8,rtol=0)]
  if len(g)!=61 or g[['subject','run']].duplicated().any():
   ap.error(f'{method}: expected 61 unique original grid rows for frozen config; found {len(g)}; check --grid files')
  if not g.status.eq('PASS').all():ap.error(f'{method}: selected grid has failures')
  selection.append(g)
 original=pd.concat(selection,ignore_index=True)
 joined=own.merge(original,left_on=['method','test_subject','run'],right_on=['method','subject','run'],validate='one_to_one',suffixes=('_script35','_original'))
 joined['script35_error_mm']=pd.to_numeric(joined.within_subject_error_mm,errors='coerce')
 joined['original_error_mm']=pd.to_numeric(joined.localization_distance_mm,errors='coerce')
 joined['error_delta_mm']=joined.script35_error_mm-joined.original_error_mm
 joined['abs_error_delta_mm']=joined.error_delta_mm.abs()
 for axis in 'xyz':
  old_col=f'selected_{axis}_mm'
  if old_col in joined:
   joined[f'coord_delta_{axis}_mm']=pd.to_numeric(joined[f'reference_head_{axis}_mm'],errors='coerce')-pd.to_numeric(joined[old_col],errors='coerce')
 # Note: old selected XYZ is likely surface/native, new reference head XYZ is HEAD.
 # They are not directly comparable, so never use coordinate deltas as spatial mismatch evidence.
 joined['peak_time_delta_ms']=np.nan
 if 'selected_time_ms' in joined:
  joined['peak_time_delta_ms']=pd.to_numeric(joined.peak_time_s,errors='coerce')*1000-pd.to_numeric(joined.selected_time_ms,errors='coerce')
 joined['match_within_tolerance']=joined.abs_error_delta_mm<=args.distance_tolerance_mm
 summary=[]
 for m,g in joined.groupby('method'):
  summary.append(dict(method=m,n=len(g),within_script35_median_mm=g.script35_error_mm.median(),
      original_grid_median_mm=g.original_error_mm.median(),
      median_difference_mm=g.script35_error_mm.median()-g.original_error_mm.median(),
      mean_absolute_paired_difference_mm=g.abs_error_delta_mm.mean(),
      max_absolute_paired_difference_mm=g.abs_error_delta_mm.max(),
      runs_with_difference_gt_tolerance=int((~g.match_within_tolerance).sum()),
      tolerance_mm=args.distance_tolerance_mm))
 args.output.mkdir(parents=True,exist_ok=True)
 joined.sort_values(['method','test_subject','run']).to_csv(args.output/'01_run_by_run_reproduction_audit.csv',index=False)
 report=pd.DataFrame(summary).sort_values('method')
 report.to_csv(args.output/'02_audit_summary.csv',index=False)
 report.to_string(index=False,float_format=lambda x:f'{x:.4f}')
 print(report.to_string(index=False,float_format=lambda x:f'{x:.4f}'))
 print('\nSaved:',args.output)
 print('CAUTION: reference_head coordinates and original grid selected coordinates are not guaranteed to share a coordinate frame.')
 if not joined.match_within_tolerance.all():
  print('AUDIT FLAG: At least one paired within-subject result differs; investigate original fit/scoring before final comparison.')

if __name__=='__main__': main()
